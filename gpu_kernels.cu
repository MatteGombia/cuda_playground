#include "gpu_kernels.h"
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/count.h>
#include <thrust/partition.h>
#include <iostream>


// ------------------------------------------------------------------
// KERNEL 1: Calculate Covariance Matrix Statistics
// ------------------------------------------------------------------
__global__ void computeCovarianceStats(
    const Point* __restrict__ points,
    const bool* __restrict__ is_ground,
    int num_points,
    bool use_mask,
    int limit_count,
    float* __restrict__ global_sums 
) {
    // 1. Setup Shared Memory
    __shared__ float s_sums[10];
    
    int tid = threadIdx.x;
    if (tid < 10) s_sums[tid] = 0.0f;
    __syncthreads();

    // 2. Main Loop
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        bool include_point = false;
        if (!use_mask) {
            if (idx < limit_count) include_point = true;
        } else {
            if (is_ground[idx]) include_point = true;
        }

        if (include_point) {
            Point p = points[idx];
            atomicAdd(&s_sums[0], p.x);
            atomicAdd(&s_sums[1], p.y);
            atomicAdd(&s_sums[2], p.z);
            atomicAdd(&s_sums[3], p.x * p.x);
            atomicAdd(&s_sums[4], p.x * p.y);
            atomicAdd(&s_sums[5], p.x * p.z);
            atomicAdd(&s_sums[6], p.y * p.y);
            atomicAdd(&s_sums[7], p.y * p.z);
            atomicAdd(&s_sums[8], p.z * p.z);
            atomicAdd(&s_sums[9], 1.0f);
        }
    }
    __syncthreads();

    // 3. Write to Global Memory
    if (tid < 10) {
        atomicAdd(&global_sums[tid], s_sums[tid]);
    }
}

// ------------------------------------------------------------------
// KERNEL 2: Classify Points
// ------------------------------------------------------------------
__global__ void classifyPointsKernel(
    const Point* __restrict__ points,
    bool* __restrict__ is_ground,
    int num_points,
    const float4* __restrict__ plane_ptr, 
    float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    // Read the plane from GPU memory
    float4 plane = *plane_ptr; 

    Point p = points[idx];
    float dist = fabsf(plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w);
    
    is_ground[idx] = (dist < threshold);
}

// ------------------------------------------------------------------
// HELPER: Solve Eigen System on CPU
// ------------------------------------------------------------------
// void solvePlaneCPU(float* d_sums, float4& plane, cudaStream_t stream) {
//     float h_sums[10];
//     // Must use Async copy to keep stream sync correct
//     cudaMemcpyAsync(h_sums, d_sums, 10 * sizeof(float), cudaMemcpyDeviceToHost, stream);
//     cudaStreamSynchronize(stream); // CPU needs this data NOW to do math

//     float N = h_sums[9];
//     if (N < 3) return; 

//     float3 centroid = {h_sums[0]/N, h_sums[1]/N, h_sums[2]/N};

//     // Construct Covariance 3x3
//     float cov[3][3];
//     cov[0][0] = h_sums[3]/N - centroid.x*centroid.x;
//     cov[0][1] = h_sums[4]/N - centroid.x*centroid.y;
//     cov[0][2] = h_sums[5]/N - centroid.x*centroid.z;
//     cov[1][1] = h_sums[6]/N - centroid.y*centroid.y;
//     cov[1][2] = h_sums[7]/N - centroid.y*centroid.z;
//     cov[2][2] = h_sums[8]/N - centroid.z*centroid.z;
//     cov[1][0] = cov[0][1]; cov[2][0] = cov[0][2]; cov[2][1] = cov[1][2];

//     // Placeholder: Assume Z-up normal for simplicity (Replace with Eigen solver)
//     float3 normal = {0, 0, 1}; 
    
//     float d_val = -(normal.x * centroid.x + normal.y * centroid.y + normal.z * centroid.z);
//     plane = make_float4(normal.x, normal.y, normal.z, d_val);
// }

// Helper: 3x3 Determinant
__device__ inline float det3x3(float xx, float xy, float xz, 
                               float yy, float yz, float zz) {
    return xx * (yy * zz - yz * yz) - 
           xy * (xy * zz - yz * xz) + 
           xz * (xy * yz - yy * xz);
}

// Device function to solve 3x3 Eigenvalues/Vectors for the smallest eigenvalue
__global__
void solvePlaneDevice(float* sums, float4* plane) {
    if (threadIdx.x != 0) return;
    
    float N = sums[9];
    if (N < 3) {
        // Not enough points to form a plane
        *plane = make_float4(0, 0, 1, 0); 
        return;
    }

    // 1. Compute Centroid
    float3 centroid = {sums[0]/N, sums[1]/N, sums[2]/N};

    // 2. Construct Covariance Matrix (Upper Triangle of Symmetric Matrix)
    // Var(X) = E[X^2] - E[X]^2
    float xx = sums[3]/N - centroid.x * centroid.x;
    float xy = sums[4]/N - centroid.x * centroid.y;
    float xz = sums[5]/N - centroid.x * centroid.z;
    float yy = sums[6]/N - centroid.y * centroid.y;
    float yz = sums[7]/N - centroid.y * centroid.z;
    float zz = sums[8]/N - centroid.z * centroid.z;

    // -----------------------------------------------------------
    // 3. ANALYTIC EIGENVALUE SOLVER (Robust 3x3)
    // -----------------------------------------------------------
    
    // Scale the matrix by its max element to avoid numerical issues with small/large coordinates
    float scale = fmaxf(fabsf(xx), fmaxf(fabsf(yy), fabsf(zz)));
    if (scale < 1e-6f) {
        // Variance is zero (points are all in one spot)
        *plane = make_float4(0, 0, 1, -centroid.z);
        return;
    }

    // Normalize matrix
    xx /= scale; xy /= scale; xz /= scale;
    yy /= scale; yz /= scale; zz /= scale;

    // Characteristic Equation: lambda^3 + a*lambda^2 + b*lambda + c = 0
    // For a symmetric matrix M:
    // a = -trace(M)
    // b = 0.5 * (trace(M)^2 - trace(M^2))
    // c = -det(M)
    
    float trace_A = xx + yy + zz;
    float trace_A2 = (xx*xx + xy*xy + xz*xz) + 
                     (xy*xy + yy*yy + yz*yz) + 
                     (xz*xz + yz*yz + zz*zz);
    
    float a = -trace_A;
    float b = 0.5f * (trace_A * trace_A - trace_A2);
    float c = -det3x3(xx, xy, xz, yy, yz, zz);

    // Solve cubic equation using trigonometric method
    float p = b - a * a / 3.0f;
    float q = 2.0f * a * a * a / 27.0f - a * b / 3.0f + c;
    float p3 = p * p * p;
    float D = 4.0f * p3 + 27.0f * q * q;

    // Smallest eigenvalue (min_lambda)
    float min_lambda;
    if (D >= 0) {
        // This case is rare for covariance matrices (usually 3 real roots),
        // but can happen if eigenvalues are identical. 
        // Fallback to a safe small value or 0.
        min_lambda = 0.0f; 
    } else {
        float r = sqrtf(-4.0f * p / 3.0f);
        float phi = acosf(-4.0f * q / (r * r * r)) / 3.0f;
        // The roots are sorted or periodic. For covariance matrices,
        // the smallest root is typically found at this offset:
        min_lambda = r * cosf(phi + 2.0f * 3.14159265f / 3.0f) - a / 3.0f;
    }

    // -----------------------------------------------------------
    // 4. COMPUTE EIGENVECTOR (Normal Vector)
    // Solve (M - min_lambda * I) * v = 0
    // The eigenvector is the cross product of any two independent rows of (M - lambda*I)
    // -----------------------------------------------------------
    
    float l = min_lambda;
    float3 r0 = {xx - l, xy,     xz};
    float3 r1 = {xy,     yy - l, yz};
    float3 r2 = {xz,     yz,     zz - l};

    // Calculate cross products of rows
    float3 v0 = {r0.y * r1.z - r0.z * r1.y,  r0.z * r1.x - r0.x * r1.z,  r0.x * r1.y - r0.y * r1.x};
    float3 v1 = {r0.y * r2.z - r0.z * r2.y,  r0.z * r2.x - r0.x * r2.z,  r0.x * r2.y - r0.y * r2.x};
    float3 v2 = {r1.y * r2.z - r1.z * r2.y,  r1.z * r2.x - r1.x * r2.z,  r1.x * r2.y - r1.y * r2.x};

    // Pick the most robust (longest) vector to avoid precision loss if rows are nearly parallel
    float d0 = v0.x*v0.x + v0.y*v0.y + v0.z*v0.z;
    float d1 = v1.x*v1.x + v1.y*v1.y + v1.z*v1.z;
    float d2 = v2.x*v2.x + v2.y*v2.y + v2.z*v2.z;

    float3 normal;
    float norm_sq;
    if (d0 >= d1 && d0 >= d2) { normal = v0; norm_sq = d0; }
    else if (d1 >= d2)        { normal = v1; norm_sq = d1; }
    else                      { normal = v2; norm_sq = d2; }

    // Normalize
    if (norm_sq < 1e-6f) {
        normal = make_float3(0, 0, 1); // Fallback
    } else {
        float inv_norm = rsqrtf(norm_sq);
        normal.x *= inv_norm;
        normal.y *= inv_norm;
        normal.z *= inv_norm;
    }

    // Ensure normal points "Up" (positive Z)
    // This is standard for ground plane segmentation
    if (normal.z < 0) {
        normal.x = -normal.x;
        normal.y = -normal.y;
        normal.z = -normal.z;
    }

    // 5. Compute 'd' in ax + by + cz + d = 0
    // d = -dot(normal, centroid)
    float d_val = -(normal.x * centroid.x + normal.y * centroid.y + normal.z * centroid.z);

    *plane = make_float4(normal.x, normal.y, normal.z, d_val);
}

// ------------------------------------------------------------------
// PROCESS ONE BATCH (Runs on a specific Stream)
// ------------------------------------------------------------------
// This replaces your 'estimateGroundCUDA' kernel. It is a HOST function
// that launches kernels into a specific stream.
void processBatch(
    thrust::device_vector<Point>& d_points,
    thrust::device_vector<bool>& d_is_ground,
    float* d_sums,
    int num_iter,
    int num_lpr,
    float th_dist,
    cudaStream_t stream
) {
    int num_points = d_points.size();
    if (num_points == 0) return;

    // Use Thrust with Execution Policy to bind to the stream
    auto policy = thrust::cuda::par.on(stream);

    // 1. Sort for seeds (Async on stream)
    thrust::sort(policy, d_points.begin(), d_points.end(), CompareZ());

    int threads = 256;
    int blocks = (num_points + threads - 1) / threads;
    float4* d_plane;
    cudaMallocAsync(&d_plane, sizeof(float4), stream);

    for (int i = 0; i < num_iter; i++) {
        // Reset sums (Async)
        cudaMemsetAsync(d_sums, 0, 10 * sizeof(float), stream);

        // Kernel 1: Stats
        // Note: 10 * sizeof(float) is the shared mem size passed as 3rd arg
        computeCovarianceStats<<<blocks, threads, 10*sizeof(float), stream>>>(
            thrust::raw_pointer_cast(d_points.data()),
            thrust::raw_pointer_cast(d_is_ground.data()),
            num_points,
            (i > 0), // use_mask
            num_lpr,
            d_sums
        );

        // Solve on CPU (This forces a sync on this stream only)
        // solvePlaneCPU(d_sums, plane, stream);

        // 2. Solve Plane (GPU)
        solvePlaneDevice<<<1, 1, 0, stream>>>(d_sums, d_plane);

        // Kernel 3: Classify
        classifyPointsKernel<<<blocks, threads, 0, stream>>>(
            thrust::raw_pointer_cast(d_points.data()),
            thrust::raw_pointer_cast(d_is_ground.data()),
            num_points,
            d_plane,
            th_dist
        );
    }
}

// ------------------------------------------------------------------
// MAIN COMPUTATION (Entry Point)
// ------------------------------------------------------------------
void compute(
    const std::vector<Point>& h_src_cloud, 
    std::vector<Point>& h_ground, 
    std::vector<Point>& h_nonground,
    int num_iter, 
    int num_lpr, 
    float th_dist
) {
    int total_points = h_src_cloud.size();
    
    // 1. Upload All Points
    thrust::device_vector<Point> d_all_points = h_src_cloud;

    auto t1 = std::chrono::steady_clock::now();
    // 2. Split Points 
    // We re-order the vector so Part 1 is at the front, Part 2 at the back
    // 1. Perform the remove (this just moves bad points to the end)
    auto new_end = thrust::remove_if(thrust::device, d_all_points.begin(), d_all_points.end(), 
        [] __device__ (Point p) { 
            return p.z < -0.1; 
        });

    // 2. Calculate the new size (Iterator math)
    total_points = new_end - d_all_points.begin();

    // 3. Resize the vector to actually delete the garbage data at the end
    d_all_points.resize(total_points);
    
    auto split_iter = thrust::partition(d_all_points.begin(), d_all_points.end(), IsPatch1());
    
    // Calculate sizes
    int size1 = split_iter - d_all_points.begin();
    int size2 = total_points - size1;

    // Create sub-vectors (Views would be better, but copies are safer for now)
    thrust::device_vector<Point> d_batch1(d_all_points.begin(), split_iter);
    thrust::device_vector<Point> d_batch2(split_iter, d_all_points.end());
    
    thrust::device_vector<bool> d_mask1(size1);
    thrust::device_vector<bool> d_mask2(size2);

    // 3. Create Streams and Buffers
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    float *d_sums1, *d_sums2;
    cudaMalloc(&d_sums1, 10 * sizeof(float));
    cudaMalloc(&d_sums2, 10 * sizeof(float));

    // 4. Launch Work
    // These functions return immediately after queuing commands
    auto t3 = std::chrono::steady_clock::now();
    processBatch(d_batch1, d_mask1, d_sums1, num_iter, num_lpr, th_dist, stream1);
    auto t4 = std::chrono::steady_clock::now();
    processBatch(d_batch2, d_mask2, d_sums2, num_iter, num_lpr, th_dist, stream2);
    auto t5 = std::chrono::steady_clock::now();
    cudaDeviceSynchronize();
    auto t2 = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t4 - t3);
    std::cout << "call of func in: " << duration.count() << " ms" << std::endl;
    duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t5 - t4);
    std::cout << "call func2 in: " << duration.count() << " ms" << std::endl;
    duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t2 - t1);
    std::cout << "computation in: " << duration.count() << " ms" << std::endl;

    // 5. Gather Results (Sync happens implicitly here)
    h_ground.clear();
    h_nonground.clear();

    // Copy batch 1 results
    // (We use a simple host download here for clarity)
    thrust::host_vector<Point> h_b1 = d_batch1;
    thrust::host_vector<bool> h_m1 = d_mask1;
    for(int i=0; i<size1; i++) {
        if(h_m1[i]) h_ground.push_back(h_b1[i]);
        else h_nonground.push_back(h_b1[i]);
    }

    // Copy batch 2 results
    thrust::host_vector<Point> h_b2 = d_batch2;
    thrust::host_vector<bool> h_m2 = d_mask2;
    for(int i=0; i<size2; i++) {
        if(h_m2[i]) h_ground.push_back(h_b2[i]);
        else h_nonground.push_back(h_b2[i]);
    }

    // Cleanup
    cudaFree(d_sums1);
    cudaFree(d_sums2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}