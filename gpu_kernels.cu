#include  "gpu_kernels.h"



// ------------------------------------------------------------------
// KERNEL 1: Calculate Covariance Matrix Statistics
// ------------------------------------------------------------------
// If 'use_mask' is true, we only consider points where is_ground[i] == true.
// If 'use_mask' is false, we use the first 'limit_count' points (for initial seeds).
__global__ void computeCovarianceStats(
    const Point* __restrict__ points,
    const bool* __restrict__ is_ground,
    int num_points,
    bool use_mask,
    int limit_count,
    float* __restrict__ global_sums // [0..2]=sum_xyz, [3..8]=sum_cov_terms, [9]=count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    bool include_point = false;

    if (!use_mask) {
        // Initial Step: Just take the first N points (assumed sorted)
        if (idx < limit_count) include_point = true;
    } else {
        // Iterative Step: Check the mask
        if (is_ground[idx]) include_point = true;
    }

    if (include_point) {
        Point p = points[idx];
        
        // Accumulate Sums for Centroid
        atomicAdd(&global_sums[0], p.x);
        atomicAdd(&global_sums[1], p.y);
        atomicAdd(&global_sums[2], p.z);

        // Accumulate Products for Covariance (xx, xy, xz, yy, yz, zz)
        atomicAdd(&global_sums[3], p.x * p.x);
        atomicAdd(&global_sums[4], p.x * p.y);
        atomicAdd(&global_sums[5], p.x * p.z);
        atomicAdd(&global_sums[6], p.y * p.y);
        atomicAdd(&global_sums[7], p.y * p.z);
        atomicAdd(&global_sums[8], p.z * p.z);

        // Count valid points
        atomicAdd(&global_sums[9], 1.0f);
    }
}

// ------------------------------------------------------------------
// KERNEL 2: Classify Points (Distance Check)
// ------------------------------------------------------------------
__global__ void classifyPointsKernel(
    const Point* __restrict__ points,
    bool* __restrict__ is_ground,
    int num_points,
    float a, float b, float c, float d, // Plane coefficients
    float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    Point p = points[idx];
    
    // Distance from point to plane: |ax + by + cz + d| / sqrt(a^2+b^2+c^2)
    // We assume the normal (a,b,c) is already normalized to length 1.
    float dist = fabsf(a * p.x + b * p.y + c * p.z + d);

    if (dist < threshold) {
        is_ground[idx] = true;
    } else {
        is_ground[idx] = false;
    }
}

// ------------------------------------------------------------------
// HELPER: Solve Eigen System on CPU
// ------------------------------------------------------------------
void solvePlaneCPU(float* gpu_sums, float4& plane, float3& centroid) {
    // Copy sums from GPU to CPU
    float cpu_sums[10];
    cudaMemcpy(cpu_sums, gpu_sums, 10 * sizeof(float), cudaMemcpyDeviceToHost);

    float N = cpu_sums[9];
    if (N < 3) return; // Not enough points

    // Calculate Centroid
    centroid.x = cpu_sums[0] / N;
    centroid.y = cpu_sums[1] / N;
    centroid.z = cpu_sums[2] / N;

    // Construct Covariance Matrix
    // E[XX] - E[X]*E[X]
    float cov[3][3];
    cov[0][0] = cpu_sums[3]/N - centroid.x*centroid.x; // xx
    cov[0][1] = cpu_sums[4]/N - centroid.x*centroid.y; // xy
    cov[0][2] = cpu_sums[5]/N - centroid.x*centroid.z; // xz
    cov[1][0] = cov[0][1];
    cov[1][1] = cpu_sums[6]/N - centroid.y*centroid.y; // yy
    cov[1][2] = cpu_sums[7]/N - centroid.y*centroid.z; // yz
    cov[2][0] = cov[0][2];
    cov[2][1] = cov[1][2];
    cov[2][2] = cpu_sums[8]/N - centroid.z*centroid.z; // zz

    // Simple Power Iteration or Closed Form to find Smallest Eigenvector
    // For simplicity, let's assume Up-vector domination or use a library like Eigen here.
    // ** Placeholder for actual Eigen Solver **
    // Let's assume we found the normal:
    float3 normal = make_float3(0, 0, 1); 

    // REAL IMPLEMENTATION TIP: Use a standard Eigen 3x3 solver here. 
    // Since this runs once per iteration on CPU, it is negligible.
    
    // Calculate D: ax + by + cz + d = 0  => d = -dot(n, centroid)
    float d_val = -(normal.x * centroid.x + normal.y * centroid.y + normal.z * centroid.z);
    
    plane = make_float4(normal.x, normal.y, normal.z, d_val);
}

// ------------------------------------------------------------------
// MAIN FUNCTION
// ------------------------------------------------------------------
void estimateGroundCUDA(
    const std::vector<Point>& h_src_cloud, 
    std::vector<Point>& h_ground, 
    std::vector<Point>& h_nonground,
    int num_iter, 
    int num_lpr, // Number of initial seeds (Lowest Point Representative)
    float th_dist
) {
    int num_points = h_src_cloud.size();

    // 1. Move Data to GPU
    thrust::device_vector<Point> d_points = h_src_cloud;
    thrust::device_vector<bool> d_is_ground(num_points, false);
    
    // Allocate memory for reduction sums (10 floats)
    float* d_sums;
    cudaMalloc(&d_sums, 10 * sizeof(float));


    auto t1 = std::chrono::steady_clock::now();
    // 2. Extract Initial Seeds
    // Sort entire cloud by Z. 
    // Note: If you want to preserve the original order, you should sort an index array instead.
    // But since Patchwork outputs new vectors anyway, sorting the points is fine.
    thrust::sort(d_points.begin(), d_points.end(), CompareZ());

    float4 plane = make_float4(0,0,1,0);
    float3 centroid = make_float3(0,0,0);
    
    int threads = 256;
    int blocks = (num_points + threads - 1) / threads;

    // --------------------------------------------------------------
    // ITERATIVE LOOP
    // --------------------------------------------------------------
    for (int i = 0; i < num_iter; i++) {
        
        // Reset sums
        cudaMemset(d_sums, 0, 10 * sizeof(float));

        // Step A: Calculate Covariance
        // Iteration 0: Use first 'num_lpr' points (Seed step)
        // Iteration >0: Use 'd_is_ground' mask
        bool use_mask = (i > 0);
        
        computeCovarianceStats<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_points.data()),
            thrust::raw_pointer_cast(d_is_ground.data()),
            num_points,
            use_mask,
            num_lpr,
            d_sums
        );

        // Step B: Solve Plane on CPU
        solvePlaneCPU(d_sums, plane, centroid);

        // Step C: Classify Points (Distance Check)
        // Check distance for ALL points against the new plane
        classifyPointsKernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_points.data()),
            thrust::raw_pointer_cast(d_is_ground.data()),
            num_points,
            plane.x, plane.y, plane.z, plane.w,
            th_dist
        );
    }

    auto t2 = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t2 - t1);
    std::cout << "conversione in: " << duration.count() << " ms" << std::endl;

    // --------------------------------------------------------------
    // OUTPUT GATHERING
    // --------------------------------------------------------------
    // Use thrust::copy_if to separate Ground and Non-Ground
    h_ground.clear();
    h_nonground.clear();


    /* ----------------- CPU Friendly ------------ */
    int num_ground = thrust::count(d_is_ground.begin(), d_is_ground.end(), true);
    int num_nonground = num_points - num_ground;

    thrust::device_vector<Point> d_ground_temp(num_ground);
    thrust::device_vector<Point> d_nonground_temp(num_nonground);

    /* ------------------------------------------- */

    // Copy back to host to split (or do it on device if next steps are on device)
    // Here we download everything to simplify the example

    thrust::copy_if(d_points.begin(), d_points.end(), d_is_ground.begin(), d_ground_temp.begin(), is_ground());
    thrust::remove_copy_if(d_points.begin(), d_points.end(), d_is_ground.begin(), d_nonground_temp.begin(), is_ground());

    /*thrust::host_vector<Point> h_points_sorted = d_points;
    thrust::host_vector<bool> h_mask = d_is_ground;

    for(size_t i=0; i<h_points_sorted.size(); ++i) {
        if(h_mask[i]) {
            h_ground.push_back(h_points_sorted[i]);
        } else {
            h_nonground.push_back(h_points_sorted[i]);
        }
    }*/

    /* ----------------- CPU Friendly ------------ */
    h_ground.resize(num_ground);
    h_nonground.resize(num_nonground);

    // 5. Bulk Copy to CPU (Memcpy)
    thrust::copy(d_ground_temp.begin(), d_ground_temp.end(), h_ground.begin());
    thrust::copy(d_nonground_temp.begin(), d_nonground_temp.end(), h_nonground.begin());
    /* ------------------------------------------- */
    cudaFree(d_sums);
}