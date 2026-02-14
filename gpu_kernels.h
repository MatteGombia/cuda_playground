#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <iostream>


// ------------------------------------------------------------------
// DATA STRUCTURES
// ------------------------------------------------------------------
struct Point {
    float x, y, z;
};

// Functor for sorting points by Z-value (for Seed Extraction)
struct CompareZ {
    __host__ __device__ bool operator()(const Point& a, const Point& b) {
        return a.z < b.z;
    }
};

struct is_ground
{
  __host__ __device__
  bool operator()(const bool x)
  {
    return x;
  }
};

struct IsPatch1
{
  __host__ __device__
  bool operator()(const Point p)
  {
    return p.x > 0;
  }
};


// ------------------------------------------------------------------
// KERNEL 1: Calculate Covariance Matrix Statistics
// ------------------------------------------------------------------
void compute(
    const std::vector<Point>& h_src_cloud, 
    std::vector<Point>& h_ground, 
    std::vector<Point>& h_nonground,
    int num_iter, 
    int num_lpr, // Number of initial seeds (Lowest Point Representative)
    float th_dist
);

void estimateGroundCUDA(
    const std::vector<Point>& h_src_cloud, 
    std::vector<Point>& h_ground, 
    std::vector<Point>& h_nonground,
    int num_iter, 
    int num_lpr, // Number of initial seeds (Lowest Point Representative)
    float th_dist,
    int* num_ground
);