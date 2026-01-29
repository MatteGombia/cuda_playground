cmake_minimum_required(VERSION 3.18) # 3.18+ is recommended for modern CUDA support
project(CudaPlaneFitting LANGUAGES CXX CUDA)

# 1. Set C++ Standard (C++14 or C++17 is typical for modern ROS/Robotics)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_C_COMPILER /usr/bin/gcc-10)
set(CMAKE_CXX_COMPILER /usr/bin/g++-10)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-10)

# 2. Find CUDA Toolkit
# This is necessary to get the CUDA runtime and helper libraries
find_package(CUDAToolkit REQUIRED)

# 3. Add Your Executable
# Note: Source files ending in .cu are automatically handled by nvcc
add_executable(plane_fitting
    main.cpp       # Your main entry point (Host code)
    gpu_kernels.cu # The CUDA code I provided previously
)

# 4. Link Libraries
# 'CUDA::cudart' is the CUDA Runtime.
# Thrust is header-only and included in the toolkit, so no extra link is usually needed,
# but linking 'CUDA::toolkit' ensures all paths are correct.
target_link_libraries(plane_fitting
    PRIVATE
    CUDA::cudart
)

