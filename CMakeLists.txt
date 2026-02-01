cmake_minimum_required(VERSION 3.18)

# 1. Force Compilers (Must be before project)
set(CMAKE_C_COMPILER /usr/bin/gcc-10)
set(CMAKE_CXX_COMPILER /usr/bin/g++-10)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-10)

project(cuda_segmentation LANGUAGES CXX CUDA)

# 2. Force Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 3. Find Packages
find_package(CUDAToolkit REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

# 4. Add Executable
add_executable(plane_fitting
    main.cpp
    gpu_kernels.cu
)

# 5. Link Libraries
target_link_libraries(plane_fitting
    CUDA::cudart
)

ament_target_dependencies(plane_fitting
  rclcpp
  sensor_msgs
)

# --- CRITICAL FIX: MANUALLY ADD C++17 FLAGS ---
# This forces g++-10 to use C++17 even if CMake fails to set it automatically
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  target_compile_options(plane_fitting PRIVATE -std=c++17)
endif()

# This tells NVCC to tell GCC to use C++17
target_compile_options(plane_fitting PRIVATE 
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -std=c++17>
)
# ---------------------------------------------

# 6. Install Rules
install(TARGETS plane_fitting
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()