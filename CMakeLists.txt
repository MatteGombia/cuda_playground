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
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)


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

ament_target_dependencies(plane_fitting
  rclcpp
  sensor_msgs
)

target_include_directories(plane_fitting PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(plane_fitting PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

install(TARGETS plane_fitting
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()