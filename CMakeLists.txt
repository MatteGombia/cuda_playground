cmake_minimum_required(VERSION 3.18)

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

# Enable CUDA C++17 for atomic_ref support
target_compile_options(plane_fitting PRIVATE 
  $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17 -arch=sm_75>
)
# Enable device code atomics for cuda::atomic_ref
set_target_properties(plane_fitting PROPERTIES CUDA_ARCHITECTURES "75;80;86")
# ---------------------------------------------

# 6. Install Rules
install(TARGETS plane_fitting
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()