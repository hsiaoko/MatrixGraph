cmake_minimum_required(VERSION 3.20)
project(matrixgraph_core LANGUAGES CUDA CXX)

#######################
# Build Configuration
#######################
# CMake Standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA Configuration
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")  # Set minimum compute capability
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")

#######################
# Directory Structure
#######################
#set(LIBRARY_OUTPUT_PATH ${PROJECT_ROOT_DIR}/lib)

#######################
# Source Files
#######################
# Core CUDA sources
file(GLOB_RECURSE CORE_SOURCES
        "*.cu"
        "components/*.cu"
        "io/*.cu"
        "data_structures/*.cu"
        "util/*.cu"
        "task/*.cu"
        "task/gpu_task/kernel/*.cu"
        "task/gpu_task/kernel/data_structures/*.cu"
        "task/gpu_task/kernel/algorithms/*.cu"
        "task/gpu_task/*.cu"
        "task/cpu_task/*.cpp"
        )


#######################
# Tests Configuration
#######################
if (TEST)
    add_subdirectory(tests)
endif ()

#######################
# Library Target
#######################
add_library(${PROJECT_NAME} STATIC ${CORE_SOURCES})

# Dependencies
target_link_libraries(${PROJECT_NAME}
	PUBLIC
        yaml-cpp
	TBB::tbb
        )

target_include_directories(${PROJECT_NAME}
	PUBLIC
        ${THIRD_PARTY_ROOT}/yaml-cpp/include/
	${PROJECT_ROOT_DIR}
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        )

# Library properties
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
