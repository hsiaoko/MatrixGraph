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
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")  # Set minimum compute capability
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")

#######################
# Directory Structure
#######################
set(LIBRARY_OUTPUT_PATH ${PROJECT_ROOT_DIR}/lib)
set(TOOLS_SRC_PATH ${PROJECT_ROOT_DIR}/tools)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

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
    "task/kernel/*.cu"
    "task/kernel/data_structures/*.cu"
    "task/kernel/algorithms/*.cu"
)

# Header files
file(GLOB_RECURSE CORE_HEADERS
    "*.h"
    "*.cuh"
    "components/*.h"
    "components/*.cuh"
    "io/*.cuh"
    "components/scheduler/*.h"
    "components/scheduler/*.cuh"
    "data_structures/*.h"
    "data_structures/*.cuh"
    "util/*.h"
    "util/*.cuh"
    "task/*.cuh"
    "task/kernel/*.cuh"
    "task/kernel/data_structures/*.cuh"
    "task/kernel/algorithms/*.cuh"
)

#######################
# Tests Configuration
#######################
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*_test\\.cpp$")
if(TEST)
    add_subdirectory(tests)
endif()

#######################
# Library Target
#######################
add_library(${PROJECT_NAME} STATIC ${CORE_SOURCES} ${CORE_HEADERS})

# Library properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Dependencies
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        yaml-cpp
        cublas
        TBB::tbb
    PRIVATE
        -static-libstdc++
)