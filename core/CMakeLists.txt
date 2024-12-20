cmake_minimum_required(VERSION 3.20)
project(matrixgraph_core LANGUAGES CUDA)

#######################
# Cmake options
#######################

# Set Cmake Standard
set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++21 -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_21 -std=c++20)

# Directories
set(LIBRARY_OUTPUT_PATH ${PROJECT_ROOT_DIR}/lib)
set(TOOLS_SRC_PATH ${PROJECT_ROOT_DIR}/tools)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#######################
# Artifacts
#######################

file(GLOB CORE_SOURCES
        "*.cu"
        "components/*.cu"
        "io/*.cu"
        "data_structures/*.cu"
        "util/*.cu"
        "task/*.cu"
        "task/kernel/*.cu"
        "task/kernel/data_structures/*.cu"
        )


file(GLOB CORE_HEADERS
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
        )

#######################
# Tests
#######################
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*_test\\.cpp$")
if (TEST)
    add_subdirectory(tests)
endif ()

#######################
# Generate libgraph_core
#######################
add_library(matrixgraph_core STATIC ${CORE_SOURCES} ${CORE_HEADERS})

target_link_libraries(matrixgraph_core
        yaml-cpp
        -lcublas
        TBB::tbb
        -static-libstdc++
        )
#target_compile_features(matrixgraph_core PUBLIC cxx_std_20)
set_target_properties(matrixgraph_core
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)