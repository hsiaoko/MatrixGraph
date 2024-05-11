cmake_minimum_required(VERSION 3.20)
project(matrixgraph_core LANGUAGES CUDA)
#project(matrixgraph_core)

#######################
# Cmake options
#######################

# Set Cmake Standard
set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++21 -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_21 -std=c++20)

# Directories
set(LIBRARY_OUTPUT_PATH ${PROJECT_ROOT_DIR}/core/lib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#######################
# Libraries
#######################
# CUDA
#if (${CMAKE_VERSION} VERSION_LESS_EQUAL "3.13.4")
#    cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto") # optional argument for arch to add
#    message("ARCH_FLAGS = ${ARCH_FLAGS}")
#    string(REPLACE "-gencode;" "--generate-code=" ARCH_FLAGS "${ARCH_FLAGS}")
#    string(APPEND CMAKE_CUDA_FLAGS "${ARCH_FLAGS}")
#else ()
#    include(FindCUDA/select_compute_arch)
#    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
#    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
#    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
#    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
#    SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
#    set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
#endif ()
#set_property(matrixgraph_core TARGET PROPERTY CUDA_ARCHITECTURES 70-real 72-virtual)

# TBB
find_package(TBB)
if (TBB_FOUND)
    message(STATUS "TBB found")
    add_compile_definitions(TBB_FOUND)
endif ()


#######################
# Artifacts
#######################

file(GLOB CORE_SOURCES
        "*.cu"
        "components/*.cu"
        "data_structures/*.cu"
        "util/*.cu"
        "task/*.cu"
        "task/kernel/*.cu"
        )

file(GLOB CORE_HEADERS
        "*.h"
        "*.cuh"
        "components/*.h"
        "components/*.cuh"
        "components/scheduler/*.h"
        "components/scheduler/*.cuh"
        "data_structures/*.h"
        "data_structures/*.cuh"
        "util/*.h"
        "util/*.cuh"
        "task/*.cuh"
        "task/kernel/*.cuh"
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
        #-lcurand
        tbb
        -static-libstdc++
        )
target_compile_features(matrixgraph_core PUBLIC cxx_std_20)
set_target_properties(matrixgraph_core
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#target_compile_options(matrixgraph_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_50,code=sm_50>)
