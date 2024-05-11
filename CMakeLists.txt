cmake_minimum_required(VERSION 3.20)
#project(matrixgraph LANGUAGES CXX CUDA)
project(matrixgraph)

###### Cmake options ######

# Set Cmake Standard
set(CMAKE_CXX_STANDARD 20)

# Set default cmake type to Debug
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
else ()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif ()


message(STATUS "[MatrixGraph] will build in type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Using compiler: ${CMAKE_CXX_COMPILER} - ${CMAKE_CXX_COMPILER_VERSION}, ${CMAKE_CUDA_COMPILER} - ${CMAKE_CUDA_COMPILER_VERSION}")

###### Directories shortcuts ######
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(THIRD_PARTY_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_ROOT_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${PROJECT_ROOT_DIR}/lib)
set(CMAKE_MODULE_PATH ${PROJECT_ROOT_DIR}/cmake)
set(LIBRARY_ROOT_DIR ${PROJECT_ROOT_DIR}/lib)

#######################
# Libraries
#######################
if (USE_JEMALLOC)
    include("cmake/FindJemalloc.cmake" OPTIONAL)
    if (NOT JEMALLOC_FOUND)
        message(STATUS "jemalloc not found, build without jemalloc")
    else ()
        add_definitions(-DUSE_JEMALLOC)
        include_directories(SYSTEM ${JEMALLOC_INCLUDE_DIRS})
    endif ()
endif ()

###### Add third party to the project, if necessary ######
# googleTest
if (TEST)
    enable_testing()
    message(STATUS "Enabling GoogleTest...")
    set(BUILD_GMOCK ON CACHE BOOL "Option: build gmock and gtest -- ON" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "Option: install gtest -- OFF" FORCE)
    include_directories(SYSTEM ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
    include_directories(SYSTEM ${gmock_SOURCE_DIR}/include ${gmock_SOURCE_DIR})
    add_subdirectory("${THIRD_PARTY_ROOT}/googletest" EXCLUDE_FROM_ALL)
endif ()

# gflags
add_subdirectory("${THIRD_PARTY_ROOT}/gflags")
#include_directories(SYSTEM ${GFLAGS_INCLUDE_DIR})

# yaml
add_subdirectory("${THIRD_PARTY_ROOT}/yaml-cpp" EXCLUDE_FROM_ALL)
include_directories(SYSTEM ${THIRD_PARTY_ROOT}/yaml-cpp/include/)

# rapidcsv
include_directories(SYSTEM ${THIRD_PARTY_ROOT}/rapidcsv/src)

# CUTLASS
include_directories(SYSTEM ${THIRD_PARTY_ROOT}/cutlass/include)

#######################
# Artifacts
#######################
include_directories(
        ${PROJECT_ROOT_DIR}
)

add_subdirectory(core)
add_subdirectory(tools)


add_subdirectory(apps)
