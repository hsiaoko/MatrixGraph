cmake_minimum_required(VERSION 3.20)
project(matrixgraph LANGUAGES CXX CUDA)

#######################
# Build Configuration
#######################
# CMake Standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build Type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif ()

# Compiler flags
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -gdwarf-4")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -gdwarf-4")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -gdwarf-4")
endif ()

# Status messages
message(STATUS "[MatrixGraph] Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compilers: ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_VERSION}, ${CMAKE_CUDA_COMPILER} ${CMAKE_CUDA_COMPILER_VERSION}")

#######################
# Directory Structure
#######################
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(THIRD_PARTY_ROOT ${PROJECT_ROOT_DIR}/third_party)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_ROOT_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_ROOT_DIR}/lib)
set(CMAKE_MODULE_PATH ${PROJECT_ROOT_DIR}/cmake)
set(LIBRARY_ROOT_DIR ${PROJECT_ROOT_DIR}/lib)

# Third party roor dir.
#set(TBB_ROOT_DIR "/hdd_data/local/oneTBB/")
set(TBB_ROOT_DIR "${THIRD_PARTY_ROOT}/oneTBB/")
set(GFLAGS_ROOT_DIR "${THIRD_PARTY_ROOT}/gflags")
set(YAML_ROOT_DIR "${THIRD_PARTY_ROOT}/yaml-cpp")

list(APPEND CMAKE_PREFIX_PATH "${TBB_ROOT_DIR}")
list(APPEND CMAKE_PREFIX_PATH "${GFLASG_ROOT_DIR}")
list(APPEND CMAKE_PREFIX_PATH "${YAML_ROOT_DIR}")

#######################
# Dependencies
#######################
# Jemalloc (Optional)
if (USE_JEMALLOC)
    include("cmake/FindJemalloc.cmake" OPTIONAL)
    if (JEMALLOC_FOUND)
        add_definitions(-DUSE_JEMALLOC)
        include_directories(SYSTEM ${JEMALLOC_INCLUDE_DIRS})
        message(STATUS "jemalloc found and enabled")
    else ()
        message(STATUS "jemalloc not found, building without it")
    endif ()
endif ()


#######################
# Third Party Libraries
#######################
# TBB
add_subdirectory("${TBB_ROOT_DIR}" EXCLUDE_FROM_ALL)
if (TBB_FOUND)
    message(STATUS "TBB found")
    add_compile_definitions(TBB_FOUND)
endif ()

# GoogleTest
if (TEST)
    enable_testing()
    message(STATUS "Enabling GoogleTest")
    set(BUILD_GMOCK ON CACHE BOOL "Build gmock" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "Don't install gtest" FORCE)
    add_subdirectory("${THIRD_PARTY_ROOT}/googletest" EXCLUDE_FROM_ALL)
    include_directories(SYSTEM
            ${gtest_SOURCE_DIR}/include
            ${gmock_SOURCE_DIR}/include
            )
endif ()

# Gflags
add_subdirectory("${GFLAGS_ROOT_DIR}" EXCLUDE_FROM_ALL)

# YAML-CPP
add_subdirectory("${YAML_ROOT_DIR}" EXCLUDE_FROM_ALL)

#######################
# Project Components
#######################
add_subdirectory(core)
add_subdirectory(tools)
add_subdirectory(apps)
