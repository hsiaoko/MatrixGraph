cmake_minimum_required(VERSION 3.20)

# CUDA host compiler: prefer g++-14 or newer when not set by caller (e.g. our build script)
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  set(_cuda_host_cxx "")
  foreach(_ver 16 15 14)
    find_program(_gxx NAMES g++-${_ver})
    if(_gxx)
      set(_cuda_host_cxx "${_gxx}")
      break()
    endif()
  endforeach()
  if(_cuda_host_cxx)
    set(CMAKE_CUDA_HOST_COMPILER "${_cuda_host_cxx}" CACHE STRING "CUDA host compiler (g++-14+ preferred)")
    message(STATUS "[MatrixGraph] Using CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER}")
  else()
    message(STATUS "[MatrixGraph] No g++-14/15/16 found; CUDA host compiler will use default CXX. Install g++-14 or set -DCMAKE_CUDA_HOST_COMPILER=... if nvcc fails.")
  endif()
  unset(_cuda_host_cxx)
  unset(_gxx)
endif()

project(matrixgraph LANGUAGES CXX CUDA)

#######################
# Build Configuration
#######################
# CMake Standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_ARCHITECTURES "sm_70")

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
set(GFLAGS_ROOT_DIR "${THIRD_PARTY_ROOT}/gflags")
set(TBB_ROOT_DIR "${THIRD_PARTY_ROOT}/oneTBB/")
set(YAML_ROOT_DIR "${THIRD_PARTY_ROOT}/yaml-cpp")

list(PREPEND CMAKE_PREFIX_PATH "${GFLASG_ROOT_DIR}")
list(PREPEND CMAKE_PREFIX_PATH "${TBB_ROOT_DIR}")
list(PREPEND CMAKE_PREFIX_PATH "${YAML_ROOT_DIR}")

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
# Gflags
add_subdirectory("${GFLAGS_ROOT_DIR}" EXCLUDE_FROM_ALL)

# TBB: 优先使用 bundled oneTBB（与项目同编译器，避免 libtbb 与 libstdc++ 版本不匹配）
# 系统 libtbb 若由 GCC 13+ 编译会依赖 GLIBCXX_3.4.32，与 g++-11 不兼容
if (EXISTS "${TBB_ROOT_DIR}/CMakeLists.txt")
  set(TBB_TEST OFF CACHE BOOL "Disable TBB tests" FORCE)
  add_subdirectory("${TBB_ROOT_DIR}" EXCLUDE_FROM_ALL)
  message(STATUS "[MatrixGraph] Using bundled oneTBB")
else ()
  find_package(TBB REQUIRED)
  message(STATUS "[MatrixGraph] Using system TBB")
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


# YAML-CPP
add_subdirectory("${YAML_ROOT_DIR}" EXCLUDE_FROM_ALL)

#######################
# Project Components
#######################
add_subdirectory(core)
add_subdirectory(tools)
add_subdirectory(apps)