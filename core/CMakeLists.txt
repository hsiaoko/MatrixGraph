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

# Require g++-14 as the CUDA host compiler
#if (NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
#  set(CMAKE_CUDA_HOST_COMPILER "g++-14")
#endif ()
#if (NOT CMAKE_CUDA_HOST_COMPILER STREQUAL "g++-14")
#  message(FATAL_ERROR "MatrixGraph requires g++-14. Use -DCMAKE_CUDA_HOST_COMPILER=g++-14")
#endif ()

# CUDA Configuration
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")  # Set minimum compute capability
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
# [AVX 暂时注释] 禁用 AVX512 和 AMX
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-mno-avx512f,-mno-avx512bf16,-mno-amx-tile")

#######################
# Directory Structure
#######################
#set(LIBRARY_OUTPUT_PATH ${PROJECT_ROOT_DIR}/lib)

#######################
# Source Files
#######################
# Core sources: collect from explicit directory whitelist (stable across CMake/env variants)
set(CORE_SOURCE_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/components"
        "${CMAKE_CURRENT_SOURCE_DIR}/io"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_structures"
        "${CMAKE_CURRENT_SOURCE_DIR}/util"
        "${CMAKE_CURRENT_SOURCE_DIR}/task"
        "${CMAKE_CURRENT_SOURCE_DIR}/task/cpu_task"
        "${CMAKE_CURRENT_SOURCE_DIR}/task/gpu_task"
        "${CMAKE_CURRENT_SOURCE_DIR}/task/gpu_task/kernel"
        "${CMAKE_CURRENT_SOURCE_DIR}/task/gpu_task/kernel/data_structures"
        "${CMAKE_CURRENT_SOURCE_DIR}/task/gpu_task/kernel/algorithms"
)
set(CORE_SOURCES "")
foreach(_d IN LISTS CORE_SOURCE_DIRS)
    if (EXISTS "${_d}")
        file(GLOB _cu CONFIGURE_DEPENDS "${_d}/*.cu")
        file(GLOB _cpp CONFIGURE_DEPENDS "${_d}/*.cpp")
        list(APPEND CORE_SOURCES ${_cu} ${_cpp})
    endif()
endforeach()
list(REMOVE_DUPLICATES CORE_SOURCES)
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*/go_api/.*")
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*_test\\.cpp$")
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*_test\\.cu$")

list(LENGTH CORE_SOURCES CORE_SOURCES_COUNT)
message(STATUS "[matrixgraph_core] source root: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "[matrixgraph_core] source files: ${CORE_SOURCES_COUNT}")
if (CORE_SOURCES_COUNT EQUAL 0)
    message(FATAL_ERROR "matrixgraph_core has no source files after filtering. Check core/CMakeLists.txt source dirs.")
endif ()

# AVX：仅对 C++ 启用，CUDA 不启用（避免 nvcc 与 GCC avx512bf16/amxtileintrin 冲突）
include(CheckCXXCompilerFlag)
option(ENABLE_AVX "Enable AVX instructions" ON)
set(MATRIXGRAPH_AVX_CXX_FLAGS "")
set(MATRIXGRAPH_AVX_DEFINES "")
if (ENABLE_AVX)
    check_cxx_compiler_flag("-mavx2" HAS_AVX2)
    check_cxx_compiler_flag("-mfma" HAS_FMA)
    if (HAS_AVX2 AND HAS_FMA)
        set(MATRIXGRAPH_AVX_CXX_FLAGS "-mavx2 -mfma")
        set(MATRIXGRAPH_AVX_DEFINES "-DUSE_AVX2")
        message(STATUS "Enabled AVX2 and FMA (CXX only)")
    else ()
        check_cxx_compiler_flag("-mavx" HAS_AVX)
        if (HAS_AVX)
            set(MATRIXGRAPH_AVX_CXX_FLAGS "-mavx")
            set(MATRIXGRAPH_AVX_DEFINES "-DUSE_AVX")
        endif ()
    endif ()
endif ()

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

# AVX 仅对 CXX 生效；CUDA 显式取消 USE_AVX2，防止父项目全局定义
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-UUSE_AVX2>
)
if (MATRIXGRAPH_AVX_CXX_FLAGS)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${MATRIXGRAPH_AVX_CXX_FLAGS}>
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${MATRIXGRAPH_AVX_DEFINES}>
    )
endif ()

# Dependencies
target_link_libraries(${PROJECT_NAME}
        PUBLIC
        yaml-cpp
        TBB::tbb
        cublas
        -ffast-math
        -funroll-loops
)

# Include paths: support both flat (core/util/) and nested (core/core/util/) structure
target_include_directories(${PROJECT_NAME}
        PUBLIC
        ${THIRD_PARTY_ROOT}/yaml-cpp/include/
        ${PROJECT_ROOT_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        $<$<BOOL:${CMAKE_CURRENT_SOURCE_DIR}>:${CMAKE_CURRENT_SOURCE_DIR}/core>
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Library properties
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

#######################
# Go C API shared library (for CGO)
#######################
add_library(matrixgraph_goapi SHARED go_api/matrixgraph_go_api.cu)
target_link_libraries(matrixgraph_goapi PRIVATE matrixgraph_core cudart)
target_include_directories(matrixgraph_goapi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_ROOT_DIR}
        ${THIRD_PARTY_ROOT}/yaml-cpp/include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
set_target_properties(matrixgraph_goapi PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_ROOT_DIR}/lib
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)
target_compile_options(matrixgraph_goapi PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-UUSE_AVX2>
)
