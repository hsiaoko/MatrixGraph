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
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*_test\\.cpp$")
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*_test\\.cu$")

# 检测 CPU 指令集支持
include(CheckCXXCompilerFlag)

# 检测并启用 AVX/AVX2
option(ENABLE_AVX "Enable AVX instructions" ON)

if (ENABLE_AVX)
    include(CheckCXXCompilerFlag)

    # 首先尝试 AVX2 + FMA
    check_cxx_compiler_flag("-mavx2" HAS_AVX2)
    check_cxx_compiler_flag("-mfma" HAS_FMA)

    if (HAS_AVX2 AND HAS_FMA)
        add_compile_options(-mavx2 -mfma)
        add_definitions(-DUSE_AVX2)
        message(STATUS "Enabled AVX2 and FMA instructions")

        # 回退到 AVX
    else ()
        check_cxx_compiler_flag("-mavx" HAS_AVX)
        if (HAS_AVX)
            add_compile_options(-mavx)
            add_definitions(-DUSE_AVX)
            message(STATUS "Enabled AVX instructions")
        else ()
            message(WARNING "AVX not available - falling back to scalar code")
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

# Dependencies
target_link_libraries(${PROJECT_NAME}
        PUBLIC
        yaml-cpp
        TBB::tbb
        cublas
        -ffast-math
        -funroll-loops
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