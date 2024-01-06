cmake_minimum_required(VERSION 3.20)
project(matrixgraph)

# Directories
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_ROOT_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_ROOT_DIR}/lib)

#######################
# Libraries
#######################
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#######################
# Artifacts
#######################

file(GLOB CORE_SOURCES
        "*.cpp"
        "*.cu"
        "components/*.cu"
        "components/*.cpp"
        "components/scheduler/*.cpp"
        "data_structures/*.cu"
        "data_structures/*.cpp"
        "util/*.cu"
        "util/*.cpp"
        "gpu/*.cpp"
        "gpu/*.cu"
        )
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*_test\\.cpp$")

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
        "gpu/*.cuh"
        )

#######################
# Tests
#######################
if (TEST)
    add_subdirectory(tests)
endif ()

#######################
# Generate libgraph_core
#######################
add_library(matrixgraph ${CORE_SOURCES} ${CORE_HEADERS})

target_link_libraries(matrixgraph
        yaml-cpp
        gflags
        -lcublas
        -lcurand
        tbb
        -static-libstdc++
        )
