cmake_minimum_required(VERSION 3.20)
project(matrixgraph_tools LANGUAGES CXX CUDA)

#######################
# Libraries
#######################
message(STATUS "PROJECT ROOT: ${PROJECT_ROOT_DIR}")
include_directories(${PROJECT_ROOT_DIR})

file(GLOB TOOLS_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/util/*.cpp
        )

#######################
# Tests
#######################
if (TEST)
    add_subdirectory(tests)
endif ()

#######################
# Artifacts
#######################
add_subdirectory(graph_converter)
add_subdirectory(graph_partitioner)
