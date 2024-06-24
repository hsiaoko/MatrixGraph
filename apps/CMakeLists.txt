cmake_minimum_required(VERSION 3.20)
#project(matrixgraph LANGUAGES CXX)
project(matrixgraph_apps LANGUAGES CXX CUDA)

#######################
# Libraries
#######################
include_directories(
        ${PROJECT_ROOT_DIR}
)

link_directories(${PROJECT_ROOT_DIR}/core/lib)

#######################
# Artifacts
#######################
file(GLOB appfiles "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

foreach (appfile ${appfiles})
    message(STATUS "Found apps: ${appfile}")
    get_filename_component(app ${appfile} NAME_WE)
    add_executable("${app}_exec" ${appfile})
    target_link_libraries(
            "${app}_exec"
            PRIVATE
            yaml-cpp
            gflags
            stdc++fs
            matrixgraph_core
    )
endforeach ()