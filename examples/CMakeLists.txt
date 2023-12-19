cmake_minimum_required(VERSION 3.20)
project(matrixgraph)

#######################
# Libraries
#######################
include_directories(
        ${PROJECT_ROOT_DIR}/core
        ${PROJECT_ROOT_DIR}/
)

#######################
# Artifacts
#######################
file(GLOB appfiles "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
foreach (appfile ${appfiles})
    message(STATUS "Found apps: ${appfile}")
    get_filename_component(app ${appfile} NAME_WE)
    cuda_add_executable("${app}_exec" ${appfile})
    target_link_libraries(
            "${app}_exec"
            yaml-cpp
            gflags
            stdc++fs
            matrixgraph
    )
endforeach ()