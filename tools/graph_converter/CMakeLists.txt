cmake_minimum_required(VERSION 3.20)
project(matrixgraph LANGUAGES CXX CUDA)

# Directories
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_ROOT_DIR}/bin/tools/)

#######################
# Artifacts
#######################
file(GLOB appfiles "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
foreach (appfile ${appfiles})
    message(STATUS "Found tool: ${appfile}")
    get_filename_component(app ${appfile} NAME_WE)
    add_executable("${app}_exec" ${appfile} ${TOOLS_SOURCES})
    target_link_libraries("${app}_exec"
            PRIVATE
            yaml-cpp
            gflags
            tbb
            #-static-libstdc++
            matrixgraph_core
            )
endforeach ()
