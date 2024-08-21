cmake_minimum_required(VERSION 3.12)
project(graph_systems_tools CXX CUDA)

# Directories
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_ROOT_DIR}/bin/tools/)

file(GLOB PARTITION_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/partitioner/*.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/partitioner/*.cpp
        )

#######################
# Artifacts
#######################
file(GLOB appfiles "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
foreach (appfile ${appfiles})
    message(STATUS "Found tool: ${appfile}")
    get_filename_component(app ${appfile} NAME_WE)
    add_executable("${app}_exec" ${appfile} ${TOOLS_SOURCES} ${PARTITION_SOURCES})
    target_link_libraries("${app}_exec"
            PRIVATE
            matrixgraph_core
            yaml-cpp
            gflags
            TBB::tbb
            )
endforeach ()
