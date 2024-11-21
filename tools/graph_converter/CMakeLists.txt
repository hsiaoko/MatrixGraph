cmake_minimum_required(VERSION 3.20)
project(matrixgraph_tools LANGUAGES CXX CUDA)

# Directories
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_ROOT_DIR}/bin/tools/)

file(GLOB CONVERTER_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/converter/*.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/converter/*.cpp
        )

#######################
# Artifacts
#######################
file(GLOB appfiles "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
foreach (appfile ${appfiles})
    message(STATUS "Found tool: ${appfile}")
    get_filename_component(app ${appfile} NAME_WE)
    add_executable("${app}_exec" ${appfile} ${TOOLS_SOURCES} ${CONVERTER_SOURCES})
    target_link_libraries("${app}_exec"
            PRIVATE
            yaml-cpp
            gflags
            TBB::tbb
            stdc++fs
            matrixgraph_core
            ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64/libcudadevrt.a
            )
endforeach ()