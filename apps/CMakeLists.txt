cmake_minimum_required(VERSION 3.20)
project(matrixgraph_apps CUDA)

#######################
# Libraries
#######################
include_directories(
        ${PROJECT_ROOT_DIR}
)

link_directories(${PROJECT_ROOT_DIR}/core/lib)

#get_property(_variableNames DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VARIABLES)
#message( STATUS "_variableNames=${_variableNames}" )

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
            TBB::tbb
            matrixgraph_core
            ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64/libcudadevrt.a
    )
endforeach ()