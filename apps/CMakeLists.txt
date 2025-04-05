cmake_minimum_required(VERSION 3.20)
project(matrixgraph_apps CXX CUDA)

#######################
# Artifacts
#######################
file(GLOB appfiles "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach (appfile ${appfiles})
    message(STATUS "Found apps: ${appfile}")
    get_filename_component(app ${appfile} NAME_WE)

    add_executable("${app}_exec" ${appfile})
    target_link_libraries(
            "${app}_exec"
            PRIVATE
            yaml-cpp
            gflags
            pthread
            matrixgraph_core
            ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64/libcudadevrt.a
            ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64/libcudart.so
    )
    target_include_directories(
            "${app}_exec"
            PRIVATE
            ${THIRD_PARTY_ROOT}/yaml-cpp/include/
            ${PROJECT_ROOT_DIR}
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
endforeach ()