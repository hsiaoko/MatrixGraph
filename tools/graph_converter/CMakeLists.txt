cmake_minimum_required(VERSION 3.20)
project(graph_converter LANGUAGES CXX CUDA)

#######################
# Directory Structure
#######################
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_ROOT_DIR}/bin/tools)

#######################
# Source Files
#######################
file(GLOB_RECURSE CONVERTER_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/converter/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/converter/*.cpp"
        )

#######################
# Executables
#######################
file(GLOB CONVERTER_APPS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
foreach (app_source ${CONVERTER_APPS})
    get_filename_component(app_name ${app_source} NAME_WE)
    message(STATUS "[Graph Converter] Adding executable: ${app_name}")

    # Create executable
    add_executable("${app_name}"
            ${app_source}
            ${TOOLS_SOURCES}
            )

    # Set target properties
    set_target_properties("${app_name}" PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
            )

    # Link dependencies
    target_link_libraries("${app_name}"
            PRIVATE
            matrixgraph_core
            yaml-cpp
            gflags
            TBB::tbb
            stdc++fs
            )
endforeach ()
