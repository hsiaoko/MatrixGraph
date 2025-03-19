cmake_minimum_required(VERSION 3.20)
project(graph_partitioner LANGUAGES CXX CUDA)

#######################
# Directory Structure
#######################
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_ROOT_DIR}/bin/tools)

#######################
# Source Files
#######################
file(GLOB_RECURSE PARTITION_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/partitioner/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/partitioner/*.cpp"
)

#######################
# Executables
#######################
file(GLOB PARTITIONER_APPS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
foreach(app_source ${PARTITIONER_APPS})
    get_filename_component(app_name ${app_source} NAME_WE)
    message(STATUS "[Graph Partitioner] Adding executable: ${app_name}")
    
    # Create executable
    add_executable("${app_name}" 
        ${app_source} 
        ${TOOLS_SOURCES} 
        ${PARTITION_SOURCES}
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
            "${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64/libcudadevrt.a"
    )
endforeach()
