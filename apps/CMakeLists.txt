cmake_minimum_required(VERSION 3.20)
project(matrixgraph_apps CXX CUDA)

#######################
# Libraries
#######################
#include_directories(${PROJECT_ROOT_DIR})

#get_property(_variableNames DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VARIABLES)
#message( STATUS "_variableNames=${_variableNames}" )

#######################
# Artifacts
#######################
#file(GLOB appfiles "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
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
    #	    #stdc++fs
           matrixgraph_core
    #CUDA::cudart
    #	    #/lib/klibc-IYXwqr0atR70aQDgNUuRThfwUwA.so
    	    ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64/libcudadevrt.a
    	    ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64/libcudart.so
    )
	message(STATUS "PROJECTROOT: "${PROJECT_ROOT_DIR})
	message(STATUS "PROJECTROOT: "${CMAKE_CUDA_COMPILER_LIBRARY_ROOT})
    target_include_directories(
    	    "${app}_exec"
    	    PRIVATE
    	    ${THIRD_PARTY_ROOT}/yaml-cpp/include/
    #	    #${THIRD_PARTY_ROOT}/cutlass/include
    	    ${PROJECT_ROOT_DIR}
	    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    	    )
    
endforeach ()
