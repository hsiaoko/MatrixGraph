cmake_minimum_required(VERSION 3.20)
project(matrixgraph_apps CXX CUDA)

#######################
# Build Configuration
#######################
# CMake Standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compiler flags
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -gdwarf-4")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -gdwarf-4")
endif ()


#find_package(TBB)
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
	    PUBLIC
            gflags::gflags
            matrixgraph_core
    )
    target_include_directories(
            "${app}_exec"
	    PUBLIC
	    ${THIRD_PARTY_ROOT}/yaml-cpp/include/
	    ${GFLAGS_INCLUDE_DIR}
            ${PROJECT_ROOT_DIR}
	    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    # Library properties
    set_target_properties(${app}_exec PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
endforeach ()
