
#######################
# graph_converter — 纯主机 CXX；ParForEach 等在 __CUDACC__ 外走 std::execution::par。
#######################
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_ROOT_DIR}/bin/tools)

set(CONVERTER_MAIN "${CMAKE_CURRENT_SOURCE_DIR}/graph_converter.cpp")

if(EXISTS "${CONVERTER_MAIN}")
    add_executable(graph_converter ${CONVERTER_MAIN} ${TOOLS_SOURCES})

    if(DEFINED CMAKE_CUDA_COMPILER)
        get_filename_component(_gc_cuda_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
        get_filename_component(_gc_cuda_root "${_gc_cuda_bin}" DIRECTORY)
        target_include_directories(graph_converter PRIVATE "${_gc_cuda_root}/include")
        target_link_directories(graph_converter PRIVATE "${_gc_cuda_root}/lib64")
    else()
        target_include_directories(graph_converter PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    endif()

    target_link_libraries(graph_converter
            PRIVATE
            matrixgraph_core
            yaml-cpp
            gflags::gflags
            TBB::tbb
            stdc++fs
            cudart
            )

    # format_converter.cuh 经 cuda_check/gpu_bitmap 等仍可包含 CUDA 头与非关键字；
    # 由 g++ 编译时需屏蔽 nvcc 专用关键字。
    target_compile_definitions(graph_converter PRIVATE
            "__host__="
            "__device__="
            )

    set_target_properties(graph_converter PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            )

    message(STATUS "[Graph Converter] CXX executable graph_converter (${CONVERTER_MAIN})")
else()
    message(WARNING "[Graph Converter] graph_converter.cpp not found; skipping tool.")
endif()
