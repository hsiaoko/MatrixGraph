project(core_tests)

include_directories(SYSTEM ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(SYSTEM ${gmock_SOURCE_DIR}/include ${gmock_SOURCE_DIR})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_ROOT_DIR}/bin/tests/)

####################
# Build Unit Tests #
####################
file(GLOB testfiles
        "${PROJECT_ROOT_DIR}/core/data_structures/*_test.cpp"
        "${PROJECT_ROOT_DIR}/core/data_structures/*_test.cu"
        "${PROJECT_ROOT_DIR}/core/task/gpu_task/*_test.cu"
)

#find_package(LibLZMA REQUIRED)

foreach (testfile ${testfiles})
    message(testfile)
    get_filename_component(filename ${testfile} NAME_WE)
    string(REPLACE "_test"
            "" testname
            ${filename})
    string(TOUPPER ${testname} TESTNAME)
    add_executable(${filename} "${testfile}")

    # GPU task tests need the full matrixgraph_core library.
    if ("${testfile}" MATCHES "gpu_task")
        target_link_libraries(${filename}
                matrixgraph_core
                gtest
                gtest_main
                gflags
                cudart
        )
    else ()
        target_link_libraries(${filename}
                gtest
                gtest_main
                gflags
        )
    endif ()

    add_test(NAME "${TESTNAME}" COMMAND "${filename}")
    target_compile_definitions(${filename} PUBLIC TEST_DATA_DIR="${PROJECT_ROOT_DIR}/testfile")
    #    set_tests_properties(${TESTNAME} PROPERTIES ENVIRONMENT "DATA_DIR=${PROJECT_ROOT_DIR}/testfile")
endforeach ()