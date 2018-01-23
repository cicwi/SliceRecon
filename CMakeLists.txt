cmake_minimum_required(VERSION 3.0.0)
include(FindPkgConfig)

project(tomopackets)

find_package(Eigen3 REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    "slicerecon/cpp/ext/bulk/include"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wfatal-errors -Wextra -g -O3")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(EXEC_NAME "test_server")

set(
    LIB_NAMES
    "pthread"
)

add_subdirectory("ext/pybind")

set(BINDING_NAME "slre_cpp")
set(BINDING_SOURCES "slicerecon/cpp/field_log_filter.cpp")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../slicerecon")

pybind11_add_module(${BINDING_NAME} ${BINDING_SOURCES})

target_link_libraries(${BINDING_NAME} PRIVATE ${LIB_NAMES})
