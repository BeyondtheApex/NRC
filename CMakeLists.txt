cmake_minimum_required(VERSION 3.25)

project(NRC
    VERSION 0.1.0
    DESCRIPTION "Reusable Vulkan Neural Radiance Cache library and reference path tracer"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(nrc_core INTERFACE)
target_include_directories(nrc_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

add_library(nrc_shaders INTERFACE)
target_include_directories(nrc_shaders INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/shaders>
    $<INSTALL_INTERFACE:shaders>)

option(NRC_BUILD_SAMPLE_PATH_TRACER "Build the NRC reference path tracer sample" ON)
option(NRC_BUILD_TESTS "Build NRC tests" ON)

if(NRC_BUILD_SAMPLE_PATH_TRACER)
    add_subdirectory(sample/path_tracer)
endif()

if(NRC_BUILD_TESTS)
    enable_testing()
endif()
