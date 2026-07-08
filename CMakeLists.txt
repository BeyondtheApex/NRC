cmake_minimum_required(VERSION 3.25)

project(NRC
    VERSION 0.1.0
    DESCRIPTION "Reusable Vulkan Neural Radiance Cache library and reference path tracer"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(nrc_core STATIC)
target_sources(nrc_core PRIVATE
    src/nrc_encoding.cpp
    src/nrc_khr_reference.cpp
    src/nrc_optimizer.cpp
    src/nrc_reference_nn.cpp
    src/nrc_weights.cpp
    src/nrc_profile.cpp
    src/nrc_resource_layout.cpp)
target_include_directories(nrc_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

add_library(nrc_shaders INTERFACE)
target_include_directories(nrc_shaders INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/shaders>
    $<INSTALL_INTERFACE:shaders>)

option(NRC_BUILD_SAMPLE_PATH_TRACER "Build the NRC reference path tracer sample" ON)
option(NRC_BUILD_TESTS "Build NRC tests" ON)
option(NRC_BUILD_VULKAN_RUNTIME "Build optional Vulkan runtime helpers" ON)

if(NRC_BUILD_VULKAN_RUNTIME)
    find_package(Vulkan QUIET)
    if(NOT Vulkan_FOUND AND EXISTS "D:/VulkanSDK/1.4.335.0/Include/vulkan/vulkan.h")
        add_library(Vulkan::Vulkan UNKNOWN IMPORTED)
        set_target_properties(Vulkan::Vulkan PROPERTIES
            IMPORTED_LOCATION "D:/VulkanSDK/1.4.335.0/Lib/vulkan-1.lib"
            INTERFACE_INCLUDE_DIRECTORIES "D:/VulkanSDK/1.4.335.0/Include")
        set(Vulkan_FOUND TRUE)
    endif()

    if(Vulkan_FOUND)
        add_library(nrc_vulkan STATIC)
        target_sources(nrc_vulkan PRIVATE
            runtime/vulkan/nrc_vulkan_barriers.cpp
            runtime/vulkan/nrc_vulkan_capabilities.cpp
            runtime/vulkan/nrc_vulkan_descriptors.cpp
            runtime/vulkan/nrc_vulkan_pipeline.cpp
            runtime/vulkan/nrc_vulkan_shader.cpp)
        target_link_libraries(nrc_vulkan PUBLIC
            nrc_core
            Vulkan::Vulkan)
    else()
        message(WARNING "Vulkan SDK was not found; nrc_vulkan will not be built")
    endif()
endif()

if(NRC_BUILD_SAMPLE_PATH_TRACER)
    add_subdirectory(sample/path_tracer)
endif()

if(NRC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
