add_executable(nrc_profile_layout_test
    profile_layout_test.cpp)

target_link_libraries(nrc_profile_layout_test PRIVATE nrc_core)

add_test(NAME nrc_profile_layout_test COMMAND nrc_profile_layout_test)

add_executable(nrc_reference_nn_test
    reference_nn_test.cpp)

target_link_libraries(nrc_reference_nn_test PRIVATE nrc_core)

add_test(NAME nrc_reference_nn_test COMMAND nrc_reference_nn_test)

add_executable(nrc_weights_test
    weights_test.cpp)

target_link_libraries(nrc_weights_test PRIVATE nrc_core)

add_test(NAME nrc_weights_test COMMAND nrc_weights_test)

if(TARGET nrc_vulkan)
    add_executable(nrc_vulkan_capabilities_test
        vulkan_capabilities_test.cpp)
    target_link_libraries(nrc_vulkan_capabilities_test PRIVATE nrc_vulkan)
    add_test(NAME nrc_vulkan_capabilities_test COMMAND nrc_vulkan_capabilities_test)
endif()

find_program(NRC_GLSLC
    NAMES glslc glslc.exe
    HINTS
        "$ENV{VULKAN_SDK}/Bin"
        "D:/VulkanSDK/1.4.335.0/Bin")

if(NRC_GLSLC)
    add_test(NAME nrc_shader_common_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_common_smoke.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_common_smoke.spv")
    add_test(NAME nrc_shader_scalar_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_scalar_smoke.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_scalar_smoke.spv")
    add_test(NAME nrc_shader_khr_sg32_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=32"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_khr_smoke.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_khr_sg32_smoke.spv")
    add_test(NAME nrc_shader_khr_sg16_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=16"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_khr_smoke.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_khr_sg16_smoke.spv")
else()
    message(WARNING "glslc was not found; shader compile smoke test will not be registered")
endif()
