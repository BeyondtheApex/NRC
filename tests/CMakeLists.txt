add_executable(nrc_profile_layout_test
    profile_layout_test.cpp)

target_link_libraries(nrc_profile_layout_test PRIVATE nrc_core)

add_test(NAME nrc_profile_layout_test COMMAND nrc_profile_layout_test)

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
else()
    message(WARNING "glslc was not found; shader compile smoke test will not be registered")
endif()
