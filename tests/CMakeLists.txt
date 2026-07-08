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
    add_test(NAME nrc_shader_inference_scalar_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_inference_scalar_smoke.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_inference_scalar_smoke.spv")
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

    if(TARGET nrc_vulkan)
        set(NRC_SCALAR_DISPATCH_SPV "${CMAKE_CURRENT_BINARY_DIR}/nrc_scalar_dispatch.spv")
        add_custom_command(
            OUTPUT "${NRC_SCALAR_DISPATCH_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.1"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_scalar_dispatch.comp"
                "-o"
                "${NRC_SCALAR_DISPATCH_SPV}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_scalar_dispatch.comp"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/scalar/NrcNNScalar.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
            VERBATIM)
        add_custom_target(nrc_scalar_dispatch_shader
            DEPENDS "${NRC_SCALAR_DISPATCH_SPV}")

        add_executable(nrc_vulkan_scalar_dispatch_test
            vulkan_scalar_dispatch_test.cpp)
        target_link_libraries(nrc_vulkan_scalar_dispatch_test PRIVATE nrc_vulkan)
        add_dependencies(nrc_vulkan_scalar_dispatch_test nrc_scalar_dispatch_shader)
        add_test(NAME nrc_vulkan_scalar_dispatch_test
            COMMAND nrc_vulkan_scalar_dispatch_test "${NRC_SCALAR_DISPATCH_SPV}")

        set(NRC_INFERENCE_SCALAR_DISPATCH_SPV "${CMAKE_CURRENT_BINARY_DIR}/nrc_inference_scalar_dispatch.spv")
        add_custom_command(
            OUTPUT "${NRC_INFERENCE_SCALAR_DISPATCH_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.1"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_inference_scalar_smoke.comp"
                "-o"
                "${NRC_INFERENCE_SCALAR_DISPATCH_SPV}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_inference_scalar_smoke.comp"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcInferenceScalar.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/scalar/NrcNNScalar.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
            VERBATIM)
        add_custom_target(nrc_inference_scalar_dispatch_shader
            DEPENDS "${NRC_INFERENCE_SCALAR_DISPATCH_SPV}")

        add_executable(nrc_vulkan_inference_scalar_test
            vulkan_inference_scalar_test.cpp)
        target_link_libraries(nrc_vulkan_inference_scalar_test PRIVATE nrc_vulkan)
        add_dependencies(nrc_vulkan_inference_scalar_test nrc_inference_scalar_dispatch_shader)
        add_test(NAME nrc_vulkan_inference_scalar_test
            COMMAND nrc_vulkan_inference_scalar_test "${NRC_INFERENCE_SCALAR_DISPATCH_SPV}")

        set(NRC_KHR_DISPATCH_SG32_SPV "${CMAKE_CURRENT_BINARY_DIR}/nrc_khr_dispatch_sg32.spv")
        add_custom_command(
            OUTPUT "${NRC_KHR_DISPATCH_SG32_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.3"
                "-DNRC_KHR_SUBGROUP_SIZE=32"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_khr_dispatch.comp"
                "-o"
                "${NRC_KHR_DISPATCH_SG32_SPV}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_khr_dispatch.comp"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
            VERBATIM)

        set(NRC_KHR_DISPATCH_SG16_SPV "${CMAKE_CURRENT_BINARY_DIR}/nrc_khr_dispatch_sg16.spv")
        add_custom_command(
            OUTPUT "${NRC_KHR_DISPATCH_SG16_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.3"
                "-DNRC_KHR_SUBGROUP_SIZE=16"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_khr_dispatch.comp"
                "-o"
                "${NRC_KHR_DISPATCH_SG16_SPV}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_khr_dispatch.comp"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
            VERBATIM)

        add_custom_target(nrc_khr_dispatch_shaders
            DEPENDS
                "${NRC_KHR_DISPATCH_SG32_SPV}"
                "${NRC_KHR_DISPATCH_SG16_SPV}")

        add_executable(nrc_vulkan_khr_dispatch_test
            vulkan_khr_dispatch_test.cpp)
        target_link_libraries(nrc_vulkan_khr_dispatch_test PRIVATE nrc_vulkan)
        add_dependencies(nrc_vulkan_khr_dispatch_test nrc_khr_dispatch_shaders)
        add_test(NAME nrc_vulkan_khr_dispatch_test
            COMMAND nrc_vulkan_khr_dispatch_test
                "${NRC_KHR_DISPATCH_SG32_SPV}"
                "${NRC_KHR_DISPATCH_SG16_SPV}")
    endif()
else()
    message(WARNING "glslc was not found; shader compile smoke test will not be registered")
endif()
