add_executable(nrc_profile_layout_test
    profile_layout_test.cpp)

target_link_libraries(nrc_profile_layout_test PRIVATE nrc_core)

add_test(NAME nrc_profile_layout_test COMMAND nrc_profile_layout_test)

add_executable(nrc_reference_nn_test
    reference_nn_test.cpp)

target_link_libraries(nrc_reference_nn_test PRIVATE nrc_core)

add_test(NAME nrc_reference_nn_test COMMAND nrc_reference_nn_test)

add_executable(nrc_reference_gradient_test
    reference_gradient_test.cpp)

target_link_libraries(nrc_reference_gradient_test PRIVATE nrc_core)

add_test(NAME nrc_reference_gradient_test COMMAND nrc_reference_gradient_test)

add_executable(nrc_khr_gradient_reference_test
    khr_gradient_reference_test.cpp)

target_link_libraries(nrc_khr_gradient_reference_test PRIVATE nrc_core)

add_test(NAME nrc_khr_gradient_reference_test COMMAND nrc_khr_gradient_reference_test)

add_executable(nrc_encoding_test
    encoding_test.cpp)

target_link_libraries(nrc_encoding_test PRIVATE nrc_core)

add_test(NAME nrc_encoding_test COMMAND nrc_encoding_test)

add_executable(nrc_weights_test
    weights_test.cpp)

target_link_libraries(nrc_weights_test PRIVATE nrc_core)

add_test(NAME nrc_weights_test COMMAND nrc_weights_test)

add_executable(nrc_optimizer_test
    optimizer_test.cpp)

target_link_libraries(nrc_optimizer_test PRIVATE nrc_core)

add_test(NAME nrc_optimizer_test COMMAND nrc_optimizer_test)

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
    add_test(NAME nrc_shader_queue_ops_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.1"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_queue_ops_dispatch.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_queue_ops_smoke.spv")
    add_test(NAME nrc_shader_train_prepare_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.1"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_train_prepare_dispatch.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_train_prepare_smoke.spv")
    add_test(NAME nrc_shader_optimize_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.1"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_optimize_dispatch.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_optimize_smoke.spv")
    add_test(NAME nrc_shader_optimize_use_weights_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.1"
            "-DNRC_OPTIMIZE_WRITE_USE_WEIGHTS=1"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_optimize_dispatch.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_optimize_use_weights_smoke.spv")
    add_test(NAME nrc_shader_gradient_scalar_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.2"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_scalar_dispatch.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_scalar_smoke.spv")
    add_test(NAME nrc_shader_gradient_khr_sg32_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=32"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_sg32_smoke.spv")
    add_test(NAME nrc_shader_gradient_khr_sg16_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=16"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_sg16_smoke.spv")
    add_test(NAME nrc_shader_gradient_khr_m8_sg32_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=32"
            "-DNRC_KHR_COOPMAT_M=8"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_m8_sg32_smoke.spv")
    add_test(NAME nrc_shader_gradient_khr_coopmat_output_sg32_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=32"
            "-DNRC_KHR_USE_COOPMAT_OUTPUT_DW=1"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_output_sg32_smoke.spv")
    add_test(NAME nrc_shader_gradient_khr_coopmat_output_sg16_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=16"
            "-DNRC_KHR_USE_COOPMAT_OUTPUT_DW=1"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_output_sg16_smoke.spv")
    add_test(NAME nrc_shader_gradient_khr_coopmat_output_m8_sg32_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=32"
            "-DNRC_KHR_COOPMAT_M=8"
            "-DNRC_KHR_USE_COOPMAT_OUTPUT_DW=1"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_output_m8_sg32_smoke.spv")
    add_test(NAME nrc_shader_gradient_khr_coopmat_final_hidden_sg32_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=32"
            "-DNRC_KHR_USE_COOPMAT_FINAL_HIDDEN_DW=1"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_final_hidden_sg32_smoke.spv")
    add_test(NAME nrc_shader_gradient_khr_coopmat_final_hidden_sg16_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=16"
            "-DNRC_KHR_USE_COOPMAT_FINAL_HIDDEN_DW=1"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_final_hidden_sg16_smoke.spv")
    add_test(NAME nrc_shader_gradient_khr_coopmat_final_hidden_m8_sg32_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=32"
            "-DNRC_KHR_COOPMAT_M=8"
            "-DNRC_KHR_USE_COOPMAT_FINAL_HIDDEN_DW=1"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_final_hidden_m8_sg32_smoke.spv")
    add_test(NAME nrc_shader_gradient_khr_coopmat_hidden3_sg32_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=32"
            "-DNRC_KHR_USE_COOPMAT_HIDDEN3_DW=1"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_hidden3_sg32_smoke.spv")
    add_test(NAME nrc_shader_gradient_khr_coopmat_hidden3_sg16_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=16"
            "-DNRC_KHR_USE_COOPMAT_HIDDEN3_DW=1"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_hidden3_sg16_smoke.spv")
    add_test(NAME nrc_shader_gradient_khr_coopmat_hidden3_m8_sg32_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=32"
            "-DNRC_KHR_COOPMAT_M=8"
            "-DNRC_KHR_USE_COOPMAT_HIDDEN3_DW=1"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_hidden3_m8_sg32_smoke.spv")
    add_test(NAME nrc_shader_gradient_khr_coopmat_hidden2_sg32_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=32"
            "-DNRC_KHR_USE_COOPMAT_HIDDEN2_DW=1"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_hidden2_sg32_smoke.spv")
    add_test(NAME nrc_shader_gradient_khr_coopmat_hidden2_sg16_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=16"
            "-DNRC_KHR_USE_COOPMAT_HIDDEN2_DW=1"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_hidden2_sg16_smoke.spv")
    add_test(NAME nrc_shader_gradient_khr_coopmat_hidden2_m8_sg32_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=32"
            "-DNRC_KHR_COOPMAT_M=8"
            "-DNRC_KHR_USE_COOPMAT_HIDDEN2_DW=1"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_hidden2_m8_sg32_smoke.spv")
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
    add_test(NAME nrc_shader_khr_m8_sg32_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=32"
            "-DNRC_KHR_COOPMAT_M=8"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_khr_smoke.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_khr_m8_sg32_smoke.spv")
    add_test(NAME nrc_shader_khr_backprop_sg32_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=32"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_khr_backprop_smoke.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_khr_backprop_sg32_smoke.spv")
    add_test(NAME nrc_shader_khr_backprop_sg16_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=16"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_khr_backprop_smoke.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_khr_backprop_sg16_smoke.spv")
    add_test(NAME nrc_shader_khr_backprop_m8_sg32_compile
        COMMAND "${NRC_GLSLC}"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=32"
            "-DNRC_KHR_COOPMAT_M=8"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_khr_backprop_smoke.comp"
            "-o"
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_khr_backprop_m8_sg32_smoke.spv")

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

        set(NRC_QUEUE_OPS_DISPATCH_SPV "${CMAKE_CURRENT_BINARY_DIR}/nrc_queue_ops_dispatch.spv")
        add_custom_command(
            OUTPUT "${NRC_QUEUE_OPS_DISPATCH_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.1"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_queue_ops_dispatch.comp"
                "-o"
                "${NRC_QUEUE_OPS_DISPATCH_SPV}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_queue_ops_dispatch.comp"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcQueueOps.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcQueue.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
            VERBATIM)
        add_custom_target(nrc_queue_ops_dispatch_shader
            DEPENDS "${NRC_QUEUE_OPS_DISPATCH_SPV}")

        add_executable(nrc_vulkan_queue_ops_test
            vulkan_queue_ops_test.cpp)
        target_link_libraries(nrc_vulkan_queue_ops_test PRIVATE nrc_vulkan)
        add_dependencies(nrc_vulkan_queue_ops_test nrc_queue_ops_dispatch_shader)
        add_test(NAME nrc_vulkan_queue_ops_test
            COMMAND nrc_vulkan_queue_ops_test "${NRC_QUEUE_OPS_DISPATCH_SPV}")

        set(NRC_GRADIENT_SCALAR_DISPATCH_SPV "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_scalar_dispatch.spv")
        add_custom_command(
            OUTPUT "${NRC_GRADIENT_SCALAR_DISPATCH_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.2"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_scalar_dispatch.comp"
                "-o"
                "${NRC_GRADIENT_SCALAR_DISPATCH_SPV}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_scalar_dispatch.comp"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcGradientScalar.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/scalar/NrcNNScalar.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
            VERBATIM)
        add_custom_target(nrc_gradient_scalar_dispatch_shader
            DEPENDS "${NRC_GRADIENT_SCALAR_DISPATCH_SPV}")

        add_executable(nrc_vulkan_gradient_scalar_test
            vulkan_gradient_scalar_test.cpp)
        target_link_libraries(nrc_vulkan_gradient_scalar_test PRIVATE nrc_vulkan)
        add_dependencies(nrc_vulkan_gradient_scalar_test nrc_gradient_scalar_dispatch_shader)
        add_test(NAME nrc_vulkan_gradient_scalar_test
            COMMAND nrc_vulkan_gradient_scalar_test "${NRC_GRADIENT_SCALAR_DISPATCH_SPV}")

        set(NRC_TRAINING_STEP_PREPARE_SPV "${CMAKE_CURRENT_BINARY_DIR}/nrc_training_step_prepare.spv")
        add_custom_command(
            OUTPUT "${NRC_TRAINING_STEP_PREPARE_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.1"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_train_prepare_dispatch.comp"
                "-o"
                "${NRC_TRAINING_STEP_PREPARE_SPV}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_train_prepare_dispatch.comp"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcTrainPrepare.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
            VERBATIM)

        set(NRC_TRAINING_STEP_OPTIMIZE_SPV "${CMAKE_CURRENT_BINARY_DIR}/nrc_training_step_optimize.spv")
        add_custom_command(
            OUTPUT "${NRC_TRAINING_STEP_OPTIMIZE_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.1"
                "-DNRC_OPTIMIZE_WRITE_USE_WEIGHTS=1"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_optimize_dispatch.comp"
                "-o"
                "${NRC_TRAINING_STEP_OPTIMIZE_SPV}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_optimize_dispatch.comp"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcOptimize.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
            VERBATIM)

        add_custom_target(nrc_training_step_shaders
            DEPENDS
                "${NRC_TRAINING_STEP_PREPARE_SPV}"
                "${NRC_GRADIENT_SCALAR_DISPATCH_SPV}"
                "${NRC_TRAINING_STEP_OPTIMIZE_SPV}")

        add_executable(nrc_vulkan_training_step_test
            vulkan_training_step_test.cpp)
        target_link_libraries(nrc_vulkan_training_step_test PRIVATE nrc_vulkan)
        add_dependencies(nrc_vulkan_training_step_test nrc_training_step_shaders)
        add_test(NAME nrc_vulkan_training_step_test
            COMMAND nrc_vulkan_training_step_test
                "${NRC_TRAINING_STEP_PREPARE_SPV}"
                "${NRC_GRADIENT_SCALAR_DISPATCH_SPV}"
                "${NRC_TRAINING_STEP_OPTIMIZE_SPV}")

        set(NRC_GRADIENT_KHR_DISPATCH_SG32_SPV "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_dispatch_sg32.spv")
        add_custom_command(
            OUTPUT "${NRC_GRADIENT_KHR_DISPATCH_SG32_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.3"
                "-DNRC_KHR_SUBGROUP_SIZE=32"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "-o"
                "${NRC_GRADIENT_KHR_DISPATCH_SG32_SPV}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcGradientKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
            VERBATIM)

        set(NRC_GRADIENT_KHR_DISPATCH_SG16_SPV "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_dispatch_sg16.spv")
        add_custom_command(
            OUTPUT "${NRC_GRADIENT_KHR_DISPATCH_SG16_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.3"
                "-DNRC_KHR_SUBGROUP_SIZE=16"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "-o"
                "${NRC_GRADIENT_KHR_DISPATCH_SG16_SPV}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcGradientKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
            VERBATIM)

        set(NRC_GRADIENT_KHR_DISPATCH_M8_SG32_SPV "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_dispatch_m8_sg32.spv")
        add_custom_command(
            OUTPUT "${NRC_GRADIENT_KHR_DISPATCH_M8_SG32_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.3"
                "-DNRC_KHR_SUBGROUP_SIZE=32"
                "-DNRC_KHR_COOPMAT_M=8"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "-o"
                "${NRC_GRADIENT_KHR_DISPATCH_M8_SG32_SPV}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcGradientKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
            VERBATIM)

        add_custom_target(nrc_gradient_khr_dispatch_shaders
            DEPENDS
                "${NRC_GRADIENT_KHR_DISPATCH_SG32_SPV}"
                "${NRC_GRADIENT_KHR_DISPATCH_SG16_SPV}"
                "${NRC_GRADIENT_KHR_DISPATCH_M8_SG32_SPV}")

        set(NRC_GRADIENT_KHR_COOPMAT_OUTPUT_DISPATCH_SG32_SPV
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_output_dispatch_sg32.spv")
        add_custom_command(
            OUTPUT "${NRC_GRADIENT_KHR_COOPMAT_OUTPUT_DISPATCH_SG32_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.3"
                "-DNRC_KHR_SUBGROUP_SIZE=32"
                "-DNRC_KHR_USE_COOPMAT_OUTPUT_DW=1"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "-o"
                "${NRC_GRADIENT_KHR_COOPMAT_OUTPUT_DISPATCH_SG32_SPV}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcGradientKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
            VERBATIM)

        set(NRC_GRADIENT_KHR_COOPMAT_OUTPUT_DISPATCH_SG16_SPV
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_output_dispatch_sg16.spv")
        add_custom_command(
            OUTPUT "${NRC_GRADIENT_KHR_COOPMAT_OUTPUT_DISPATCH_SG16_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.3"
                "-DNRC_KHR_SUBGROUP_SIZE=16"
                "-DNRC_KHR_USE_COOPMAT_OUTPUT_DW=1"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "-o"
                "${NRC_GRADIENT_KHR_COOPMAT_OUTPUT_DISPATCH_SG16_SPV}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcGradientKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
            VERBATIM)

        set(NRC_GRADIENT_KHR_COOPMAT_OUTPUT_DISPATCH_M8_SG32_SPV
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_output_dispatch_m8_sg32.spv")
        add_custom_command(
            OUTPUT "${NRC_GRADIENT_KHR_COOPMAT_OUTPUT_DISPATCH_M8_SG32_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.3"
                "-DNRC_KHR_SUBGROUP_SIZE=32"
                "-DNRC_KHR_COOPMAT_M=8"
                "-DNRC_KHR_USE_COOPMAT_OUTPUT_DW=1"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "-o"
                "${NRC_GRADIENT_KHR_COOPMAT_OUTPUT_DISPATCH_M8_SG32_SPV}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcGradientKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
            VERBATIM)

        add_custom_target(nrc_gradient_khr_coopmat_output_dispatch_shaders
            DEPENDS
                "${NRC_GRADIENT_KHR_COOPMAT_OUTPUT_DISPATCH_SG32_SPV}"
                "${NRC_GRADIENT_KHR_COOPMAT_OUTPUT_DISPATCH_SG16_SPV}"
                "${NRC_GRADIENT_KHR_COOPMAT_OUTPUT_DISPATCH_M8_SG32_SPV}")

        set(NRC_GRADIENT_KHR_COOPMAT_FINAL_HIDDEN_DISPATCH_SG32_SPV
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_final_hidden_dispatch_sg32.spv")
        add_custom_command(
            OUTPUT "${NRC_GRADIENT_KHR_COOPMAT_FINAL_HIDDEN_DISPATCH_SG32_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.3"
                "-DNRC_KHR_SUBGROUP_SIZE=32"
                "-DNRC_KHR_USE_COOPMAT_FINAL_HIDDEN_DW=1"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "-o"
                "${NRC_GRADIENT_KHR_COOPMAT_FINAL_HIDDEN_DISPATCH_SG32_SPV}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcGradientKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
            VERBATIM)

        set(NRC_GRADIENT_KHR_COOPMAT_FINAL_HIDDEN_DISPATCH_SG16_SPV
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_final_hidden_dispatch_sg16.spv")
        add_custom_command(
            OUTPUT "${NRC_GRADIENT_KHR_COOPMAT_FINAL_HIDDEN_DISPATCH_SG16_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.3"
                "-DNRC_KHR_SUBGROUP_SIZE=16"
                "-DNRC_KHR_USE_COOPMAT_FINAL_HIDDEN_DW=1"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "-o"
                "${NRC_GRADIENT_KHR_COOPMAT_FINAL_HIDDEN_DISPATCH_SG16_SPV}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcGradientKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
            VERBATIM)

        set(NRC_GRADIENT_KHR_COOPMAT_FINAL_HIDDEN_DISPATCH_M8_SG32_SPV
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_final_hidden_dispatch_m8_sg32.spv")
        add_custom_command(
            OUTPUT "${NRC_GRADIENT_KHR_COOPMAT_FINAL_HIDDEN_DISPATCH_M8_SG32_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.3"
                "-DNRC_KHR_SUBGROUP_SIZE=32"
                "-DNRC_KHR_COOPMAT_M=8"
                "-DNRC_KHR_USE_COOPMAT_FINAL_HIDDEN_DW=1"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "-o"
                "${NRC_GRADIENT_KHR_COOPMAT_FINAL_HIDDEN_DISPATCH_M8_SG32_SPV}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcGradientKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
            VERBATIM)

        add_custom_target(nrc_gradient_khr_coopmat_final_hidden_dispatch_shaders
            DEPENDS
                "${NRC_GRADIENT_KHR_COOPMAT_FINAL_HIDDEN_DISPATCH_SG32_SPV}"
                "${NRC_GRADIENT_KHR_COOPMAT_FINAL_HIDDEN_DISPATCH_SG16_SPV}"
                "${NRC_GRADIENT_KHR_COOPMAT_FINAL_HIDDEN_DISPATCH_M8_SG32_SPV}")

        set(NRC_GRADIENT_KHR_COOPMAT_HIDDEN3_DISPATCH_SG32_SPV
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_hidden3_dispatch_sg32.spv")
        add_custom_command(
            OUTPUT "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN3_DISPATCH_SG32_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.3"
                "-DNRC_KHR_SUBGROUP_SIZE=32"
                "-DNRC_KHR_USE_COOPMAT_HIDDEN3_DW=1"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "-o"
                "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN3_DISPATCH_SG32_SPV}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcGradientKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
            VERBATIM)

        set(NRC_GRADIENT_KHR_COOPMAT_HIDDEN3_DISPATCH_SG16_SPV
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_hidden3_dispatch_sg16.spv")
        add_custom_command(
            OUTPUT "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN3_DISPATCH_SG16_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.3"
                "-DNRC_KHR_SUBGROUP_SIZE=16"
                "-DNRC_KHR_USE_COOPMAT_HIDDEN3_DW=1"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "-o"
                "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN3_DISPATCH_SG16_SPV}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcGradientKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
            VERBATIM)

        set(NRC_GRADIENT_KHR_COOPMAT_HIDDEN3_DISPATCH_M8_SG32_SPV
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_hidden3_dispatch_m8_sg32.spv")
        add_custom_command(
            OUTPUT "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN3_DISPATCH_M8_SG32_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.3"
                "-DNRC_KHR_SUBGROUP_SIZE=32"
                "-DNRC_KHR_COOPMAT_M=8"
                "-DNRC_KHR_USE_COOPMAT_HIDDEN3_DW=1"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "-o"
                "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN3_DISPATCH_M8_SG32_SPV}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcGradientKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
            VERBATIM)

        add_custom_target(nrc_gradient_khr_coopmat_hidden3_dispatch_shaders
            DEPENDS
                "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN3_DISPATCH_SG32_SPV}"
                "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN3_DISPATCH_SG16_SPV}"
                "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN3_DISPATCH_M8_SG32_SPV}")

        set(NRC_GRADIENT_KHR_COOPMAT_HIDDEN2_DISPATCH_SG32_SPV
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_hidden2_dispatch_sg32.spv")
        add_custom_command(
            OUTPUT "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN2_DISPATCH_SG32_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.3"
                "-DNRC_KHR_SUBGROUP_SIZE=32"
                "-DNRC_KHR_USE_COOPMAT_HIDDEN2_DW=1"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "-o"
                "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN2_DISPATCH_SG32_SPV}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcGradientKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
            VERBATIM)

        set(NRC_GRADIENT_KHR_COOPMAT_HIDDEN2_DISPATCH_SG16_SPV
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_hidden2_dispatch_sg16.spv")
        add_custom_command(
            OUTPUT "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN2_DISPATCH_SG16_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.3"
                "-DNRC_KHR_SUBGROUP_SIZE=16"
                "-DNRC_KHR_USE_COOPMAT_HIDDEN2_DW=1"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "-o"
                "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN2_DISPATCH_SG16_SPV}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcGradientKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
            VERBATIM)

        set(NRC_GRADIENT_KHR_COOPMAT_HIDDEN2_DISPATCH_M8_SG32_SPV
            "${CMAKE_CURRENT_BINARY_DIR}/nrc_gradient_khr_coopmat_hidden2_dispatch_m8_sg32.spv")
        add_custom_command(
            OUTPUT "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN2_DISPATCH_M8_SG32_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.3"
                "-DNRC_KHR_SUBGROUP_SIZE=32"
                "-DNRC_KHR_COOPMAT_M=8"
                "-DNRC_KHR_USE_COOPMAT_HIDDEN2_DW=1"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "-o"
                "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN2_DISPATCH_M8_SG32_SPV}"
            DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_gradient_khr_dispatch.comp"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcGradientKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
                "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
            VERBATIM)

        add_custom_target(nrc_gradient_khr_coopmat_hidden2_dispatch_shaders
            DEPENDS
                "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN2_DISPATCH_SG32_SPV}"
                "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN2_DISPATCH_SG16_SPV}"
                "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN2_DISPATCH_M8_SG32_SPV}")

        add_executable(nrc_vulkan_gradient_khr_test
            vulkan_gradient_khr_test.cpp)
        target_link_libraries(nrc_vulkan_gradient_khr_test PRIVATE nrc_vulkan)
        add_dependencies(nrc_vulkan_gradient_khr_test
            nrc_gradient_khr_dispatch_shaders
            nrc_gradient_khr_coopmat_output_dispatch_shaders
            nrc_gradient_khr_coopmat_final_hidden_dispatch_shaders
            nrc_gradient_khr_coopmat_hidden3_dispatch_shaders
            nrc_gradient_khr_coopmat_hidden2_dispatch_shaders)
        add_test(NAME nrc_vulkan_gradient_khr_test
            COMMAND nrc_vulkan_gradient_khr_test
                "${NRC_GRADIENT_KHR_DISPATCH_SG32_SPV}"
                "${NRC_GRADIENT_KHR_DISPATCH_SG16_SPV}"
                "${NRC_GRADIENT_KHR_DISPATCH_M8_SG32_SPV}")
        add_test(NAME nrc_vulkan_gradient_khr_coopmat_output_test
            COMMAND nrc_vulkan_gradient_khr_test
                "${NRC_GRADIENT_KHR_COOPMAT_OUTPUT_DISPATCH_SG32_SPV}"
                "${NRC_GRADIENT_KHR_COOPMAT_OUTPUT_DISPATCH_SG16_SPV}"
                "${NRC_GRADIENT_KHR_COOPMAT_OUTPUT_DISPATCH_M8_SG32_SPV}"
                "--coopmat-output")
        add_test(NAME nrc_vulkan_gradient_khr_coopmat_final_hidden_test
            COMMAND nrc_vulkan_gradient_khr_test
                "${NRC_GRADIENT_KHR_COOPMAT_FINAL_HIDDEN_DISPATCH_SG32_SPV}"
                "${NRC_GRADIENT_KHR_COOPMAT_FINAL_HIDDEN_DISPATCH_SG16_SPV}"
                "${NRC_GRADIENT_KHR_COOPMAT_FINAL_HIDDEN_DISPATCH_M8_SG32_SPV}"
                "--coopmat-final-hidden")
        add_test(NAME nrc_vulkan_gradient_khr_coopmat_hidden3_test
            COMMAND nrc_vulkan_gradient_khr_test
                "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN3_DISPATCH_SG32_SPV}"
                "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN3_DISPATCH_SG16_SPV}"
                "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN3_DISPATCH_M8_SG32_SPV}"
                "--coopmat-hidden3")
        add_test(NAME nrc_vulkan_gradient_khr_coopmat_hidden2_test
            COMMAND nrc_vulkan_gradient_khr_test
                "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN2_DISPATCH_SG32_SPV}"
                "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN2_DISPATCH_SG16_SPV}"
                "${NRC_GRADIENT_KHR_COOPMAT_HIDDEN2_DISPATCH_M8_SG32_SPV}"
                "--coopmat-hidden2")

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

        set(NRC_KHR_DISPATCH_M8_SG32_SPV "${CMAKE_CURRENT_BINARY_DIR}/nrc_khr_dispatch_m8_sg32.spv")
        add_custom_command(
            OUTPUT "${NRC_KHR_DISPATCH_M8_SG32_SPV}"
            COMMAND "${NRC_GLSLC}"
                "--target-env=vulkan1.3"
                "-DNRC_KHR_SUBGROUP_SIZE=32"
                "-DNRC_KHR_COOPMAT_M=8"
                "-I${PROJECT_SOURCE_DIR}/shaders"
                "${CMAKE_CURRENT_SOURCE_DIR}/shader_compile/nrc_khr_dispatch.comp"
                "-o"
                "${NRC_KHR_DISPATCH_M8_SG32_SPV}"
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
                "${NRC_KHR_DISPATCH_SG16_SPV}"
                "${NRC_KHR_DISPATCH_M8_SG32_SPV}")

        add_executable(nrc_vulkan_khr_dispatch_test
            vulkan_khr_dispatch_test.cpp)
        target_link_libraries(nrc_vulkan_khr_dispatch_test PRIVATE nrc_vulkan)
        add_dependencies(nrc_vulkan_khr_dispatch_test nrc_khr_dispatch_shaders)
        add_test(NAME nrc_vulkan_khr_dispatch_test
            COMMAND nrc_vulkan_khr_dispatch_test
                "${NRC_KHR_DISPATCH_SG32_SPV}"
                "${NRC_KHR_DISPATCH_SG16_SPV}"
                "${NRC_KHR_DISPATCH_M8_SG32_SPV}")
    endif()
else()
    message(WARNING "glslc was not found; shader compile smoke test will not be registered")
endif()

if(TARGET nrc_reference_path_tracer)
    add_test(NAME nrc_sample_reference_path_tracer_smoke
        COMMAND nrc_reference_path_tracer)
endif()
