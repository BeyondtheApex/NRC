find_program(NRC_SAMPLE_GLSLC
    NAMES glslc glslc.exe
    HINTS
        "$ENV{VULKAN_SDK}/Bin"
        "D:/VulkanSDK/1.4.335.0/Bin")

if(TARGET nrc_vulkan AND NRC_SAMPLE_GLSLC)
    set(NRC_SAMPLE_PRODUCER_SPV "${CMAKE_CURRENT_BINARY_DIR}/nrc_sample_producer.spv")
    add_custom_command(
        OUTPUT "${NRC_SAMPLE_PRODUCER_SPV}"
        COMMAND "${NRC_SAMPLE_GLSLC}"
            "--target-env=vulkan1.1"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "-I${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sample_producer.comp"
            "-o"
            "${NRC_SAMPLE_PRODUCER_SPV}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sample_producer.comp"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sample_scene.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcQueueOps.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcQueue.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
        VERBATIM)

    set(NRC_SAMPLE_INFERENCE_SPV "${CMAKE_CURRENT_BINARY_DIR}/nrc_sample_inference_scalar.spv")
    add_custom_command(
        OUTPUT "${NRC_SAMPLE_INFERENCE_SPV}"
        COMMAND "${NRC_SAMPLE_GLSLC}"
            "--target-env=vulkan1.1"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "-I${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sample_inference_scalar.comp"
            "-o"
            "${NRC_SAMPLE_INFERENCE_SPV}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sample_inference_scalar.comp"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sample_scene.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcInferenceScalar.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/scalar/NrcNNScalar.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
        VERBATIM)

    set(NRC_SAMPLE_TRAIN_PREPARE_SPV "${CMAKE_CURRENT_BINARY_DIR}/nrc_sample_train_prepare.spv")
    add_custom_command(
        OUTPUT "${NRC_SAMPLE_TRAIN_PREPARE_SPV}"
        COMMAND "${NRC_SAMPLE_GLSLC}"
            "--target-env=vulkan1.1"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "-I${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sample_train_prepare.comp"
            "-o"
            "${NRC_SAMPLE_TRAIN_PREPARE_SPV}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sample_train_prepare.comp"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcTrainPrepare.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
        VERBATIM)

    set(NRC_SAMPLE_GRADIENT_SPV "${CMAKE_CURRENT_BINARY_DIR}/nrc_sample_gradient_scalar.spv")
    add_custom_command(
        OUTPUT "${NRC_SAMPLE_GRADIENT_SPV}"
        COMMAND "${NRC_SAMPLE_GLSLC}"
            "--target-env=vulkan1.2"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "-I${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sample_gradient_scalar.comp"
            "-o"
            "${NRC_SAMPLE_GRADIENT_SPV}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sample_gradient_scalar.comp"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sample_scene.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcGradientScalar.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/scalar/NrcNNScalar.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
        VERBATIM)

    set(NRC_SAMPLE_GRADIENT_KHR_SG32_SPV
        "${CMAKE_CURRENT_BINARY_DIR}/nrc_sample_gradient_khr_sg32.spv")
    add_custom_command(
        OUTPUT "${NRC_SAMPLE_GRADIENT_KHR_SG32_SPV}"
        COMMAND "${NRC_SAMPLE_GLSLC}"
            "-O"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=32"
            "-DNRC_KHR_RECOMPUTE_BACKPROP=1"
            "-DNRC_KHR_USE_COOPMAT_ALL_DW=1"
            "-DNRC_KHR_USE_COOPMAT_ALL_DA=1"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "-I${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sample_gradient_khr.comp"
            "-o"
            "${NRC_SAMPLE_GRADIENT_KHR_SG32_SPV}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sample_gradient_khr.comp"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sample_scene.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcGradientKhr.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
        VERBATIM)

    set(NRC_SAMPLE_GRADIENT_KHR_SG16_SPV
        "${CMAKE_CURRENT_BINARY_DIR}/nrc_sample_gradient_khr_sg16.spv")
    add_custom_command(
        OUTPUT "${NRC_SAMPLE_GRADIENT_KHR_SG16_SPV}"
        COMMAND "${NRC_SAMPLE_GLSLC}"
            "-O"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=16"
            "-DNRC_KHR_RECOMPUTE_BACKPROP=1"
            "-DNRC_KHR_USE_COOPMAT_ALL_DW=1"
            "-DNRC_KHR_USE_COOPMAT_ALL_DA=1"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "-I${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sample_gradient_khr.comp"
            "-o"
            "${NRC_SAMPLE_GRADIENT_KHR_SG16_SPV}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sample_gradient_khr.comp"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sample_scene.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcGradientKhr.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
        VERBATIM)

    set(NRC_SAMPLE_GRADIENT_KHR_M8_SG32_SPV
        "${CMAKE_CURRENT_BINARY_DIR}/nrc_sample_gradient_khr_m8_sg32.spv")
    add_custom_command(
        OUTPUT "${NRC_SAMPLE_GRADIENT_KHR_M8_SG32_SPV}"
        COMMAND "${NRC_SAMPLE_GLSLC}"
            "-O"
            "--target-env=vulkan1.3"
            "-DNRC_KHR_SUBGROUP_SIZE=32"
            "-DNRC_KHR_COOPMAT_M=8"
            "-DNRC_KHR_RECOMPUTE_BACKPROP=1"
            "-DNRC_KHR_USE_COOPMAT_ALL_DW=1"
            "-DNRC_KHR_USE_COOPMAT_ALL_DA=1"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "-I${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sample_gradient_khr.comp"
            "-o"
            "${NRC_SAMPLE_GRADIENT_KHR_M8_SG32_SPV}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sample_gradient_khr.comp"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sample_scene.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcGradientKhr.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcEncoding.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
        VERBATIM)

    set(NRC_SAMPLE_OPTIMIZE_SPV "${CMAKE_CURRENT_BINARY_DIR}/nrc_sample_optimize.spv")
    add_custom_command(
        OUTPUT "${NRC_SAMPLE_OPTIMIZE_SPV}"
        COMMAND "${NRC_SAMPLE_GLSLC}"
            "--target-env=vulkan1.1"
            "-I${PROJECT_SOURCE_DIR}/shaders"
            "-I${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sample_optimize.comp"
            "-o"
            "${NRC_SAMPLE_OPTIMIZE_SPV}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sample_optimize.comp"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/passes/NrcOptimize.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcTypes.glsl"
            "${PROJECT_SOURCE_DIR}/shaders/nrc/common/NrcConstants.glsl"
        VERBATIM)

    add_custom_target(nrc_reference_path_tracer_shaders
        DEPENDS
            "${NRC_SAMPLE_PRODUCER_SPV}"
            "${NRC_SAMPLE_INFERENCE_SPV}"
            "${NRC_SAMPLE_TRAIN_PREPARE_SPV}"
            "${NRC_SAMPLE_GRADIENT_SPV}"
            "${NRC_SAMPLE_GRADIENT_KHR_SG32_SPV}"
            "${NRC_SAMPLE_GRADIENT_KHR_SG16_SPV}"
            "${NRC_SAMPLE_GRADIENT_KHR_M8_SG32_SPV}"
            "${NRC_SAMPLE_OPTIMIZE_SPV}")

    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/sample_config.hpp.in"
        "${CMAKE_CURRENT_BINARY_DIR}/sample_config.hpp"
        @ONLY)

    add_executable(nrc_reference_path_tracer)
    target_sources(nrc_reference_path_tracer PRIVATE
        sample.cpp)
    target_include_directories(nrc_reference_path_tracer PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}")
    target_link_libraries(nrc_reference_path_tracer PRIVATE
        nrc_vulkan
        nrc_shaders)
    add_dependencies(nrc_reference_path_tracer nrc_reference_path_tracer_shaders)
else()
    add_executable(nrc_reference_path_tracer)
    target_sources(nrc_reference_path_tracer PRIVATE
        placeholder.cpp)
    target_link_libraries(nrc_reference_path_tracer PRIVATE
        nrc_core
        nrc_shaders)
endif()
