# 16_ray_query_nrc_live - Ray Query + live NRC integration
#
# This sample is a clean NRC integration target based on 16_ray_query.

# Setup with RT common sources and ray query capability
setup_rt_tutorial_sample(
    USE_RT_COMMON
    CAPABILITIES spvRayQueryKHR
) 

if(TARGET nrc_vulkan)
    target_link_libraries(${RT_TUTORIAL_TARGET} PRIVATE nrc_core nrc_vulkan nrc_shaders)
    target_compile_definitions(${RT_TUTORIAL_TARGET} PRIVATE NRC_AVAILABLE)

    set(NRC_LIVE_GLSL_FILES
        "${CMAKE_CURRENT_LIST_DIR}/shaders/nrc_demo_producer.comp"
        "${CMAKE_CURRENT_LIST_DIR}/shaders/nrc_demo_inference_prepare.comp"
        "${CMAKE_CURRENT_LIST_DIR}/shaders/nrc_demo_train_prepare.comp"
        "${CMAKE_CURRENT_LIST_DIR}/shaders/nrc_demo_optimize.comp"
        "${CMAKE_CURRENT_LIST_DIR}/shaders/nrc_demo_visualize.comp")

    set(NRC_LIVE_GLSL_HEADERS "")
    foreach(NRC_LIVE_SHADER ${NRC_LIVE_GLSL_FILES})
        get_filename_component(NRC_LIVE_SHADER_NAME "${NRC_LIVE_SHADER}" NAME)
        string(REPLACE "." "_" NRC_LIVE_SHADER_VARNAME "${NRC_LIVE_SHADER_NAME}")
        set(NRC_LIVE_SHADER_HEADER "${CMAKE_CURRENT_LIST_DIR}/_autogen/${NRC_LIVE_SHADER_NAME}.h")
        set(NRC_LIVE_SHADER_SPV "${CMAKE_CURRENT_LIST_DIR}/_autogen/${NRC_LIVE_SHADER_NAME}.spv")
        add_custom_command(
            OUTPUT "${NRC_LIVE_SHADER_HEADER}" "${NRC_LIVE_SHADER_SPV}"
            COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}
                    -D__GLSL__
                    --target-env vulkan1.2
                    --vn ${NRC_LIVE_SHADER_VARNAME}
                    -I${NRC_ROOT}/shaders
                    -o "${NRC_LIVE_SHADER_HEADER}"
                    "${NRC_LIVE_SHADER}"
            COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}
                    -D__GLSL__
                    --target-env vulkan1.2
                    -I${NRC_ROOT}/shaders
                    -o "${NRC_LIVE_SHADER_SPV}"
                    "${NRC_LIVE_SHADER}"
            MAIN_DEPENDENCY "${NRC_LIVE_SHADER}"
            COMMENT "Compiling NRC live GLSL shader: ${NRC_LIVE_SHADER_NAME}")
        list(APPEND NRC_LIVE_GLSL_HEADERS "${NRC_LIVE_SHADER_HEADER}")
    endforeach()

    set(NRC_LIVE_GLSLC "${Vulkan_GLSLC_EXECUTABLE}")
    if(NOT NRC_LIVE_GLSLC)
        find_program(NRC_LIVE_GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin")
    endif()

    if(NRC_LIVE_GLSLC)
        set(NRC_LIVE_INFERENCE_KHR_SG32_SPV "${CMAKE_CURRENT_LIST_DIR}/_autogen/nrc_live_inference_khr_sg32.spv")
        set(NRC_LIVE_INFERENCE_KHR_SG16_SPV "${CMAKE_CURRENT_LIST_DIR}/_autogen/nrc_live_inference_khr_sg16.spv")
        set(NRC_LIVE_INFERENCE_KHR_M8_SG32_SPV "${CMAKE_CURRENT_LIST_DIR}/_autogen/nrc_live_inference_khr_m8_sg32.spv")
        set(NRC_LIVE_GRADIENT_KHR_SG32_SPV "${CMAKE_CURRENT_LIST_DIR}/_autogen/nrc_live_gradient_khr_sg32.spv")
        set(NRC_LIVE_GRADIENT_KHR_SG16_SPV "${CMAKE_CURRENT_LIST_DIR}/_autogen/nrc_live_gradient_khr_sg16.spv")
        set(NRC_LIVE_GRADIENT_KHR_M8_SG32_SPV "${CMAKE_CURRENT_LIST_DIR}/_autogen/nrc_live_gradient_khr_m8_sg32.spv")

        function(add_nrc_live_khr_inference output_spv)
            add_custom_command(
                OUTPUT "${output_spv}"
                COMMAND "${NRC_LIVE_GLSLC}"
                        -O
                        --target-env=vulkan1.3
                        ${ARGN}
                        "-I${NRC_ROOT}/shaders"
                        "-I${CMAKE_CURRENT_LIST_DIR}/shaders"
                        "${CMAKE_CURRENT_LIST_DIR}/shaders/nrc_demo_inference_khr.comp"
                        -o "${output_spv}"
                DEPENDS
                    "${CMAKE_CURRENT_LIST_DIR}/shaders/nrc_demo_inference_khr.comp"
                    "${NRC_ROOT}/shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl"
                    "${NRC_ROOT}/shaders/nrc/common/NrcEncoding.glsl"
                    "${NRC_ROOT}/shaders/nrc/common/NrcTypes.glsl"
                    "${NRC_ROOT}/shaders/nrc/common/NrcConstants.glsl"
                COMMENT "Compiling NRC live KHR inference shader: ${output_spv}"
                VERBATIM)
        endfunction()

        function(add_nrc_live_khr_gradient output_spv)
            add_custom_command(
                OUTPUT "${output_spv}"
                COMMAND "${NRC_LIVE_GLSLC}"
                        -O
                        --target-env=vulkan1.3
                        ${ARGN}
                        -DNRC_KHR_RECOMPUTE_BACKPROP=1
                        -DNRC_KHR_USE_COOPMAT_ALL_DW=1
                        -DNRC_KHR_USE_COOPMAT_ALL_DA=1
                        "-I${NRC_ROOT}/shaders"
                        "-I${CMAKE_CURRENT_LIST_DIR}/shaders"
                        "${CMAKE_CURRENT_LIST_DIR}/shaders/nrc_demo_gradient_khr.comp"
                        -o "${output_spv}"
                DEPENDS
                    "${CMAKE_CURRENT_LIST_DIR}/shaders/nrc_demo_gradient_khr.comp"
                    "${NRC_ROOT}/shaders/nrc/passes/NrcGradientKhr.glsl"
                    "${NRC_ROOT}/shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl"
                    "${NRC_ROOT}/shaders/nrc/common/NrcEncoding.glsl"
                    "${NRC_ROOT}/shaders/nrc/common/NrcTypes.glsl"
                    "${NRC_ROOT}/shaders/nrc/common/NrcConstants.glsl"
                COMMENT "Compiling NRC live KHR gradient shader: ${output_spv}"
                VERBATIM)
        endfunction()

        add_nrc_live_khr_inference("${NRC_LIVE_INFERENCE_KHR_SG32_SPV}" -DNRC_KHR_SUBGROUP_SIZE=32)
        add_nrc_live_khr_inference("${NRC_LIVE_INFERENCE_KHR_SG16_SPV}" -DNRC_KHR_SUBGROUP_SIZE=16)
        add_nrc_live_khr_inference("${NRC_LIVE_INFERENCE_KHR_M8_SG32_SPV}" -DNRC_KHR_SUBGROUP_SIZE=32 -DNRC_KHR_COOPMAT_M=8)

        add_nrc_live_khr_gradient("${NRC_LIVE_GRADIENT_KHR_SG32_SPV}" -DNRC_KHR_SUBGROUP_SIZE=32)
        add_nrc_live_khr_gradient("${NRC_LIVE_GRADIENT_KHR_SG16_SPV}" -DNRC_KHR_SUBGROUP_SIZE=16)
        add_nrc_live_khr_gradient("${NRC_LIVE_GRADIENT_KHR_M8_SG32_SPV}" -DNRC_KHR_SUBGROUP_SIZE=32 -DNRC_KHR_COOPMAT_M=8)

        set(NRC_LIVE_KHR_SPV_FILES
            "${NRC_LIVE_INFERENCE_KHR_SG32_SPV}"
            "${NRC_LIVE_INFERENCE_KHR_SG16_SPV}"
            "${NRC_LIVE_INFERENCE_KHR_M8_SG32_SPV}"
            "${NRC_LIVE_GRADIENT_KHR_SG32_SPV}"
            "${NRC_LIVE_GRADIENT_KHR_SG16_SPV}"
            "${NRC_LIVE_GRADIENT_KHR_M8_SG32_SPV}")
        add_custom_target(${RT_TUTORIAL_TARGET}_nrc_khr_spv DEPENDS ${NRC_LIVE_KHR_SPV_FILES})
        add_dependencies(${RT_TUTORIAL_TARGET} ${RT_TUTORIAL_TARGET}_nrc_khr_spv)
        target_compile_definitions(${RT_TUTORIAL_TARGET} PRIVATE
            NRC_LIVE_INFERENCE_KHR_SG32_SPV="${NRC_LIVE_INFERENCE_KHR_SG32_SPV}"
            NRC_LIVE_INFERENCE_KHR_SG16_SPV="${NRC_LIVE_INFERENCE_KHR_SG16_SPV}"
            NRC_LIVE_INFERENCE_KHR_M8_SG32_SPV="${NRC_LIVE_INFERENCE_KHR_M8_SG32_SPV}"
            NRC_LIVE_GRADIENT_KHR_SG32_SPV="${NRC_LIVE_GRADIENT_KHR_SG32_SPV}"
            NRC_LIVE_GRADIENT_KHR_SG16_SPV="${NRC_LIVE_GRADIENT_KHR_SG16_SPV}"
            NRC_LIVE_GRADIENT_KHR_M8_SG32_SPV="${NRC_LIVE_GRADIENT_KHR_M8_SG32_SPV}")
    else()
        message(WARNING "glslc was not found; ${RT_TUTORIAL_TARGET} cannot build NRC KHR shaders")
    endif()

    source_group("Shaders/NRC Live" FILES ${NRC_LIVE_GLSL_FILES})
    source_group("Shaders/Compiled/NRC Live" FILES ${NRC_LIVE_GLSL_HEADERS})
    target_sources(${RT_TUTORIAL_TARGET} PRIVATE ${NRC_LIVE_GLSL_FILES} ${NRC_LIVE_GLSL_HEADERS})
else()
    message(WARNING "NRC target was not found; ${RT_TUTORIAL_TARGET} will build without NRC")
endif()
