#include "nrc/vulkan/nrc_vulkan_barriers.hpp"
#include "nrc/vulkan/nrc_vulkan_capabilities.hpp"
#include "nrc/vulkan/nrc_vulkan_descriptors.hpp"
#include "nrc/vulkan/nrc_vulkan_pipeline.hpp"
#include "nrc/vulkan/nrc_vulkan_shader.hpp"

#include <array>
#include <cstdlib>
#include <iostream>

namespace {

bool require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }
    return true;
}

} // namespace

int main() {
    bool ok = true;

    ok &= require(nrc::vulkan::map_component_type(VK_COMPONENT_TYPE_FLOAT16_KHR) ==
                      nrc::ComponentType::Float16,
                  "map fp16 component type");
    ok &= require(nrc::vulkan::map_component_type(VK_COMPONENT_TYPE_FLOAT32_KHR) ==
                      nrc::ComponentType::Float32,
                  "map fp32 component type");

    const auto subgroupSizes = nrc::vulkan::enumerate_subgroup_sizes(16, 64, 32);
    ok &= require(subgroupSizes.size() == 3, "subgroup size count");
    ok &= require(subgroupSizes[0] == 16 && subgroupSizes[1] == 32 && subgroupSizes[2] == 64,
                  "subgroup size powers of two");

    VkCooperativeMatrixPropertiesKHR property{};
    property.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    property.MSize = 16;
    property.NSize = 16;
    property.KSize = 16;
    property.AType = VK_COMPONENT_TYPE_FLOAT16_KHR;
    property.BType = VK_COMPONENT_TYPE_FLOAT16_KHR;
    property.CType = VK_COMPONENT_TYPE_FLOAT16_KHR;
    property.ResultType = VK_COMPONENT_TYPE_FLOAT16_KHR;
    property.scope = VK_SCOPE_SUBGROUP_KHR;

    auto translated = nrc::vulkan::translate_cooperative_matrix_property(property, VK_SHADER_STAGE_COMPUTE_BIT);
    ok &= require(translated.has_value(), "translate compute subgroup cooperative matrix");
    if (translated.has_value()) {
        ok &= require(translated->m == 16 && translated->n == 16 && translated->k == 16,
                      "translated matrix dimensions");
        ok &= require(translated->aType == nrc::ComponentType::Float16,
                      "translated matrix component type");
        ok &= require(translated->computeStage, "translated compute stage");
    }

    auto noCompute = nrc::vulkan::translate_cooperative_matrix_property(property, VK_SHADER_STAGE_VERTEX_BIT);
    ok &= require(!noCompute.has_value(), "reject non-compute cooperative matrix property");

    property.scope = VK_SCOPE_WORKGROUP_KHR;
    auto noSubgroup = nrc::vulkan::translate_cooperative_matrix_property(property, VK_SHADER_STAGE_COMPUTE_BIT);
    ok &= require(!noSubgroup.has_value(), "reject non-subgroup cooperative matrix property");

    const nrc::vulkan::VulkanMemoryBarrier hostToCompute =
        nrc::vulkan::host_write_to_compute_shader_barrier(VK_ACCESS_SHADER_READ_BIT);
    ok &= require(hostToCompute.srcStageMask == VK_PIPELINE_STAGE_HOST_BIT,
                  "host to compute barrier source stage");
    ok &= require(hostToCompute.dstStageMask == VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  "host to compute barrier destination stage");
    ok &= require(hostToCompute.srcAccessMask == VK_ACCESS_HOST_WRITE_BIT,
                  "host to compute barrier source access");
    ok &= require(hostToCompute.dstAccessMask == VK_ACCESS_SHADER_READ_BIT,
                  "host to compute barrier destination access");

    const nrc::vulkan::VulkanMemoryBarrier shaderToHost =
        nrc::vulkan::compute_shader_write_to_host_read_barrier();
    ok &= require(shaderToHost.srcAccessMask == VK_ACCESS_SHADER_WRITE_BIT,
                  "shader to host barrier source access");
    ok &= require(shaderToHost.dstAccessMask == VK_ACCESS_HOST_READ_BIT,
                  "shader to host barrier destination access");

    const nrc::vulkan::VulkanImageBarrier colorBarrier =
        nrc::vulkan::color_image_barrier(VK_NULL_HANDLE,
                                         VK_IMAGE_LAYOUT_UNDEFINED,
                                         VK_IMAGE_LAYOUT_GENERAL,
                                         0,
                                         VK_ACCESS_SHADER_WRITE_BIT);
    ok &= require(colorBarrier.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED,
                  "color image barrier old layout");
    ok &= require(colorBarrier.newLayout == VK_IMAGE_LAYOUT_GENERAL,
                  "color image barrier new layout");
    ok &= require(colorBarrier.dstAccessMask == VK_ACCESS_SHADER_WRITE_BIT,
                  "color image barrier destination access");
    ok &= require(colorBarrier.subresourceRange.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT,
                  "color image barrier aspect mask");
    ok &= require(colorBarrier.subresourceRange.levelCount == 1,
                  "color image barrier level count");
    ok &= require(colorBarrier.subresourceRange.layerCount == 1,
                  "color image barrier layer count");

    const nrc::vulkan::VulkanValidationStep invalidBarrier =
        nrc::vulkan::cmd_memory_barrier(VK_NULL_HANDLE, hostToCompute);
    ok &= require(invalidBarrier.status == nrc::vulkan::VulkanValidationStepStatus::Failed,
                  "invalid memory barrier request fails safely");
    ok &= require(invalidBarrier.vkResult == VK_ERROR_INITIALIZATION_FAILED,
                  "invalid memory barrier request records initialization failure");

    const nrc::vulkan::VulkanValidationStep invalidPipelineBarrier =
        nrc::vulkan::cmd_pipeline_barrier({});
    ok &= require(invalidPipelineBarrier.status == nrc::vulkan::VulkanValidationStepStatus::Failed,
                  "invalid pipeline barrier request fails safely");
    ok &= require(invalidPipelineBarrier.vkResult == VK_ERROR_INITIALIZATION_FAILED,
                  "invalid pipeline barrier request records initialization failure");

    const nrc::vulkan::VulkanDescriptorSetLayoutCreateResult invalidDescriptorLayout =
        nrc::vulkan::create_storage_buffer_descriptor_set_layout({});
    ok &= require(invalidDescriptorLayout.layout == VK_NULL_HANDLE,
                  "invalid descriptor layout request returns null layout");
    ok &= require(invalidDescriptorLayout.validation.status ==
                      nrc::vulkan::VulkanValidationStepStatus::Failed,
                  "invalid descriptor layout request fails safely");
    ok &= require(invalidDescriptorLayout.validation.vkResult == VK_ERROR_INITIALIZATION_FAILED,
                  "invalid descriptor layout request records initialization failure");

    const nrc::vulkan::VulkanDescriptorSetLayoutCreateResult invalidGenericDescriptorLayout =
        nrc::vulkan::create_descriptor_set_layout({});
    ok &= require(invalidGenericDescriptorLayout.layout == VK_NULL_HANDLE,
                  "invalid generic descriptor layout request returns null layout");
    ok &= require(invalidGenericDescriptorLayout.validation.status ==
                      nrc::vulkan::VulkanValidationStepStatus::Failed,
                  "invalid generic descriptor layout request fails safely");
    ok &= require(invalidGenericDescriptorLayout.validation.vkResult == VK_ERROR_INITIALIZATION_FAILED,
                  "invalid generic descriptor layout request records initialization failure");

    const std::array<nrc::vulkan::VulkanStorageBufferBinding, 2> duplicateBindings = {{
        {0, VK_SHADER_STAGE_COMPUTE_BIT},
        {0, VK_SHADER_STAGE_COMPUTE_BIT},
    }};
    nrc::vulkan::VulkanStorageBufferDescriptorSetLayoutCreateInfo duplicateLayoutInfo{};
    duplicateLayoutInfo.bindings = duplicateBindings;
    const nrc::vulkan::VulkanDescriptorSetLayoutCreateResult duplicateDescriptorLayout =
        nrc::vulkan::create_storage_buffer_descriptor_set_layout(duplicateLayoutInfo);
    ok &= require(duplicateDescriptorLayout.validation.status ==
                      nrc::vulkan::VulkanValidationStepStatus::Failed,
                  "duplicate descriptor layout bindings fail safely");

    const std::array<nrc::vulkan::VulkanDescriptorBinding, 2> duplicateGenericBindings = {{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT},
    }};
    nrc::vulkan::VulkanDescriptorSetLayoutCreateInfo duplicateGenericLayoutInfo{};
    duplicateGenericLayoutInfo.bindings = duplicateGenericBindings;
    const nrc::vulkan::VulkanDescriptorSetLayoutCreateResult duplicateGenericDescriptorLayout =
        nrc::vulkan::create_descriptor_set_layout(duplicateGenericLayoutInfo);
    ok &= require(duplicateGenericDescriptorLayout.validation.status ==
                      nrc::vulkan::VulkanValidationStepStatus::Failed,
                  "duplicate generic descriptor layout bindings fail safely");

    const nrc::vulkan::VulkanValidationStep invalidDescriptorUpdate =
        nrc::vulkan::update_storage_buffer_descriptors({});
    ok &= require(invalidDescriptorUpdate.status == nrc::vulkan::VulkanValidationStepStatus::Failed,
                  "invalid descriptor update fails safely");
    ok &= require(invalidDescriptorUpdate.vkResult == VK_ERROR_INITIALIZATION_FAILED,
                  "invalid descriptor update records initialization failure");

    const nrc::vulkan::VulkanValidationStep invalidImageDescriptorUpdate =
        nrc::vulkan::update_storage_image_descriptors({});
    ok &= require(invalidImageDescriptorUpdate.status == nrc::vulkan::VulkanValidationStepStatus::Failed,
                  "invalid image descriptor update fails safely");
    ok &= require(invalidImageDescriptorUpdate.vkResult == VK_ERROR_INITIALIZATION_FAILED,
                  "invalid image descriptor update records initialization failure");

    const nrc::vulkan::VulkanShaderModuleCreateResult invalidShaderModule =
        nrc::vulkan::create_shader_module({});
    ok &= require(invalidShaderModule.shaderModule == VK_NULL_HANDLE,
                  "invalid shader module request returns null shader module");
    ok &= require(invalidShaderModule.validation.status ==
                      nrc::vulkan::VulkanValidationStepStatus::Failed,
                  "invalid shader module request fails safely");
    ok &= require(invalidShaderModule.validation.vkResult == VK_ERROR_INITIALIZATION_FAILED,
                  "invalid shader module request records initialization failure");

    const nrc::vulkan::VulkanComputePipelineCreateResult invalidBasicPipeline =
        nrc::vulkan::create_compute_pipeline({});
    ok &= require(invalidBasicPipeline.pipeline == VK_NULL_HANDLE,
                  "invalid basic pipeline request returns null pipeline");
    ok &= require(invalidBasicPipeline.validation.status ==
                      nrc::vulkan::VulkanValidationStepStatus::Failed,
                  "invalid basic pipeline request fails safely");
    ok &= require(invalidBasicPipeline.validation.vkResult == VK_ERROR_INITIALIZATION_FAILED,
                  "invalid basic pipeline request records initialization failure");

    const nrc::vulkan::VulkanComputePipelineCreateResult invalidPipeline =
        nrc::vulkan::create_profile_compute_pipeline({});
    ok &= require(invalidPipeline.pipeline == VK_NULL_HANDLE,
                  "invalid profile pipeline request returns null pipeline");
    ok &= require(invalidPipeline.validation.status ==
                      nrc::vulkan::VulkanValidationStepStatus::Failed,
                  "invalid profile pipeline request fails safely");
    ok &= require(invalidPipeline.validation.vkResult == VK_ERROR_INITIALIZATION_FAILED,
                  "invalid profile pipeline request records initialization failure");
    ok &= require(!invalidPipeline.validation.structuredException,
                  "invalid profile pipeline request does not record structured exception");

    nrc::vulkan::VulkanDeviceCapabilities validCapabilities{};
    validCapabilities.shaderFloat16 = true;
    validCapabilities.storageBuffer16BitAccess = true;
    validCapabilities.shaderBufferFloat32AtomicAdd = true;
    validCapabilities.subgroupSizeControl = true;
    validCapabilities.computeFullSubgroups = true;
    validCapabilities.khrCooperativeMatrix = true;
    validCapabilities.supportedSubgroupSizes = {32};
    validCapabilities.cooperativeMatrices = {{
        {
            16,
            16,
            16,
            nrc::ComponentType::Float16,
            nrc::ComponentType::Float16,
            nrc::ComponentType::Float16,
            nrc::ComponentType::Float16,
            nrc::MatrixScope::Subgroup,
            true,
        },
    }};

    const nrc::Profile *khrProfile =
        nrc::find_profile(nrc::ProfileId::KhrFp16M16N16K16Sg32Wg128);
    ok &= require(khrProfile != nullptr, "find KHR profile for Vulkan validation");
    if (khrProfile != nullptr) {
        const nrc::vulkan::VulkanProfileValidationReport report =
            nrc::vulkan::evaluate_vulkan_profile_validation(validCapabilities, {}, *khrProfile);
        ok &= require(report.usable, "Vulkan validation report is usable for supported profile");
        ok &= require(report.status == nrc::vulkan::VulkanProfileValidationStatus::Supported,
                      "Vulkan validation report records supported status");
        ok &= require(report.profileSupport.status == nrc::ProfileSupportStatus::Supported,
                      "Vulkan validation report carries capability support");
        ok &= require(report.shaderConfig != nullptr &&
                          report.shaderConfig->id == khrProfile->id,
                      "Vulkan validation report carries shader config");
        ok &= require(report.shaderModuleCreation.status ==
                          nrc::vulkan::VulkanValidationStepStatus::NotRun,
                      "Vulkan validation defaults shader module creation to not run");
        ok &= require(report.pipelineCreation.status ==
                          nrc::vulkan::VulkanValidationStepStatus::NotRun,
                      "Vulkan validation defaults pipeline creation to not run");
        ok &= require(report.numericalValidation.status ==
                          nrc::vulkan::VulkanValidationStepStatus::NotRun,
                      "Vulkan validation defaults numerical validation to not run");

        nrc::vulkan::VulkanProfileValidationRequirements requireShaderModule{};
        requireShaderModule.requireShaderModuleCreation = true;
        const nrc::vulkan::VulkanProfileValidationReport missingShaderModule =
            nrc::vulkan::evaluate_vulkan_profile_validation(
                validCapabilities,
                {},
                *khrProfile,
                requireShaderModule);
        ok &= require(!missingShaderModule.usable,
                      "required shader module creation blocks usable report when not run");
        ok &= require(missingShaderModule.status ==
                          nrc::vulkan::VulkanProfileValidationStatus::ShaderModuleCreationNotRun,
                      "Vulkan validation reports missing shader module creation");

        requireShaderModule.shaderModuleCreation.status = nrc::vulkan::VulkanValidationStepStatus::Passed;
        const nrc::vulkan::VulkanProfileValidationReport passedShaderModule =
            nrc::vulkan::evaluate_vulkan_profile_validation(
                validCapabilities,
                {},
                *khrProfile,
                requireShaderModule);
        ok &= require(passedShaderModule.usable,
                      "passed required shader module creation allows usable report");

        requireShaderModule.shaderModuleCreation.status = nrc::vulkan::VulkanValidationStepStatus::Skipped;
        const nrc::vulkan::VulkanProfileValidationReport skippedShaderModule =
            nrc::vulkan::evaluate_vulkan_profile_validation(
                validCapabilities,
                {},
                *khrProfile,
                requireShaderModule);
        ok &= require(skippedShaderModule.status ==
                          nrc::vulkan::VulkanProfileValidationStatus::ShaderModuleCreationSkipped,
                      "Vulkan validation reports skipped required shader module creation");

        requireShaderModule.shaderModuleCreation.status = nrc::vulkan::VulkanValidationStepStatus::Failed;
        requireShaderModule.shaderModuleCreation.vkResult = VK_ERROR_UNKNOWN;
        const nrc::vulkan::VulkanProfileValidationReport failedShaderModule =
            nrc::vulkan::evaluate_vulkan_profile_validation(
                validCapabilities,
                {},
                *khrProfile,
                requireShaderModule);
        ok &= require(failedShaderModule.status ==
                          nrc::vulkan::VulkanProfileValidationStatus::ShaderModuleCreationFailed,
                      "Vulkan validation reports failed shader module creation");
        ok &= require(failedShaderModule.shaderModuleCreation.vkResult == VK_ERROR_UNKNOWN,
                      "Vulkan validation preserves shader module VkResult");

        nrc::vulkan::VulkanProfileValidationRequirements requirePipeline{};
        requirePipeline.requirePipelineCreation = true;
        const nrc::vulkan::VulkanProfileValidationReport missingPipeline =
            nrc::vulkan::evaluate_vulkan_profile_validation(
                validCapabilities,
                {},
                *khrProfile,
                requirePipeline);
        ok &= require(!missingPipeline.usable,
                      "required pipeline creation blocks usable report when not run");
        ok &= require(missingPipeline.status ==
                          nrc::vulkan::VulkanProfileValidationStatus::PipelineCreationNotRun,
                      "Vulkan validation reports missing pipeline creation");

        requirePipeline.pipelineCreation.status = nrc::vulkan::VulkanValidationStepStatus::Passed;
        const nrc::vulkan::VulkanProfileValidationReport passedPipeline =
            nrc::vulkan::evaluate_vulkan_profile_validation(
                validCapabilities,
                {},
                *khrProfile,
                requirePipeline);
        ok &= require(passedPipeline.usable,
                      "passed required pipeline creation allows usable report");

        requirePipeline.pipelineCreation.status = nrc::vulkan::VulkanValidationStepStatus::Skipped;
        const nrc::vulkan::VulkanProfileValidationReport skippedPipeline =
            nrc::vulkan::evaluate_vulkan_profile_validation(
                validCapabilities,
                {},
                *khrProfile,
                requirePipeline);
        ok &= require(skippedPipeline.status ==
                          nrc::vulkan::VulkanProfileValidationStatus::PipelineCreationSkipped,
                      "Vulkan validation reports skipped required pipeline creation");

        requirePipeline.pipelineCreation.status = nrc::vulkan::VulkanValidationStepStatus::Failed;
        requirePipeline.pipelineCreation.structuredException = false;
        requirePipeline.pipelineCreation.vkResult = VK_ERROR_INITIALIZATION_FAILED;
        const nrc::vulkan::VulkanProfileValidationReport failedPipeline =
            nrc::vulkan::evaluate_vulkan_profile_validation(
                validCapabilities,
                {},
                *khrProfile,
                requirePipeline);
        ok &= require(!failedPipeline.usable,
                      "failed pipeline creation blocks usable report");
        ok &= require(failedPipeline.status ==
                          nrc::vulkan::VulkanProfileValidationStatus::PipelineCreationFailed,
                      "Vulkan validation reports failed pipeline creation");
        ok &= require(failedPipeline.pipelineCreation.vkResult == VK_ERROR_INITIALIZATION_FAILED,
                      "Vulkan validation preserves pipeline VkResult");

        requirePipeline.pipelineCreation.status = nrc::vulkan::VulkanValidationStepStatus::Skipped;
        requirePipeline.pipelineCreation.structuredException = true;
        requirePipeline.pipelineCreation.vkResult = VK_ERROR_UNKNOWN;
        const nrc::vulkan::VulkanProfileValidationReport skippedExceptionPipeline =
            nrc::vulkan::evaluate_vulkan_profile_validation(
                validCapabilities,
                {},
                *khrProfile,
                requirePipeline);
        ok &= require(skippedExceptionPipeline.pipelineCreation.structuredException,
                      "Vulkan validation preserves structured exception flag");

        nrc::vulkan::VulkanProfileValidationRequirements requireNumerical{};
        requireNumerical.requireNumericalValidation = true;
        requireNumerical.pipelineCreation.status = nrc::vulkan::VulkanValidationStepStatus::Passed;
        const nrc::vulkan::VulkanProfileValidationReport missingNumerical =
            nrc::vulkan::evaluate_vulkan_profile_validation(
                validCapabilities,
                {},
                *khrProfile,
                requireNumerical);
        ok &= require(missingNumerical.status ==
                          nrc::vulkan::VulkanProfileValidationStatus::NumericalValidationNotRun,
                      "Vulkan validation reports missing numerical validation");

        requireNumerical.numericalValidation.status = nrc::vulkan::VulkanValidationStepStatus::Failed;
        const nrc::vulkan::VulkanProfileValidationReport failedNumerical =
            nrc::vulkan::evaluate_vulkan_profile_validation(
                validCapabilities,
                {},
                *khrProfile,
                requireNumerical);
        ok &= require(failedNumerical.status ==
                          nrc::vulkan::VulkanProfileValidationStatus::NumericalValidationFailed,
                      "Vulkan validation reports failed numerical validation");
    }

    nrc::vulkan::VulkanDeviceCapabilities emptyCapabilities{};
    if (khrProfile != nullptr) {
        const nrc::vulkan::VulkanProfileValidationReport rejectedCapability =
            nrc::vulkan::evaluate_vulkan_profile_validation(emptyCapabilities, {}, *khrProfile);
        ok &= require(!rejectedCapability.usable,
                      "Vulkan validation rejects unsupported capability profile");
        ok &= require(rejectedCapability.status ==
                          nrc::vulkan::VulkanProfileValidationStatus::ProfileNotSupported,
                      "Vulkan validation reports capability rejection");
        ok &= require(rejectedCapability.profileSupport.status ==
                          nrc::ProfileSupportStatus::MissingRequiredFeature,
                      "Vulkan validation carries detailed capability rejection");
    }

    const nrc::Profile missingConfigProfile{
        nrc::ProfileId::None,
        nrc::BackendKind::ScalarReference,
        "SUPPORTED_PROFILE_WITHOUT_CONFIG",
        0,
        0,
        0,
        0,
        64,
        false,
        true,
    };
    const nrc::vulkan::VulkanProfileValidationReport missingConfig =
        nrc::vulkan::evaluate_vulkan_profile_validation(
            emptyCapabilities,
            {.allowReferenceBackend = true},
            missingConfigProfile);
    ok &= require(!missingConfig.usable, "Vulkan validation rejects missing shader config");
    ok &= require(missingConfig.status ==
                      nrc::vulkan::VulkanProfileValidationStatus::MissingShaderConfig,
                  "Vulkan validation reports missing shader config");

    const nrc::Profile mismatchedConfigProfile{
        nrc::ProfileId::ScalarReferenceWg64,
        nrc::BackendKind::KhrCooperativeMatrix,
        "KHR_PROFILE_WITH_SCALAR_CONFIG_ID",
        16,
        16,
        16,
        32,
        128,
        true,
        true,
    };
    const nrc::vulkan::VulkanProfileValidationReport mismatchedConfig =
        nrc::vulkan::evaluate_vulkan_profile_validation(
            validCapabilities,
            {},
            mismatchedConfigProfile);
    ok &= require(!mismatchedConfig.usable, "Vulkan validation rejects shader config mismatch");
    ok &= require(mismatchedConfig.status ==
                      nrc::vulkan::VulkanProfileValidationStatus::ShaderConfigMismatch,
                  "Vulkan validation reports shader config mismatch");

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
