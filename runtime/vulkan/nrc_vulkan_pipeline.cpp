#include "nrc/vulkan/nrc_vulkan_pipeline.hpp"

#if defined(_MSC_VER) && defined(_WIN32)
#include <excpt.h>
#endif

namespace nrc::vulkan {

namespace {

VkResult create_compute_pipeline_guarded(VkDevice device,
                                         VkPipelineCache pipelineCache,
                                         const VkComputePipelineCreateInfo *pipelineInfo,
                                         const VkAllocationCallbacks *allocator,
                                         VkPipeline *pipeline,
                                         bool &structuredException) {
    structuredException = false;
    *pipeline = VK_NULL_HANDLE;
#if defined(_MSC_VER) && defined(_WIN32)
    VkResult result = VK_ERROR_UNKNOWN;
    __try {
        result = vkCreateComputePipelines(device, pipelineCache, 1, pipelineInfo, allocator, pipeline);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        structuredException = true;
    }
    return result;
#else
    return vkCreateComputePipelines(device, pipelineCache, 1, pipelineInfo, allocator, pipeline);
#endif
}

VulkanComputePipelineCreateResult make_invalid_result() {
    VulkanComputePipelineCreateResult result{};
    result.validation.status = VulkanValidationStepStatus::Failed;
    result.validation.vkResult = VK_ERROR_INITIALIZATION_FAILED;
    return result;
}

} // namespace

VulkanComputePipelineCreateResult create_profile_compute_pipeline(
    const VulkanComputePipelineCreateInfo &createInfo) {
    if (createInfo.device == VK_NULL_HANDLE ||
        createInfo.pipelineLayout == VK_NULL_HANDLE ||
        createInfo.shaderModule == VK_NULL_HANDLE ||
        createInfo.shaderConfig == nullptr ||
        createInfo.entryPoint == nullptr ||
        createInfo.entryPoint[0] == '\0') {
        return make_invalid_result();
    }

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo requiredSubgroupSize{};
    requiredSubgroupSize.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    requiredSubgroupSize.requiredSubgroupSize = createInfo.shaderConfig->subgroupSize;

    VkPipelineShaderStageCreateFlags stageFlags = createInfo.shaderStageFlags;
    const bool useRequiredSubgroupSize =
        createInfo.shaderConfig->requireSubgroupSizeControl &&
        createInfo.shaderConfig->subgroupSize != 0;
    if (createInfo.shaderConfig->requireFullSubgroups) {
        stageFlags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    }

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.pNext = useRequiredSubgroupSize ? &requiredSubgroupSize : nullptr;
    stageInfo.flags = stageFlags;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = createInfo.shaderModule;
    stageInfo.pName = createInfo.entryPoint;
    stageInfo.pSpecializationInfo = createInfo.specializationInfo;

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.flags = createInfo.pipelineFlags;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = createInfo.pipelineLayout;
    pipelineInfo.basePipelineHandle = createInfo.basePipelineHandle;
    pipelineInfo.basePipelineIndex = createInfo.basePipelineIndex;

    VulkanComputePipelineCreateResult result{};
    VkPipeline pipeline = VK_NULL_HANDLE;
    bool structuredException = false;
    const VkResult vkResult = create_compute_pipeline_guarded(
        createInfo.device,
        createInfo.pipelineCache,
        &pipelineInfo,
        createInfo.allocator,
        &pipeline,
        structuredException);

    result.validation.vkResult = structuredException ? VK_ERROR_UNKNOWN : vkResult;
    result.validation.structuredException = structuredException;
    if (structuredException) {
        result.validation.status = VulkanValidationStepStatus::Skipped;
        return result;
    }
    if (vkResult != VK_SUCCESS) {
        result.validation.status = VulkanValidationStepStatus::Failed;
        return result;
    }
    if (pipeline == VK_NULL_HANDLE) {
        result.validation.status = VulkanValidationStepStatus::Failed;
        result.validation.vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return result;
    }

    result.validation.status = VulkanValidationStepStatus::Passed;
    result.pipeline = pipeline;
    return result;
}

} // namespace nrc::vulkan
