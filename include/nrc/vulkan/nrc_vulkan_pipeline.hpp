#pragma once

#include "nrc/nrc_profile.hpp"
#include "nrc/vulkan/nrc_vulkan_capabilities.hpp"

#include <cstdint>
#include <vulkan/vulkan.h>

namespace nrc::vulkan {

struct VulkanComputePipelineCreateInfo {
    VkDevice device = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    const ProfileShaderConfig *shaderConfig = nullptr;
    const char *entryPoint = "main";
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    VkPipelineCreateFlags pipelineFlags = 0;
    VkPipelineShaderStageCreateFlags shaderStageFlags = 0;
    const VkSpecializationInfo *specializationInfo = nullptr;
    const VkAllocationCallbacks *allocator = nullptr;
    VkPipeline basePipelineHandle = VK_NULL_HANDLE;
    std::int32_t basePipelineIndex = -1;
};

struct VulkanComputePipelineCreateResult {
    VulkanValidationStep validation;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

struct VulkanBasicComputePipelineCreateInfo {
    VkDevice device = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    const char *entryPoint = "main";
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    VkPipelineCreateFlags pipelineFlags = 0;
    VkPipelineShaderStageCreateFlags shaderStageFlags = 0;
    const VkSpecializationInfo *specializationInfo = nullptr;
    const VkAllocationCallbacks *allocator = nullptr;
    VkPipeline basePipelineHandle = VK_NULL_HANDLE;
    std::int32_t basePipelineIndex = -1;
};

VulkanComputePipelineCreateResult create_compute_pipeline(
    const VulkanBasicComputePipelineCreateInfo &createInfo);

VulkanComputePipelineCreateResult create_profile_compute_pipeline(
    const VulkanComputePipelineCreateInfo &createInfo);

} // namespace nrc::vulkan
