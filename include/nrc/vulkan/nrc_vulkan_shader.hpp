#pragma once

#include "nrc/vulkan/nrc_vulkan_capabilities.hpp"

#include <cstdint>
#include <span>
#include <vulkan/vulkan.h>

namespace nrc::vulkan {

struct VulkanShaderModuleCreateInfo {
    VkDevice device = VK_NULL_HANDLE;
    std::span<const std::uint32_t> code;
    VkShaderModuleCreateFlags flags = 0;
    const VkAllocationCallbacks *allocator = nullptr;
};

struct VulkanShaderModuleCreateResult {
    VulkanValidationStep validation;
    VkShaderModule shaderModule = VK_NULL_HANDLE;
};

VulkanShaderModuleCreateResult create_shader_module(
    const VulkanShaderModuleCreateInfo &createInfo);

} // namespace nrc::vulkan
