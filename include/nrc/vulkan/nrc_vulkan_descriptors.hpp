#pragma once

#include "nrc/vulkan/nrc_vulkan_capabilities.hpp"

#include <cstdint>
#include <span>
#include <vulkan/vulkan.h>

namespace nrc::vulkan {

struct VulkanStorageBufferBinding {
    std::uint32_t binding = 0;
    VkShaderStageFlags stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
};

struct VulkanStorageBufferDescriptorSetLayoutCreateInfo {
    VkDevice device = VK_NULL_HANDLE;
    std::span<const VulkanStorageBufferBinding> bindings;
    VkDescriptorSetLayoutCreateFlags flags = 0;
    const VkAllocationCallbacks *allocator = nullptr;
};

struct VulkanDescriptorSetLayoutCreateResult {
    VulkanValidationStep validation;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
};

struct VulkanStorageBufferDescriptorWrite {
    std::uint32_t binding = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize range = VK_WHOLE_SIZE;
};

struct VulkanStorageBufferDescriptorUpdate {
    VkDevice device = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    std::span<const VulkanStorageBufferDescriptorWrite> buffers;
};

VulkanDescriptorSetLayoutCreateResult create_storage_buffer_descriptor_set_layout(
    const VulkanStorageBufferDescriptorSetLayoutCreateInfo &createInfo);

VulkanValidationStep update_storage_buffer_descriptors(
    const VulkanStorageBufferDescriptorUpdate &update);

} // namespace nrc::vulkan
