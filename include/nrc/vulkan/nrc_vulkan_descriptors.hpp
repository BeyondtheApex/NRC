#pragma once

#include "nrc/vulkan/nrc_vulkan_capabilities.hpp"

#include <cstdint>
#include <span>
#include <vulkan/vulkan.h>

namespace nrc::vulkan {

struct VulkanDescriptorBinding {
    std::uint32_t binding = 0;
    VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    std::uint32_t descriptorCount = 1;
    VkShaderStageFlags stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    const VkSampler *immutableSamplers = nullptr;
};

struct VulkanDescriptorSetLayoutCreateInfo {
    VkDevice device = VK_NULL_HANDLE;
    std::span<const VulkanDescriptorBinding> bindings;
    VkDescriptorSetLayoutCreateFlags flags = 0;
    const VkAllocationCallbacks *allocator = nullptr;
};

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

struct VulkanStorageImageDescriptorWrite {
    std::uint32_t binding = 0;
    VkImageView imageView = VK_NULL_HANDLE;
    VkImageLayout imageLayout = VK_IMAGE_LAYOUT_GENERAL;
};

struct VulkanStorageImageDescriptorUpdate {
    VkDevice device = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    std::span<const VulkanStorageImageDescriptorWrite> images;
};

VulkanDescriptorSetLayoutCreateResult create_descriptor_set_layout(
    const VulkanDescriptorSetLayoutCreateInfo &createInfo);

VulkanDescriptorSetLayoutCreateResult create_storage_buffer_descriptor_set_layout(
    const VulkanStorageBufferDescriptorSetLayoutCreateInfo &createInfo);

VulkanValidationStep update_storage_buffer_descriptors(
    const VulkanStorageBufferDescriptorUpdate &update);

VulkanValidationStep update_storage_image_descriptors(
    const VulkanStorageImageDescriptorUpdate &update);

} // namespace nrc::vulkan
