#pragma once

#include "nrc/vulkan/nrc_vulkan_capabilities.hpp"

#include <cstdint>
#include <span>
#include <vulkan/vulkan.h>

namespace nrc::vulkan {

struct VulkanMemoryBarrier {
    VkPipelineStageFlags srcStageMask = 0;
    VkPipelineStageFlags dstStageMask = 0;
    VkAccessFlags srcAccessMask = 0;
    VkAccessFlags dstAccessMask = 0;
    VkDependencyFlags dependencyFlags = 0;
};

struct VulkanMemoryAccessBarrier {
    VkAccessFlags srcAccessMask = 0;
    VkAccessFlags dstAccessMask = 0;
};

struct VulkanImageBarrier {
    VkAccessFlags srcAccessMask = 0;
    VkAccessFlags dstAccessMask = 0;
    VkImageLayout oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout newLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    std::uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    std::uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    VkImage image = VK_NULL_HANDLE;
    VkImageSubresourceRange subresourceRange{};
};

struct VulkanPipelineBarrierInfo {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkPipelineStageFlags srcStageMask = 0;
    VkPipelineStageFlags dstStageMask = 0;
    VkDependencyFlags dependencyFlags = 0;
    std::span<const VulkanMemoryAccessBarrier> memoryBarriers;
    std::span<const VulkanImageBarrier> imageBarriers;
};

VulkanMemoryBarrier host_write_to_compute_shader_barrier(VkAccessFlags shaderAccessMask);
VulkanMemoryBarrier compute_shader_write_to_host_read_barrier();
VulkanMemoryBarrier compute_shader_write_to_compute_shader_barrier(VkAccessFlags shaderAccessMask);

VulkanImageBarrier color_image_barrier(VkImage image,
                                       VkImageLayout oldLayout,
                                       VkImageLayout newLayout,
                                       VkAccessFlags srcAccessMask,
                                       VkAccessFlags dstAccessMask);

VulkanValidationStep cmd_memory_barrier(VkCommandBuffer commandBuffer,
                                        const VulkanMemoryBarrier &barrier);

VulkanValidationStep cmd_pipeline_barrier(const VulkanPipelineBarrierInfo &barrier);

} // namespace nrc::vulkan
