#pragma once

#include "nrc/vulkan/nrc_vulkan_capabilities.hpp"

#include <vulkan/vulkan.h>

namespace nrc::vulkan {

struct VulkanMemoryBarrier {
    VkPipelineStageFlags srcStageMask = 0;
    VkPipelineStageFlags dstStageMask = 0;
    VkAccessFlags srcAccessMask = 0;
    VkAccessFlags dstAccessMask = 0;
    VkDependencyFlags dependencyFlags = 0;
};

VulkanMemoryBarrier host_write_to_compute_shader_barrier(VkAccessFlags shaderAccessMask);
VulkanMemoryBarrier compute_shader_write_to_host_read_barrier();
VulkanMemoryBarrier compute_shader_write_to_compute_shader_barrier(VkAccessFlags shaderAccessMask);

VulkanValidationStep cmd_memory_barrier(VkCommandBuffer commandBuffer,
                                        const VulkanMemoryBarrier &barrier);

} // namespace nrc::vulkan
