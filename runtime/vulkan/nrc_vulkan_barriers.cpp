#include "nrc/vulkan/nrc_vulkan_barriers.hpp"

#include <cstddef>
#include <vector>

namespace nrc::vulkan {

namespace {

VulkanValidationStep failed_step() {
    VulkanValidationStep validation{};
    validation.status = VulkanValidationStepStatus::Failed;
    validation.vkResult = VK_ERROR_INITIALIZATION_FAILED;
    return validation;
}

bool valid_barrier(const VulkanMemoryBarrier &barrier) {
    return barrier.srcStageMask != 0 && barrier.dstStageMask != 0;
}

bool valid_image_barrier(const VulkanImageBarrier &barrier) {
    return barrier.image != VK_NULL_HANDLE &&
           barrier.newLayout != VK_IMAGE_LAYOUT_UNDEFINED &&
           barrier.subresourceRange.aspectMask != 0 &&
           barrier.subresourceRange.levelCount != 0 &&
           barrier.subresourceRange.layerCount != 0;
}

bool valid_pipeline_barrier(const VulkanPipelineBarrierInfo &barrier) {
    if (barrier.commandBuffer == VK_NULL_HANDLE ||
        barrier.srcStageMask == 0 ||
        barrier.dstStageMask == 0 ||
        (barrier.memoryBarriers.empty() && barrier.imageBarriers.empty())) {
        return false;
    }

    for (const VulkanImageBarrier &imageBarrier : barrier.imageBarriers) {
        if (!valid_image_barrier(imageBarrier)) {
            return false;
        }
    }
    return true;
}

} // namespace

VulkanMemoryBarrier host_write_to_compute_shader_barrier(VkAccessFlags shaderAccessMask) {
    return {
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_HOST_WRITE_BIT,
        shaderAccessMask,
        0,
    };
}

VulkanMemoryBarrier compute_shader_write_to_host_read_barrier() {
    return {
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_HOST_READ_BIT,
        0,
    };
}

VulkanMemoryBarrier compute_shader_write_to_compute_shader_barrier(VkAccessFlags shaderAccessMask) {
    return {
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        shaderAccessMask,
        0,
    };
}

VulkanImageBarrier color_image_barrier(VkImage image,
                                       VkImageLayout oldLayout,
                                       VkImageLayout newLayout,
                                       VkAccessFlags srcAccessMask,
                                       VkAccessFlags dstAccessMask) {
    VulkanImageBarrier barrier{};
    barrier.srcAccessMask = srcAccessMask;
    barrier.dstAccessMask = dstAccessMask;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    return barrier;
}

VulkanValidationStep cmd_memory_barrier(VkCommandBuffer commandBuffer,
                                        const VulkanMemoryBarrier &barrier) {
    if (commandBuffer == VK_NULL_HANDLE || !valid_barrier(barrier)) {
        return failed_step();
    }

    VkMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = barrier.srcAccessMask;
    memoryBarrier.dstAccessMask = barrier.dstAccessMask;

    vkCmdPipelineBarrier(commandBuffer,
                         barrier.srcStageMask,
                         barrier.dstStageMask,
                         barrier.dependencyFlags,
                         1,
                         &memoryBarrier,
                         0,
                         nullptr,
                         0,
                         nullptr);

    VulkanValidationStep validation{};
    validation.status = VulkanValidationStepStatus::Passed;
    validation.vkResult = VK_SUCCESS;
    return validation;
}

VulkanValidationStep cmd_pipeline_barrier(const VulkanPipelineBarrierInfo &barrier) {
    if (!valid_pipeline_barrier(barrier)) {
        return failed_step();
    }

    std::vector<VkMemoryBarrier> memoryBarriers(barrier.memoryBarriers.size());
    for (std::size_t index = 0; index < barrier.memoryBarriers.size(); ++index) {
        memoryBarriers[index].sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarriers[index].srcAccessMask = barrier.memoryBarriers[index].srcAccessMask;
        memoryBarriers[index].dstAccessMask = barrier.memoryBarriers[index].dstAccessMask;
    }

    std::vector<VkImageMemoryBarrier> imageBarriers(barrier.imageBarriers.size());
    for (std::size_t index = 0; index < barrier.imageBarriers.size(); ++index) {
        imageBarriers[index].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imageBarriers[index].srcAccessMask = barrier.imageBarriers[index].srcAccessMask;
        imageBarriers[index].dstAccessMask = barrier.imageBarriers[index].dstAccessMask;
        imageBarriers[index].oldLayout = barrier.imageBarriers[index].oldLayout;
        imageBarriers[index].newLayout = barrier.imageBarriers[index].newLayout;
        imageBarriers[index].srcQueueFamilyIndex = barrier.imageBarriers[index].srcQueueFamilyIndex;
        imageBarriers[index].dstQueueFamilyIndex = barrier.imageBarriers[index].dstQueueFamilyIndex;
        imageBarriers[index].image = barrier.imageBarriers[index].image;
        imageBarriers[index].subresourceRange = barrier.imageBarriers[index].subresourceRange;
    }

    vkCmdPipelineBarrier(barrier.commandBuffer,
                         barrier.srcStageMask,
                         barrier.dstStageMask,
                         barrier.dependencyFlags,
                         static_cast<std::uint32_t>(memoryBarriers.size()),
                         memoryBarriers.empty() ? nullptr : memoryBarriers.data(),
                         0,
                         nullptr,
                         static_cast<std::uint32_t>(imageBarriers.size()),
                         imageBarriers.empty() ? nullptr : imageBarriers.data());

    VulkanValidationStep validation{};
    validation.status = VulkanValidationStepStatus::Passed;
    validation.vkResult = VK_SUCCESS;
    return validation;
}

} // namespace nrc::vulkan
