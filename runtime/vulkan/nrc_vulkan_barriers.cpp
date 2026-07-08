#include "nrc/vulkan/nrc_vulkan_barriers.hpp"

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

} // namespace nrc::vulkan
