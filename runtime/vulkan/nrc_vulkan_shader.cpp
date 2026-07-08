#include "nrc/vulkan/nrc_vulkan_shader.hpp"

namespace nrc::vulkan {

namespace {

VulkanShaderModuleCreateResult make_invalid_result() {
    VulkanShaderModuleCreateResult result{};
    result.validation.status = VulkanValidationStepStatus::Failed;
    result.validation.vkResult = VK_ERROR_INITIALIZATION_FAILED;
    return result;
}

} // namespace

VulkanShaderModuleCreateResult create_shader_module(
    const VulkanShaderModuleCreateInfo &createInfo) {
    if (createInfo.device == VK_NULL_HANDLE || createInfo.code.empty()) {
        return make_invalid_result();
    }

    VkShaderModuleCreateInfo shaderInfo{};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.flags = createInfo.flags;
    shaderInfo.codeSize = createInfo.code.size_bytes();
    shaderInfo.pCode = createInfo.code.data();

    VulkanShaderModuleCreateResult result{};
    const VkResult vkResult =
        vkCreateShaderModule(createInfo.device,
                             &shaderInfo,
                             createInfo.allocator,
                             &result.shaderModule);
    result.validation.vkResult = vkResult;
    if (vkResult != VK_SUCCESS) {
        result.validation.status = VulkanValidationStepStatus::Failed;
        result.shaderModule = VK_NULL_HANDLE;
        return result;
    }
    if (result.shaderModule == VK_NULL_HANDLE) {
        result.validation.status = VulkanValidationStepStatus::Failed;
        result.validation.vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return result;
    }

    result.validation.status = VulkanValidationStepStatus::Passed;
    return result;
}

} // namespace nrc::vulkan
