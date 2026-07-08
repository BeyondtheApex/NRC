#include "nrc/vulkan/nrc_vulkan_descriptors.hpp"

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

bool has_duplicate_binding(std::span<const VulkanStorageBufferBinding> bindings) {
    for (std::size_t lhs = 0; lhs < bindings.size(); ++lhs) {
        for (std::size_t rhs = lhs + 1; rhs < bindings.size(); ++rhs) {
            if (bindings[lhs].binding == bindings[rhs].binding) {
                return true;
            }
        }
    }
    return false;
}

bool valid_bindings(std::span<const VulkanStorageBufferBinding> bindings) {
    if (bindings.empty() || has_duplicate_binding(bindings)) {
        return false;
    }
    for (const VulkanStorageBufferBinding &binding : bindings) {
        if (binding.stageFlags == 0) {
            return false;
        }
    }
    return true;
}

bool valid_descriptor_writes(std::span<const VulkanStorageBufferDescriptorWrite> writes) {
    if (writes.empty()) {
        return false;
    }
    for (const VulkanStorageBufferDescriptorWrite &write : writes) {
        if (write.buffer == VK_NULL_HANDLE || write.range == 0) {
            return false;
        }
    }
    return true;
}

} // namespace

VulkanDescriptorSetLayoutCreateResult create_storage_buffer_descriptor_set_layout(
    const VulkanStorageBufferDescriptorSetLayoutCreateInfo &createInfo) {
    VulkanDescriptorSetLayoutCreateResult result{};
    if (createInfo.device == VK_NULL_HANDLE || !valid_bindings(createInfo.bindings)) {
        result.validation = failed_step();
        return result;
    }

    std::vector<VkDescriptorSetLayoutBinding> bindings(createInfo.bindings.size());
    for (std::size_t index = 0; index < createInfo.bindings.size(); ++index) {
        bindings[index].binding = createInfo.bindings[index].binding;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[index].descriptorCount = 1;
        bindings[index].stageFlags = createInfo.bindings[index].stageFlags;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.flags = createInfo.flags;
    layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    const VkResult vkResult = vkCreateDescriptorSetLayout(
        createInfo.device,
        &layoutInfo,
        createInfo.allocator,
        &result.layout);
    result.validation.vkResult = vkResult;
    if (vkResult != VK_SUCCESS) {
        result.validation.status = VulkanValidationStepStatus::Failed;
        result.layout = VK_NULL_HANDLE;
        return result;
    }
    if (result.layout == VK_NULL_HANDLE) {
        result.validation.status = VulkanValidationStepStatus::Failed;
        result.validation.vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return result;
    }

    result.validation.status = VulkanValidationStepStatus::Passed;
    return result;
}

VulkanValidationStep update_storage_buffer_descriptors(
    const VulkanStorageBufferDescriptorUpdate &update) {
    if (update.device == VK_NULL_HANDLE ||
        update.descriptorSet == VK_NULL_HANDLE ||
        !valid_descriptor_writes(update.buffers)) {
        return failed_step();
    }

    std::vector<VkDescriptorBufferInfo> bufferInfos(update.buffers.size());
    std::vector<VkWriteDescriptorSet> writes(update.buffers.size());
    for (std::size_t index = 0; index < update.buffers.size(); ++index) {
        bufferInfos[index].buffer = update.buffers[index].buffer;
        bufferInfos[index].offset = update.buffers[index].offset;
        bufferInfos[index].range = update.buffers[index].range;

        writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index].dstSet = update.descriptorSet;
        writes[index].dstBinding = update.buffers[index].binding;
        writes[index].descriptorCount = 1;
        writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[index].pBufferInfo = &bufferInfos[index];
    }

    vkUpdateDescriptorSets(update.device,
                           static_cast<std::uint32_t>(writes.size()),
                           writes.data(),
                           0,
                           nullptr);

    VulkanValidationStep validation{};
    validation.status = VulkanValidationStepStatus::Passed;
    validation.vkResult = VK_SUCCESS;
    return validation;
}

} // namespace nrc::vulkan
