#include "nrc/nrc_encoding.hpp"
#include "nrc/nrc_records.hpp"
#include "nrc/nrc_reference_nn.hpp"
#include "nrc/nrc_weights.hpp"
#include "nrc/vulkan/nrc_vulkan_barriers.hpp"
#include "nrc/vulkan/nrc_vulkan_descriptors.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kGradientWorkgroupSize = 128;
constexpr std::uint32_t kTrainCount = 3;

struct SkipTest : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct PushConstants {
    std::uint32_t batch = 0;
};

std::string vk_result_string(VkResult result) {
    return std::to_string(static_cast<int>(result));
}

void check_vk(VkResult result, const char *operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with VkResult " + vk_result_string(result));
    }
}

std::vector<std::uint32_t> read_spirv(const char *path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error(std::string("could not open SPIR-V file: ") + path);
    }

    const std::streamsize size = file.tellg();
    if (size <= 0 || (size % 4) != 0) {
        throw std::runtime_error("SPIR-V file has an invalid size");
    }

    std::vector<std::uint32_t> words(static_cast<std::size_t>(size) / sizeof(std::uint32_t));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char *>(words.data()), size);
    if (!file) {
        throw std::runtime_error("failed to read SPIR-V file");
    }

    return words;
}

std::uint32_t pack_unorm2x16(float x, float y) {
    const auto pack = [](float value) {
        const float clamped = std::clamp(value, 0.0f, 1.0f);
        return static_cast<std::uint32_t>(std::round(clamped * 65535.0f)) & 0xffffu;
    };
    return pack(x) | (pack(y) << 16u);
}

nrc::Vec2 unpack_unorm2x16(std::uint32_t packed) {
    constexpr float kInvUnorm16 = 1.0f / 65535.0f;
    return {
        static_cast<float>(packed & 0xffffu) * kInvUnorm16,
        static_cast<float>((packed >> 16u) & 0xffffu) * kInvUnorm16,
    };
}

nrc::UnpackedNrcInput unpack_test_input(const nrc::PackedNrcInput &packedInput) {
    const nrc::Vec2 barycentric = unpack_unorm2x16(packedInput.barycentric2x16U);
    const nrc::Vec2 scattered = unpack_unorm2x16(packedInput.scatteredDir2x16U);

    nrc::UnpackedNrcInput input{};
    input.position = {barycentric.x, barycentric.y, 0.25f};
    input.scatteredDir = scattered;
    input.normal = {0.5f, 0.5f};
    input.roughness = 0.35f;
    input.diffuse = {0.7f, 0.6f, 0.5f};
    input.specular = {0.1f, 0.2f, 0.3f};
    return input;
}

std::vector<float> make_test_weights() {
    std::vector<float> weights(nrc::weight_count(), 0.0f);
    for (std::uint32_t layer = 0; layer < nrc::kNnHiddenLayerCount; ++layer) {
        const std::size_t offset = nrc::layer_weight_offset(layer);
        for (std::uint32_t i = 0; i < nrc::kNnHiddenWidth; ++i) {
            weights[offset + i * nrc::kNnHiddenWidth + i] = 0.8f;
        }
    }

    const std::size_t outputOffset = nrc::layer_weight_offset(nrc::kNnHiddenLayerCount);
    weights[outputOffset + 0 * nrc::kNnHiddenWidth + 0] = 0.7f;
    weights[outputOffset + 1 * nrc::kNnHiddenWidth + 1] = 0.6f;
    weights[outputOffset + 2 * nrc::kNnHiddenWidth + 2] = 0.5f;
    return weights;
}

std::vector<nrc::NrcTrainRecord> make_train_records() {
    std::vector<nrc::NrcTrainRecord> records(nrc::kTrainBatchCount * nrc::kTrainBatchSize);
    for (std::uint32_t i = 0; i < kTrainCount; ++i) {
        nrc::NrcTrainRecord record{};
        record.biasR = 0.12f + 0.03f * static_cast<float>(i);
        record.biasG = 0.08f + 0.02f * static_cast<float>(i);
        record.biasB = 0.04f + 0.01f * static_cast<float>(i);
        record.factorR = 1.0f;
        record.factorG = 1.0f;
        record.factorB = 1.0f;
        record.packedInput.primitiveId = i;
        record.packedInput.flipBitInstanceId = i;
        record.packedInput.barycentric2x16U = pack_unorm2x16(0.25f + 0.10f * static_cast<float>(i),
                                                             0.20f + 0.07f * static_cast<float>(i));
        record.packedInput.scatteredDir2x16U = pack_unorm2x16(0.35f + 0.05f * static_cast<float>(i),
                                                              0.65f - 0.04f * static_cast<float>(i));
        records[i] = record;
    }
    return records;
}

std::vector<float> make_expected_gradients(std::span<const float> gpuVisibleWeights,
                                           const std::vector<nrc::NrcTrainRecord> &trainRecords) {
    std::vector<float> gradients(nrc::weight_count(), 0.0f);
    for (std::uint32_t i = 0; i < kTrainCount; ++i) {
        nrc::EncodedInput encoded{};
        nrc::encode_input(unpack_test_input(trainRecords[i].packedInput), encoded);
        const nrc::ReferenceOutput target{
            trainRecords[i].biasR,
            trainRecords[i].biasG,
            trainRecords[i].biasB,
        };
        nrc::reference_backward(gpuVisibleWeights,
                                encoded,
                                target,
                                gradients,
                                nullptr,
                                nrc::ReferenceLoss::RelativeL2Luminance,
                                nrc::kLossScale,
                                true);
    }
    return gradients;
}

bool has_extension(VkPhysicalDevice device, const char *name) {
    std::uint32_t count = 0;
    check_vk(vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr),
             "vkEnumerateDeviceExtensionProperties");
    std::vector<VkExtensionProperties> extensions(count);
    check_vk(vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data()),
             "vkEnumerateDeviceExtensionProperties");
    for (const VkExtensionProperties &extension : extensions) {
        if (std::strcmp(extension.extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

class VulkanGradientRunner {
public:
    explicit VulkanGradientRunner(const char *spirvPath) {
        create_instance();
        pick_physical_device();
        create_device();
        create_descriptor_set_layout();
        create_pipeline_layout();
        create_pipeline(spirvPath);
        create_command_pool();
    }

    VulkanGradientRunner(const VulkanGradientRunner &) = delete;
    VulkanGradientRunner &operator=(const VulkanGradientRunner &) = delete;

    ~VulkanGradientRunner() {
        if (device_ != VK_NULL_HANDLE) {
            if (commandPool_ != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device_, commandPool_, nullptr);
            }
            if (pipeline_ != VK_NULL_HANDLE) {
                vkDestroyPipeline(device_, pipeline_, nullptr);
            }
            if (pipelineLayout_ != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
            }
            if (descriptorSetLayout_ != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
            }
            vkDestroyDevice(device_, nullptr);
        }
        if (instance_ != VK_NULL_HANDLE) {
            vkDestroyInstance(instance_, nullptr);
        }
    }

    std::vector<float> run(std::span<const nrc::NrcTrainRecord> trainRecords,
                           const nrc::NrcQueueState &queueState,
                           std::span<const nrc::PackedFp16x8> packedWeights) {
        Buffer trainRecordBuffer = create_host_buffer(static_cast<VkDeviceSize>(trainRecords.size_bytes()));
        Buffer queueStateBuffer = create_host_buffer(sizeof(nrc::NrcQueueState));
        Buffer weightBuffer = create_host_buffer(static_cast<VkDeviceSize>(packedWeights.size_bytes()));
        Buffer gradientBuffer = create_host_buffer(static_cast<VkDeviceSize>(nrc::weight_count() * sizeof(float)));

        upload(trainRecordBuffer, trainRecords.data(), static_cast<VkDeviceSize>(trainRecords.size_bytes()));
        upload(queueStateBuffer, &queueState, sizeof(queueState));
        upload(weightBuffer, packedWeights.data(), static_cast<VkDeviceSize>(packedWeights.size_bytes()));
        std::vector<float> zeroGradients(nrc::weight_count(), 0.0f);
        upload(gradientBuffer, zeroGradients.data(), static_cast<VkDeviceSize>(zeroGradients.size() * sizeof(float)));

        VkDescriptorPool descriptorPool = create_descriptor_pool();
        VkDescriptorSet descriptorSet = allocate_descriptor_set(descriptorPool);
        update_descriptor_set(descriptorSet, trainRecordBuffer, queueStateBuffer, weightBuffer, gradientBuffer);

        VkCommandBuffer commandBuffer = allocate_command_buffer();
        VkFence fence = VK_NULL_HANDLE;
        try {
            VkFenceCreateInfo fenceInfo{};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            check_vk(vkCreateFence(device_, &fenceInfo, nullptr, &fence), "vkCreateFence");
            record_dispatch(commandBuffer, descriptorSet);

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;
            check_vk(vkQueueSubmit(queue_, 1, &submitInfo, fence), "vkQueueSubmit");
            check_vk(vkWaitForFences(device_, 1, &fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max()),
                     "vkWaitForFences");
        } catch (...) {
            if (fence != VK_NULL_HANDLE) {
                vkDestroyFence(device_, fence, nullptr);
            }
            vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer);
            vkDestroyDescriptorPool(device_, descriptorPool, nullptr);
            throw;
        }

        vkDestroyFence(device_, fence, nullptr);
        vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer);
        vkDestroyDescriptorPool(device_, descriptorPool, nullptr);

        std::vector<float> gradients(nrc::weight_count());
        download(gradientBuffer, gradients.data(), static_cast<VkDeviceSize>(gradients.size() * sizeof(float)));
        return gradients;
    }

private:
    struct Buffer {
        VkDevice device = VK_NULL_HANDLE;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;

        Buffer() = default;
        Buffer(const Buffer &) = delete;
        Buffer &operator=(const Buffer &) = delete;

        Buffer(Buffer &&other) noexcept {
            *this = std::move(other);
        }

        Buffer &operator=(Buffer &&other) noexcept {
            if (this != &other) {
                destroy();
                device = other.device;
                buffer = other.buffer;
                memory = other.memory;
                size = other.size;
                other.device = VK_NULL_HANDLE;
                other.buffer = VK_NULL_HANDLE;
                other.memory = VK_NULL_HANDLE;
                other.size = 0;
            }
            return *this;
        }

        ~Buffer() {
            destroy();
        }

        void destroy() {
            if (device != VK_NULL_HANDLE) {
                if (buffer != VK_NULL_HANDLE) {
                    vkDestroyBuffer(device, buffer, nullptr);
                }
                if (memory != VK_NULL_HANDLE) {
                    vkFreeMemory(device, memory, nullptr);
                }
            }
            buffer = VK_NULL_HANDLE;
            memory = VK_NULL_HANDLE;
            size = 0;
        }
    };

    void create_instance() {
        std::uint32_t loaderVersion = VK_API_VERSION_1_0;
        const auto enumerateInstanceVersion =
            reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
                vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
        if (enumerateInstanceVersion != nullptr) {
            check_vk(enumerateInstanceVersion(&loaderVersion), "vkEnumerateInstanceVersion");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "nrc_vulkan_gradient_scalar_test";
        appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        appInfo.pEngineName = "NRC";
        appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
        appInfo.apiVersion = std::min(loaderVersion, VK_API_VERSION_1_1);

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        const VkResult result = vkCreateInstance(&createInfo, nullptr, &instance_);
        if (result == VK_ERROR_INCOMPATIBLE_DRIVER) {
            throw SkipTest("no compatible Vulkan loader/driver was found");
        }
        check_vk(result, "vkCreateInstance");
    }

    void pick_physical_device() {
        std::uint32_t count = 0;
        check_vk(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "vkEnumeratePhysicalDevices");
        if (count == 0) {
            throw SkipTest("no Vulkan physical devices were found");
        }

        std::vector<VkPhysicalDevice> devices(count);
        check_vk(vkEnumeratePhysicalDevices(instance_, &count, devices.data()), "vkEnumeratePhysicalDevices");

        for (VkPhysicalDevice device : devices) {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(device, &properties);
            if (properties.apiVersion < VK_API_VERSION_1_1 ||
                !has_extension(device, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)) {
                continue;
            }

            VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloat{};
            atomicFloat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
            VkPhysicalDeviceFeatures2 features2{};
            features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features2.pNext = &atomicFloat;
            vkGetPhysicalDeviceFeatures2(device, &features2);
            if (atomicFloat.shaderBufferFloat32AtomicAdd != VK_TRUE) {
                continue;
            }

            std::uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
            std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

            for (std::uint32_t i = 0; i < queueFamilyCount; ++i) {
                if ((queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                    physicalDevice_ = device;
                    queueFamilyIndex_ = i;
                    vkGetPhysicalDeviceMemoryProperties(device, &memoryProperties_);
                    return;
                }
            }
        }

        throw SkipTest("no Vulkan compute device supports shaderBufferFloat32AtomicAdd");
    }

    void create_device() {
        float priority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = queueFamilyIndex_;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &priority;

        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloat{};
        atomicFloat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        atomicFloat.shaderBufferFloat32AtomicAdd = VK_TRUE;

        const char *extensions[] = {VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME};

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pNext = &atomicFloat;
        createInfo.queueCreateInfoCount = 1;
        createInfo.pQueueCreateInfos = &queueInfo;
        createInfo.enabledExtensionCount = 1;
        createInfo.ppEnabledExtensionNames = extensions;

        check_vk(vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_), "vkCreateDevice");
        vkGetDeviceQueue(device_, queueFamilyIndex_, 0, &queue_);
    }

    void create_descriptor_set_layout() {
        const std::array<nrc::vulkan::VulkanStorageBufferBinding, 4> bindings = {{
            {0, VK_SHADER_STAGE_COMPUTE_BIT},
            {1, VK_SHADER_STAGE_COMPUTE_BIT},
            {2, VK_SHADER_STAGE_COMPUTE_BIT},
            {3, VK_SHADER_STAGE_COMPUTE_BIT},
        }};

        nrc::vulkan::VulkanStorageBufferDescriptorSetLayoutCreateInfo createInfo{};
        createInfo.device = device_;
        createInfo.bindings = bindings;
        const nrc::vulkan::VulkanDescriptorSetLayoutCreateResult result =
            nrc::vulkan::create_storage_buffer_descriptor_set_layout(createInfo);
        check_vk(result.validation.vkResult, "vkCreateDescriptorSetLayout");
        descriptorSetLayout_ = result.layout;
    }

    void create_pipeline_layout() {
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        createInfo.setLayoutCount = 1;
        createInfo.pSetLayouts = &descriptorSetLayout_;
        createInfo.pushConstantRangeCount = 1;
        createInfo.pPushConstantRanges = &pushRange;
        check_vk(vkCreatePipelineLayout(device_, &createInfo, nullptr, &pipelineLayout_),
                 "vkCreatePipelineLayout");
    }

    void create_pipeline(const char *spirvPath) {
        const std::vector<std::uint32_t> spirv = read_spirv(spirvPath);

        VkShaderModuleCreateInfo shaderInfo{};
        shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderInfo.codeSize = spirv.size() * sizeof(std::uint32_t);
        shaderInfo.pCode = spirv.data();

        VkShaderModule shaderModule = VK_NULL_HANDLE;
        check_vk(vkCreateShaderModule(device_, &shaderInfo, nullptr, &shaderModule), "vkCreateShaderModule");

        VkPipelineShaderStageCreateInfo stageInfo{};
        stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = shaderModule;
        stageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = pipelineLayout_;

        const VkResult result =
            vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_);
        vkDestroyShaderModule(device_, shaderModule, nullptr);
        check_vk(result, "vkCreateComputePipelines");
    }

    void create_command_pool() {
        VkCommandPoolCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        createInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        createInfo.queueFamilyIndex = queueFamilyIndex_;
        check_vk(vkCreateCommandPool(device_, &createInfo, nullptr, &commandPool_), "vkCreateCommandPool");
    }

    std::uint32_t find_host_visible_memory_type(std::uint32_t typeBits) const {
        const VkMemoryPropertyFlags required =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (std::uint32_t i = 0; i < memoryProperties_.memoryTypeCount; ++i) {
            if ((typeBits & (1u << i)) != 0 &&
                (memoryProperties_.memoryTypes[i].propertyFlags & required) == required) {
                return i;
            }
        }
        throw SkipTest("no host-visible coherent memory type is available for storage buffers");
    }

    Buffer create_host_buffer(VkDeviceSize size) {
        Buffer out{};
        out.device = device_;
        out.size = size;

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check_vk(vkCreateBuffer(device_, &bufferInfo, nullptr, &out.buffer), "vkCreateBuffer");

        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, out.buffer, &requirements);

        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = requirements.size;
        allocateInfo.memoryTypeIndex = find_host_visible_memory_type(requirements.memoryTypeBits);
        check_vk(vkAllocateMemory(device_, &allocateInfo, nullptr, &out.memory), "vkAllocateMemory");
        check_vk(vkBindBufferMemory(device_, out.buffer, out.memory, 0), "vkBindBufferMemory");
        return out;
    }

    void upload(const Buffer &buffer, const void *data, VkDeviceSize size) const {
        void *mapped = nullptr;
        check_vk(vkMapMemory(device_, buffer.memory, 0, size, 0, &mapped), "vkMapMemory");
        std::memcpy(mapped, data, static_cast<std::size_t>(size));
        vkUnmapMemory(device_, buffer.memory);
    }

    void download(const Buffer &buffer, void *data, VkDeviceSize size) const {
        void *mapped = nullptr;
        check_vk(vkMapMemory(device_, buffer.memory, 0, size, 0, &mapped), "vkMapMemory");
        std::memcpy(data, mapped, static_cast<std::size_t>(size));
        vkUnmapMemory(device_, buffer.memory);
    }

    VkDescriptorPool create_descriptor_pool() const {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 4;

        VkDescriptorPoolCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        createInfo.maxSets = 1;
        createInfo.poolSizeCount = 1;
        createInfo.pPoolSizes = &poolSize;

        VkDescriptorPool pool = VK_NULL_HANDLE;
        check_vk(vkCreateDescriptorPool(device_, &createInfo, nullptr, &pool), "vkCreateDescriptorPool");
        return pool;
    }

    VkDescriptorSet allocate_descriptor_set(VkDescriptorPool descriptorPool) const {
        VkDescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool = descriptorPool;
        allocateInfo.descriptorSetCount = 1;
        allocateInfo.pSetLayouts = &descriptorSetLayout_;

        VkDescriptorSet set = VK_NULL_HANDLE;
        check_vk(vkAllocateDescriptorSets(device_, &allocateInfo, &set), "vkAllocateDescriptorSets");
        return set;
    }

    void update_descriptor_set(VkDescriptorSet set,
                               const Buffer &trainRecords,
                               const Buffer &queueState,
                               const Buffer &weights,
                               const Buffer &gradients) const {
        const std::array<nrc::vulkan::VulkanStorageBufferDescriptorWrite, 4> writes = {{
            {0, trainRecords.buffer, 0, trainRecords.size},
            {1, queueState.buffer, 0, queueState.size},
            {2, weights.buffer, 0, weights.size},
            {3, gradients.buffer, 0, gradients.size},
        }};
        nrc::vulkan::VulkanStorageBufferDescriptorUpdate update{};
        update.device = device_;
        update.descriptorSet = set;
        update.buffers = writes;
        const nrc::vulkan::VulkanValidationStep validation =
            nrc::vulkan::update_storage_buffer_descriptors(update);
        check_vk(validation.vkResult, "vkUpdateDescriptorSets");
    }

    VkCommandBuffer allocate_command_buffer() const {
        VkCommandBufferAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocateInfo.commandPool = commandPool_;
        allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocateInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        check_vk(vkAllocateCommandBuffers(device_, &allocateInfo, &commandBuffer), "vkAllocateCommandBuffers");
        return commandBuffer;
    }

    void record_dispatch(VkCommandBuffer commandBuffer, VkDescriptorSet set) const {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        check_vk(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");

        nrc::vulkan::VulkanValidationStep barrierValidation =
            nrc::vulkan::cmd_memory_barrier(
                commandBuffer,
                nrc::vulkan::host_write_to_compute_shader_barrier(
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));
        check_vk(barrierValidation.vkResult, "host-to-shader barrier");

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &set, 0, nullptr);
        const PushConstants constants{0u};
        vkCmdPushConstants(commandBuffer,
                           pipelineLayout_,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           sizeof(constants),
                           &constants);
        vkCmdDispatch(commandBuffer, (kTrainCount + kGradientWorkgroupSize - 1u) / kGradientWorkgroupSize, 1, 1);

        barrierValidation =
            nrc::vulkan::cmd_memory_barrier(
                commandBuffer,
                nrc::vulkan::compute_shader_write_to_host_read_barrier());
        check_vk(barrierValidation.vkResult, "shader-to-host barrier");

        check_vk(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
    }

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memoryProperties_{};
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::uint32_t queueFamilyIndex_ = 0;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
};

bool close_enough(float lhs, float rhs) {
    const float tolerance = 5.0e-3f + 2.0e-3f * std::max(std::fabs(lhs), std::fabs(rhs));
    return std::fabs(lhs - rhs) <= tolerance;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: nrc_vulkan_gradient_scalar_test <nrc_gradient_scalar_dispatch.spv>\n";
        return EXIT_FAILURE;
    }

    try {
        const std::vector<float> weights = make_test_weights();
        const std::vector<nrc::PackedFp16x8> packedWeights = nrc::pack_weights_fp16_uvec4(weights);
        const std::vector<float> gpuVisibleWeights =
            nrc::unpack_weights_fp16_uvec4(packedWeights, nrc::weight_count());
        const std::vector<nrc::NrcTrainRecord> trainRecords = make_train_records();
        const std::vector<float> expectedGradients = make_expected_gradients(gpuVisibleWeights, trainRecords);

        nrc::NrcQueueState queueState{};
        queueState.trainCounts[0] = kTrainCount;

        VulkanGradientRunner runner(argv[1]);
        const std::vector<float> actualGradients = runner.run(trainRecords, queueState, packedWeights);

        bool ok = true;
        std::uint32_t mismatchCount = 0;
        for (std::uint32_t i = 0; i < nrc::weight_count(); ++i) {
            if (!close_enough(actualGradients[i], expectedGradients[i])) {
                if (mismatchCount < 16u) {
                    std::cerr << "FAIL: gradient " << i
                              << " expected " << expectedGradients[i]
                              << " got " << actualGradients[i] << '\n';
                }
                ++mismatchCount;
                ok = false;
            }
        }
        if (mismatchCount != 0u) {
            std::cerr << "FAIL: total gradient mismatches " << mismatchCount << '\n';
        }

        float maxMagnitude = 0.0f;
        for (float value : actualGradients) {
            maxMagnitude = std::max(maxMagnitude, std::fabs(value));
        }
        if (maxMagnitude == 0.0f) {
            std::cerr << "FAIL: scalar gradient dispatch produced all-zero gradients\n";
            ok = false;
        }

        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const SkipTest &skip) {
        std::cout << "SKIP: " << skip.what() << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
