#include "nrc/nrc_profile.hpp"
#include "nrc/nrc_reference_nn.hpp"
#include "nrc/nrc_weights.hpp"
#include "nrc/vulkan/nrc_vulkan_capabilities.hpp"
#include "nrc/vulkan/nrc_vulkan_pipeline.hpp"

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

constexpr std::uint32_t kSampleCount = 149;

struct SkipTest : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Vec4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

std::string vk_result_string(VkResult result) {
    return std::to_string(static_cast<int>(result));
}

void check_vk(VkResult result, const char *operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with VkResult " + vk_result_string(result));
    }
}

std::uint32_t pack_half2(float x, float y) {
    return static_cast<std::uint32_t>(nrc::float_to_half_bits(x)) |
           (static_cast<std::uint32_t>(nrc::float_to_half_bits(y)) << 16u);
}

float fp16_round(float value) {
    return nrc::half_bits_to_float(nrc::float_to_half_bits(value));
}

std::vector<float> make_test_weights() {
    std::vector<float> weights(nrc::weight_count(), 0.0f);

    for (std::uint32_t layer = 0; layer < nrc::kNnHiddenLayerCount; ++layer) {
        const std::size_t offset = nrc::layer_weight_offset(layer);
        for (std::uint32_t row = 0; row < nrc::kNnHiddenWidth; ++row) {
            for (std::uint32_t col = 0; col < nrc::kNnHiddenWidth; ++col) {
                const int pattern = static_cast<int>(((layer + 2u) * 11u + row * 5u + col * 7u) % 17u) - 8;
                const float diagonal = row == col ? 0.625f : 0.0f;
                weights[offset + row * nrc::kNnHiddenWidth + col] =
                    diagonal + static_cast<float>(pattern) * 0.001953125f;
            }
        }
    }

    const std::size_t outputOffset = nrc::layer_weight_offset(nrc::kNnHiddenLayerCount);
    for (std::uint32_t row = 0; row < nrc::kNnOutputWidth; ++row) {
        for (std::uint32_t col = 0; col < nrc::kNnHiddenWidth; ++col) {
            const int pattern = static_cast<int>(((row + 3u) * 19u + col * 13u) % 19u) - 9;
            weights[outputOffset + row * nrc::kNnHiddenWidth + col] =
                static_cast<float>(pattern) * 0.0068359375f;
        }
    }

    return weights;
}

void make_test_inputs(std::vector<nrc::ReferenceInput> &cpuInputs,
                      std::vector<nrc::PackedFp16x8> &packedInputs) {
    cpuInputs.resize(kSampleCount);
    packedInputs.resize(kSampleCount * nrc::kEncodedInputUvec4Count);

    for (std::uint32_t sample = 0; sample < kSampleCount; ++sample) {
        nrc::ReferenceInput input{};
        for (std::uint32_t i = 0; i < nrc::kEncodedInputWidth; ++i) {
            const int pattern = static_cast<int>(((sample + 5u) * 23u + (i + 11u) * 31u) % 53u) - 26;
            const float rawValue = static_cast<float>(pattern) / 37.0f;
            input[i] = fp16_round(rawValue);
        }
        cpuInputs[sample] = input;

        for (std::uint32_t block = 0; block < nrc::kEncodedInputUvec4Count; ++block) {
            nrc::PackedFp16x8 packed{};
            const std::uint32_t base = block * nrc::kFp16PerUvec4;
            for (std::uint32_t component = 0; component < 4; ++component) {
                packed[component] = pack_half2(input[base + component * 2u],
                                               input[base + component * 2u + 1u]);
            }
            packedInputs[sample * nrc::kEncodedInputUvec4Count + block] = packed;
        }
    }
}

void reference_forward_fp16_layers(std::span<const float> weights,
                                   const nrc::ReferenceInput &input,
                                   nrc::ReferenceOutput &output) {
    std::array<float, nrc::kNnHiddenWidth> src{};
    std::array<float, nrc::kNnHiddenWidth> dst{};
    std::copy(input.begin(), input.end(), src.begin());

    for (std::uint32_t layer = 0; layer < nrc::kNnHiddenLayerCount; ++layer) {
        const std::size_t offset = nrc::layer_weight_offset(layer);
        for (std::uint32_t row = 0; row < nrc::kNnHiddenWidth; ++row) {
            float value = 0.0f;
            for (std::uint32_t col = 0; col < nrc::kNnHiddenWidth; ++col) {
                value += weights[offset + row * nrc::kNnHiddenWidth + col] * src[col];
            }
            dst[row] = fp16_round(std::max(value, 0.0f));
        }
        src = dst;
    }

    const std::size_t outputOffset = nrc::layer_weight_offset(nrc::kNnHiddenLayerCount);
    for (std::uint32_t row = 0; row < nrc::kNnOutputWidth; ++row) {
        float value = 0.0f;
        for (std::uint32_t col = 0; col < nrc::kNnHiddenWidth; ++col) {
            value += weights[outputOffset + row * nrc::kNnHiddenWidth + col] * src[col];
        }
        output[row] = fp16_round(value);
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

class VulkanKhrRunner {
public:
    VulkanKhrRunner(const char *sg32SpirvPath, const char *sg16SpirvPath, const char *m8Sg32SpirvPath) {
        create_instance();
        pick_physical_device();
        create_device();
        create_descriptor_set_layout();
        create_pipeline_layout();
        create_pipeline(selected_spirv_path(sg32SpirvPath, sg16SpirvPath, m8Sg32SpirvPath));
        create_command_pool();
    }

    VulkanKhrRunner(const VulkanKhrRunner &) = delete;
    VulkanKhrRunner &operator=(const VulkanKhrRunner &) = delete;

    ~VulkanKhrRunner() {
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

    std::vector<Vec4> run(std::span<const nrc::PackedFp16x8> packedWeights,
                          std::span<const nrc::PackedFp16x8> packedInputs,
                          std::uint32_t sampleCount) {
        const VkDeviceSize weightBytes =
            static_cast<VkDeviceSize>(packedWeights.size_bytes());
        const VkDeviceSize inputBytes =
            static_cast<VkDeviceSize>(packedInputs.size_bytes());
        const VkDeviceSize outputBytes =
            static_cast<VkDeviceSize>(sizeof(Vec4) * sampleCount);

        Buffer weights = create_host_buffer(weightBytes);
        Buffer inputs = create_host_buffer(inputBytes);
        Buffer outputs = create_host_buffer(outputBytes);

        upload(weights, packedWeights.data(), weightBytes);
        upload(inputs, packedInputs.data(), inputBytes);
        std::vector<Vec4> zeroed(sampleCount);
        upload(outputs, zeroed.data(), outputBytes);

        VkDescriptorPool descriptorPool = create_descriptor_pool();
        VkDescriptorSet descriptorSet = allocate_descriptor_set(descriptorPool);
        update_descriptor_set(descriptorSet, weights, inputs, outputs);

        VkCommandBuffer commandBuffer = allocate_command_buffer();
        VkFence fence = VK_NULL_HANDLE;

        try {
            VkFenceCreateInfo fenceInfo{};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            check_vk(vkCreateFence(device_, &fenceInfo, nullptr, &fence), "vkCreateFence");

            record_dispatch(commandBuffer, descriptorSet, sampleCount);

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

        if (fence != VK_NULL_HANDLE) {
            vkDestroyFence(device_, fence, nullptr);
        }
        vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer);
        vkDestroyDescriptorPool(device_, descriptorPool, nullptr);

        std::vector<Vec4> result(sampleCount);
        download(outputs, result.data(), outputBytes);
        return result;
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
        if (loaderVersion < VK_API_VERSION_1_3) {
            throw SkipTest("Vulkan loader does not support Vulkan 1.3");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "nrc_vulkan_khr_dispatch_test";
        appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        appInfo.pEngineName = "NRC";
        appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
        appInfo.apiVersion = VK_API_VERSION_1_3;

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
        getCooperativeMatrixProperties_ =
            reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
                vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
        if (getCooperativeMatrixProperties_ == nullptr) {
            throw SkipTest("vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR is not available");
        }

        std::uint32_t count = 0;
        check_vk(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "vkEnumeratePhysicalDevices");
        if (count == 0) {
            throw SkipTest("no Vulkan physical devices were found");
        }

        std::vector<VkPhysicalDevice> devices(count);
        check_vk(vkEnumeratePhysicalDevices(instance_, &count, devices.data()), "vkEnumeratePhysicalDevices");

        nrc::ProfileRequirements requirements{};
        requirements.requireTraining = false;
        requirements.allowReferenceBackend = false;

        for (VkPhysicalDevice device : devices) {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(device, &properties);
            if (properties.apiVersion < VK_API_VERSION_1_3) {
                continue;
            }

            std::uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
            std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

            std::uint32_t computeQueue = std::numeric_limits<std::uint32_t>::max();
            for (std::uint32_t i = 0; i < queueFamilyCount; ++i) {
                if ((queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                    computeQueue = i;
                    break;
                }
            }
            if (computeQueue == std::numeric_limits<std::uint32_t>::max()) {
                continue;
            }

            nrc::vulkan::VulkanDeviceCapabilities capabilities =
                nrc::vulkan::query_device_capabilities(device, getCooperativeMatrixProperties_);
            const nrc::ProfileId profileId = nrc::choose_profile(capabilities.view(), requirements);
            const nrc::Profile *profile = nrc::find_profile(profileId);
            if (profile == nullptr || profile->backend != nrc::BackendKind::KhrCooperativeMatrix) {
                continue;
            }

            physicalDevice_ = device;
            selectedProfile_ = profile;
            queueFamilyIndex_ = computeQueue;
            vkGetPhysicalDeviceMemoryProperties(device, &memoryProperties_);
            return;
        }

        throw SkipTest("no Vulkan device supports an implemented NRC KHR cooperative matrix inference profile");
    }

    void create_device() {
        float priority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = queueFamilyIndex_;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &priority;

        VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperativeMatrixFeatures{};
        cooperativeMatrixFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        cooperativeMatrixFeatures.cooperativeMatrix = VK_TRUE;

        VkPhysicalDeviceVulkan13Features vulkan13Features{};
        vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        vulkan13Features.pNext = &cooperativeMatrixFeatures;
        vulkan13Features.subgroupSizeControl = VK_TRUE;
        vulkan13Features.computeFullSubgroups = VK_TRUE;

        VkPhysicalDeviceVulkan12Features vulkan12Features{};
        vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vulkan12Features.pNext = &vulkan13Features;
        vulkan12Features.shaderFloat16 = VK_TRUE;

        VkPhysicalDeviceVulkan11Features vulkan11Features{};
        vulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        vulkan11Features.pNext = &vulkan12Features;
        vulkan11Features.storageBuffer16BitAccess = VK_TRUE;

        VkPhysicalDeviceFeatures2 features{};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features.pNext = &vulkan11Features;

        const char *extensions[] = {
            VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
        };

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pNext = &features;
        createInfo.queueCreateInfoCount = 1;
        createInfo.pQueueCreateInfos = &queueInfo;
        createInfo.enabledExtensionCount = 1;
        createInfo.ppEnabledExtensionNames = extensions;

        check_vk(vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_), "vkCreateDevice");
        vkGetDeviceQueue(device_, queueFamilyIndex_, 0, &queue_);
    }

    void create_descriptor_set_layout() {
        std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
        for (std::uint32_t i = 0; i < bindings.size(); ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        createInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
        createInfo.pBindings = bindings.data();

        check_vk(vkCreateDescriptorSetLayout(device_, &createInfo, nullptr, &descriptorSetLayout_),
                 "vkCreateDescriptorSetLayout");
    }

    void create_pipeline_layout() {
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(std::uint32_t);

        VkPipelineLayoutCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        createInfo.setLayoutCount = 1;
        createInfo.pSetLayouts = &descriptorSetLayout_;
        createInfo.pushConstantRangeCount = 1;
        createInfo.pPushConstantRanges = &pushConstantRange;

        check_vk(vkCreatePipelineLayout(device_, &createInfo, nullptr, &pipelineLayout_),
                 "vkCreatePipelineLayout");
    }

    const char *selected_spirv_path(const char *sg32SpirvPath,
                                    const char *sg16SpirvPath,
                                    const char *m8Sg32SpirvPath) const {
        const nrc::ProfileShaderConfig *config = nrc::find_profile_shader_config(selectedProfile_->id);
        if (config == nullptr || config->backend != nrc::BackendKind::KhrCooperativeMatrix) {
            throw std::runtime_error("selected KHR profile has no shader config");
        }

        if (config->cooperativeMatrixM == 16u && config->subgroupSize == 32u) {
            return sg32SpirvPath;
        }
        if (config->cooperativeMatrixM == 16u && config->subgroupSize == 16u) {
            return sg16SpirvPath;
        }
        if (config->cooperativeMatrixM == 8u && config->subgroupSize == 32u) {
            return m8Sg32SpirvPath;
        }
        throw std::runtime_error("selected KHR profile has no dispatch SPIR-V");
    }

    void create_pipeline(const char *spirvPath) {
        const std::vector<std::uint32_t> spirv = read_spirv(spirvPath);

        VkShaderModuleCreateInfo shaderInfo{};
        shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderInfo.codeSize = spirv.size() * sizeof(std::uint32_t);
        shaderInfo.pCode = spirv.data();

        VkShaderModule shaderModule = VK_NULL_HANDLE;
        check_vk(vkCreateShaderModule(device_, &shaderInfo, nullptr, &shaderModule), "vkCreateShaderModule");

        const nrc::ProfileShaderConfig *config = nrc::find_profile_shader_config(selectedProfile_->id);
        if (config == nullptr) {
            vkDestroyShaderModule(device_, shaderModule, nullptr);
            throw std::runtime_error("selected KHR profile has no shader config");
        }

        nrc::vulkan::VulkanComputePipelineCreateInfo pipelineCreateInfo{};
        pipelineCreateInfo.device = device_;
        pipelineCreateInfo.pipelineLayout = pipelineLayout_;
        pipelineCreateInfo.shaderModule = shaderModule;
        pipelineCreateInfo.shaderConfig = config;

        const nrc::vulkan::VulkanComputePipelineCreateResult pipelineResult =
            nrc::vulkan::create_profile_compute_pipeline(pipelineCreateInfo);
        vkDestroyShaderModule(device_, shaderModule, nullptr);
        if (pipelineResult.validation.structuredException) {
            throw SkipTest("vkCreateComputePipelines raised a structured exception for the KHR cooperative matrix shader");
        }
        check_vk(pipelineResult.validation.vkResult, "vkCreateComputePipelines");
        pipeline_ = pipelineResult.pipeline;
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
        poolSize.descriptorCount = 3;

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

        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        check_vk(vkAllocateDescriptorSets(device_, &allocateInfo, &descriptorSet), "vkAllocateDescriptorSets");
        return descriptorSet;
    }

    void update_descriptor_set(VkDescriptorSet descriptorSet,
                               const Buffer &weights,
                               const Buffer &inputs,
                               const Buffer &outputs) const {
        std::array<VkDescriptorBufferInfo, 3> bufferInfos{};
        bufferInfos[0].buffer = weights.buffer;
        bufferInfos[0].range = weights.size;
        bufferInfos[1].buffer = inputs.buffer;
        bufferInfos[1].range = inputs.size;
        bufferInfos[2].buffer = outputs.buffer;
        bufferInfos[2].range = outputs.size;

        std::array<VkWriteDescriptorSet, 3> writes{};
        for (std::uint32_t i = 0; i < writes.size(); ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptorSet;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &bufferInfos[i];
        }

        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
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

    void record_dispatch(VkCommandBuffer commandBuffer,
                         VkDescriptorSet descriptorSet,
                         std::uint32_t sampleCount) const {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        check_vk(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");

        VkMemoryBarrier hostToShader{};
        hostToShader.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        hostToShader.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        hostToShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_HOST_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                             1,
                             &hostToShader,
                             0,
                             nullptr,
                             0,
                             nullptr);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(commandBuffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipelineLayout_,
                                0,
                                1,
                                &descriptorSet,
                                0,
                                nullptr);
        vkCmdPushConstants(commandBuffer,
                           pipelineLayout_,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           sizeof(sampleCount),
                           &sampleCount);

        const std::uint32_t groupCount =
            (sampleCount + selectedProfile_->workgroupSize - 1u) / selectedProfile_->workgroupSize;
        vkCmdDispatch(commandBuffer, groupCount, 1, 1);

        VkMemoryBarrier shaderToHost{};
        shaderToHost.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        shaderToHost.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        shaderToHost.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT,
                             0,
                             1,
                             &shaderToHost,
                             0,
                             nullptr,
                             0,
                             nullptr);

        check_vk(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
    }

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memoryProperties_{};
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::uint32_t queueFamilyIndex_ = 0;
    const nrc::Profile *selectedProfile_ = nullptr;
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR getCooperativeMatrixProperties_ = nullptr;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
};

bool close_enough(float lhs, float rhs) {
    const float tolerance = 8.0e-2f + 8.0e-2f * std::max(std::fabs(lhs), std::fabs(rhs));
    return std::fabs(lhs - rhs) <= tolerance;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "usage: nrc_vulkan_khr_dispatch_test <sg32.spv> <sg16.spv> <m8_sg32.spv>\n";
        return EXIT_FAILURE;
    }

    try {
        const std::vector<float> weights = make_test_weights();
        const std::vector<nrc::PackedFp16x8> packedWeights = nrc::pack_weights_fp16_uvec4(weights);
        const std::vector<float> gpuVisibleWeights =
            nrc::unpack_weights_fp16_uvec4(packedWeights, nrc::weight_count());

        std::vector<nrc::ReferenceInput> cpuInputs;
        std::vector<nrc::PackedFp16x8> packedInputs;
        make_test_inputs(cpuInputs, packedInputs);

        VulkanKhrRunner runner(argv[1], argv[2], argv[3]);
        const std::vector<Vec4> gpuOutputs = runner.run(packedWeights, packedInputs, kSampleCount);

        bool ok = true;
        for (std::uint32_t sample = 0; sample < kSampleCount; ++sample) {
            nrc::ReferenceOutput expected{};
            reference_forward_fp16_layers(gpuVisibleWeights, cpuInputs[sample], expected);

            const std::array<float, 3> actual = {
                gpuOutputs[sample].x,
                gpuOutputs[sample].y,
                gpuOutputs[sample].z,
            };
            for (std::uint32_t channel = 0; channel < nrc::kNnOutputWidth; ++channel) {
                if (!close_enough(actual[channel], expected[channel])) {
                    std::cerr << "FAIL: sample " << sample
                              << " channel " << channel
                              << " expected " << expected[channel]
                              << " got " << actual[channel] << '\n';
                    ok = false;
                }
            }

            if (!close_enough(gpuOutputs[sample].w, 1.0f)) {
                std::cerr << "FAIL: sample " << sample
                          << " expected output.w 1 got " << gpuOutputs[sample].w << '\n';
                ok = false;
            }
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
