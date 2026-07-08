#include "nrc/nrc_encoding.hpp"
#include "nrc/nrc_khr_reference.hpp"
#include "nrc/nrc_profile.hpp"
#include "nrc/nrc_records.hpp"
#include "nrc/nrc_reference_nn.hpp"
#include "nrc/nrc_weights.hpp"
#include "nrc/vulkan/nrc_vulkan_barriers.hpp"
#include "nrc/vulkan/nrc_vulkan_capabilities.hpp"
#include "nrc/vulkan/nrc_vulkan_descriptors.hpp"
#include "nrc/vulkan/nrc_vulkan_pipeline.hpp"
#include "nrc/vulkan/nrc_vulkan_shader.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
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

std::vector<nrc::NrcTrainRecord> make_train_records() {
    std::vector<nrc::NrcTrainRecord> records(nrc::kTrainBatchCount * nrc::kTrainBatchSize);
    for (std::uint32_t i = 0; i < kTrainCount; ++i) {
        nrc::NrcTrainRecord record{};
        record.biasR = 0.10f + 0.03f * static_cast<float>(i);
        record.biasG = 0.07f + 0.02f * static_cast<float>(i);
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

std::vector<float> make_expected_khr_groundwork_gradients(std::span<const float> gpuVisibleWeights,
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
        nrc::reference_backward_khr_fp16(gpuVisibleWeights,
                                         encoded,
                                         target,
                                         gradients,
                                         nullptr,
                                         nrc::kLossScale,
                                         true);
    }
    return gradients;
}

float fp16_round(float value) {
    return nrc::half_bits_to_float(nrc::float_to_half_bits(value));
}

using HiddenVector = std::array<float, nrc::kNnHiddenWidth>;
using HiddenLayers = std::array<HiddenVector, nrc::kNnHiddenLayerCount>;

HiddenLayers forward_hidden_layers_fp16(std::span<const float> weights,
                                        const nrc::ReferenceInput &input) {
    HiddenLayers hiddenLayers{};
    HiddenVector src{};
    std::copy(input.begin(), input.end(), src.begin());

    for (std::uint32_t layer = 0; layer < nrc::kNnHiddenLayerCount; ++layer) {
        const std::size_t offset = nrc::layer_weight_offset(layer);
        for (std::uint32_t row = 0; row < nrc::kNnHiddenWidth; ++row) {
            float value = 0.0f;
            for (std::uint32_t col = 0; col < nrc::kNnHiddenWidth; ++col) {
                value += weights[offset + row * nrc::kNnHiddenWidth + col] * src[col];
            }
            hiddenLayers[layer][row] = fp16_round(std::max(value, 0.0f));
        }
        src = hiddenLayers[layer];
    }
    return hiddenLayers;
}

nrc::ReferenceOutput output_from_final_hidden_fp16(std::span<const float> weights,
                                                   const HiddenVector &finalHidden) {
    nrc::ReferenceOutput output{};
    const std::size_t outputOffset = nrc::layer_weight_offset(nrc::kNnHiddenLayerCount);
    for (std::uint32_t row = 0; row < nrc::kNnOutputWidth; ++row) {
        float value = 0.0f;
        for (std::uint32_t col = 0; col < nrc::kNnHiddenWidth; ++col) {
            value += weights[outputOffset + row * nrc::kNnHiddenWidth + col] * finalHidden[col];
        }
        output[row] = fp16_round(value);
    }
    return output;
}

nrc::ReferenceOutput relative_luminance_output_derivative(const nrc::ReferenceOutput &prediction,
                                                          const nrc::ReferenceOutput &target) {
    const float luminance =
        0.299f * std::max(prediction[0], 0.0f) +
        0.587f * std::max(prediction[1], 0.0f) +
        0.114f * std::max(prediction[2], 0.0f);
    const float denominator = luminance * luminance + 0.01f;

    nrc::ReferenceOutput dOutput{};
    for (std::uint32_t row = 0; row < nrc::kNnOutputWidth; ++row) {
        dOutput[row] = 2.0f * nrc::kLossScale * (prediction[row] - target[row]) / denominator;
    }
    return dOutput;
}

std::vector<float> make_expected_khr_output_coopmat_gradients(
    std::span<const float> gpuVisibleWeights,
    const std::vector<nrc::NrcTrainRecord> &trainRecords) {
    std::vector<float> gradients =
        make_expected_khr_groundwork_gradients(gpuVisibleWeights, trainRecords);

    const std::size_t outputOffset = nrc::layer_weight_offset(nrc::kNnHiddenLayerCount);
    std::fill(gradients.begin() + static_cast<std::ptrdiff_t>(outputOffset),
              gradients.begin() + static_cast<std::ptrdiff_t>(outputOffset + nrc::output_layer_weight_count()),
              0.0f);

    std::array<float, nrc::kNnOutputWidth * nrc::kNnHiddenWidth> outputPartials{};
    for (std::uint32_t i = 0; i < kTrainCount; ++i) {
        nrc::EncodedInput encoded{};
        nrc::encode_input(unpack_test_input(trainRecords[i].packedInput), encoded);
        const HiddenLayers hiddenLayers = forward_hidden_layers_fp16(gpuVisibleWeights, encoded);
        const HiddenVector &finalHidden = hiddenLayers[nrc::kNnHiddenLayerCount - 1u];
        const nrc::ReferenceOutput prediction =
            output_from_final_hidden_fp16(gpuVisibleWeights, finalHidden);
        const nrc::ReferenceOutput target{
            trainRecords[i].biasR,
            trainRecords[i].biasG,
            trainRecords[i].biasB,
        };
        const nrc::ReferenceOutput dOutput =
            relative_luminance_output_derivative(prediction, target);

        for (std::uint32_t row = 0; row < nrc::kNnOutputWidth; ++row) {
            const float stagedDOutput = fp16_round(dOutput[row]);
            for (std::uint32_t col = 0; col < nrc::kNnHiddenWidth; ++col) {
                outputPartials[row * nrc::kNnHiddenWidth + col] +=
                    finalHidden[col] * stagedDOutput;
            }
        }
    }

    for (std::uint32_t i = 0; i < nrc::output_layer_weight_count(); ++i) {
        gradients[outputOffset + i] = fp16_round(outputPartials[i]);
    }
    return gradients;
}

class VulkanKhrGradientRunner {
public:
    VulkanKhrGradientRunner(const char *sg32SpirvPath,
                            const char *sg16SpirvPath,
                            const char *m8Sg32SpirvPath) {
        create_instance();
        pick_physical_device();
        create_device();
        create_descriptor_set_layout();
        create_pipeline_layout();
        create_pipeline(selected_spirv_path(sg32SpirvPath, sg16SpirvPath, m8Sg32SpirvPath));
        create_command_pool();
    }

    VulkanKhrGradientRunner(const VulkanKhrGradientRunner &) = delete;
    VulkanKhrGradientRunner &operator=(const VulkanKhrGradientRunner &) = delete;

    ~VulkanKhrGradientRunner() {
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
        if (loaderVersion < VK_API_VERSION_1_3) {
            throw SkipTest("Vulkan loader does not support Vulkan 1.3");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "nrc_vulkan_gradient_khr_test";
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
        requirements.requireTraining = true;
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
            selectedCapabilities_ = std::move(capabilities);
            queueFamilyIndex_ = computeQueue;
            vkGetPhysicalDeviceMemoryProperties(device, &memoryProperties_);
            return;
        }

        throw SkipTest("no Vulkan device supports an implemented NRC KHR cooperative matrix training profile");
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

        VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperativeMatrixFeatures{};
        cooperativeMatrixFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        cooperativeMatrixFeatures.pNext = &atomicFloat;
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
            VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
        };

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pNext = &features;
        createInfo.queueCreateInfoCount = 1;
        createInfo.pQueueCreateInfos = &queueInfo;
        createInfo.enabledExtensionCount = static_cast<std::uint32_t>(std::size(extensions));
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
        throw std::runtime_error("selected KHR profile has no gradient SPIR-V");
    }

    void create_pipeline(const char *spirvPath) {
        const std::vector<std::uint32_t> spirv = read_spirv(spirvPath);
        const nrc::ProfileShaderConfig *config = nrc::find_profile_shader_config(selectedProfile_->id);
        if (config == nullptr) {
            throw std::runtime_error("selected KHR profile has no shader config");
        }

        nrc::ProfileRequirements profileRequirements{};
        profileRequirements.requireTraining = true;
        profileRequirements.allowReferenceBackend = false;

        nrc::vulkan::VulkanShaderModuleCreateInfo shaderCreateInfo{};
        shaderCreateInfo.device = device_;
        shaderCreateInfo.code = std::span<const std::uint32_t>(spirv.data(), spirv.size());
        const nrc::vulkan::VulkanShaderModuleCreateResult shaderResult =
            nrc::vulkan::create_shader_module(shaderCreateInfo);

        nrc::vulkan::VulkanProfileValidationRequirements validationRequirements{};
        validationRequirements.requireShaderModuleCreation = true;
        validationRequirements.shaderModuleCreation = shaderResult.validation;
        nrc::vulkan::VulkanProfileValidationReport validationReport =
            nrc::vulkan::evaluate_vulkan_profile_validation(
                selectedCapabilities_,
                profileRequirements,
                *selectedProfile_,
                validationRequirements);
        if (!validationReport.usable) {
            check_vk(validationReport.shaderModuleCreation.vkResult, "vkCreateShaderModule");
            throw std::runtime_error("selected KHR training profile failed Vulkan shader module validation");
        }

        VkShaderModule shaderModule = shaderResult.shaderModule;
        nrc::vulkan::VulkanComputePipelineCreateInfo pipelineCreateInfo{};
        pipelineCreateInfo.device = device_;
        pipelineCreateInfo.pipelineLayout = pipelineLayout_;
        pipelineCreateInfo.shaderModule = shaderModule;
        pipelineCreateInfo.shaderConfig = config;

        const nrc::vulkan::VulkanComputePipelineCreateResult pipelineResult =
            nrc::vulkan::create_profile_compute_pipeline(pipelineCreateInfo);
        vkDestroyShaderModule(device_, shaderModule, nullptr);

        validationRequirements.requirePipelineCreation = true;
        validationRequirements.pipelineCreation = pipelineResult.validation;
        validationReport =
            nrc::vulkan::evaluate_vulkan_profile_validation(
                selectedCapabilities_,
                profileRequirements,
                *selectedProfile_,
                validationRequirements);
        if (validationReport.pipelineCreation.structuredException) {
            throw SkipTest("vkCreateComputePipelines raised a structured exception for the KHR gradient shader");
        }
        if (!validationReport.usable) {
            check_vk(validationReport.pipelineCreation.vkResult, "vkCreateComputePipelines");
            throw std::runtime_error("selected KHR training profile failed Vulkan pipeline validation");
        }
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

        const std::uint32_t groupCount =
            (kTrainCount + selectedProfile_->workgroupSize - 1u) / selectedProfile_->workgroupSize;
        vkCmdDispatch(commandBuffer, groupCount, 1, 1);

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
    const nrc::Profile *selectedProfile_ = nullptr;
    nrc::vulkan::VulkanDeviceCapabilities selectedCapabilities_{};
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR getCooperativeMatrixProperties_ = nullptr;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
};

bool close_enough(float lhs, float rhs) {
    const float tolerance = 1.0e-1f + 1.0e-1f * std::max(std::fabs(lhs), std::fabs(rhs));
    return std::fabs(lhs - rhs) <= tolerance;
}

} // namespace

int main(int argc, char **argv) {
    const bool useCoopmatOutput = argc == 5 && std::strcmp(argv[4], "--coopmat-output") == 0;
    if (argc != 4 && !useCoopmatOutput) {
        std::cerr << "usage: nrc_vulkan_gradient_khr_test <sg32.spv> <sg16.spv> <m8_sg32.spv> [--coopmat-output]\n";
        return EXIT_FAILURE;
    }

    try {
        const std::vector<float> weights = make_test_weights();
        const std::vector<nrc::PackedFp16x8> packedWeights = nrc::pack_weights_fp16_uvec4(weights);
        const std::vector<float> gpuVisibleWeights =
            nrc::unpack_weights_fp16_uvec4(packedWeights, nrc::weight_count());
        const std::vector<nrc::NrcTrainRecord> trainRecords = make_train_records();
        const std::vector<float> expectedGradients =
            useCoopmatOutput
                ? make_expected_khr_output_coopmat_gradients(gpuVisibleWeights, trainRecords)
                : make_expected_khr_groundwork_gradients(gpuVisibleWeights, trainRecords);

        nrc::NrcQueueState queueState{};
        queueState.trainCounts[0] = kTrainCount;

        VulkanKhrGradientRunner runner(argv[1], argv[2], argv[3]);
        const std::vector<float> actualGradients = runner.run(trainRecords, queueState, packedWeights);

        bool ok = true;
        std::uint32_t mismatchCount = 0;
        const std::size_t firstCoveredHiddenOffset = nrc::layer_weight_offset(0u);
        const std::size_t outputOffset = nrc::layer_weight_offset(nrc::kNnHiddenLayerCount);
        for (std::uint32_t i = 0; i < nrc::weight_count(); ++i) {
            const bool coveredHiddenWeight = i >= firstCoveredHiddenOffset && i < outputOffset;
            const bool outputWeight =
                i >= outputOffset && i < outputOffset + nrc::output_layer_weight_count();
            const float expected = coveredHiddenWeight || outputWeight ? expectedGradients[i] : 0.0f;
            if (!close_enough(actualGradients[i], expected)) {
                if (mismatchCount < 16u) {
                    std::cerr << "FAIL: gradient " << i
                              << " expected " << expected
                              << " got " << actualGradients[i] << '\n';
                }
                ++mismatchCount;
                ok = false;
            }
        }
        if (mismatchCount != 0u) {
            std::cerr << "FAIL: total KHR groundwork gradient mismatches " << mismatchCount << '\n';
        }

        float maxGroundworkMagnitude = 0.0f;
        for (std::size_t i = firstCoveredHiddenOffset; i < outputOffset; ++i) {
            maxGroundworkMagnitude =
                std::max(maxGroundworkMagnitude, std::fabs(actualGradients[i]));
        }
        for (std::uint32_t i = 0; i < nrc::output_layer_weight_count(); ++i) {
            maxGroundworkMagnitude =
                std::max(maxGroundworkMagnitude, std::fabs(actualGradients[outputOffset + i]));
        }
        if (maxGroundworkMagnitude == 0.0f) {
            std::cerr << "FAIL: KHR gradient dispatch produced all-zero groundwork gradients\n";
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
