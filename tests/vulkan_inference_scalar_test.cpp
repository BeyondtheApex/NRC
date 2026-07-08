#include "nrc/nrc_records.hpp"
#include "nrc/nrc_reference_nn.hpp"
#include "nrc/nrc_weights.hpp"

#include <vulkan/vulkan.h>

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

constexpr std::uint32_t kWorkgroupSize = 64;
constexpr std::uint32_t kEvalCount = 4;
constexpr std::uint32_t kScreenWidth = 4;
constexpr std::uint32_t kScreenHeight = 3;
constexpr std::uint32_t kScreenPixelCount = kScreenWidth * kScreenHeight;

struct SkipTest : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
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

std::uint32_t encode_screen(std::uint32_t x, std::uint32_t y) {
    return (x | (y << 15u)) << 1u;
}

std::uint32_t encode_train(std::uint32_t batch, std::uint32_t left, std::uint32_t right) {
    return (batch | (left << 2u) | (right << 16u)) << 1u | 1u;
}

std::uint32_t pack_unorm2(float x, float y) {
    const auto pack = [](float value) {
        const float clamped = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
        return static_cast<std::uint32_t>(std::lround(clamped * 65535.0f));
    };
    return pack(x) | (pack(y) << 16u);
}

nrc::PackedNrcInput make_packed_input(std::uint32_t seed) {
    nrc::PackedNrcInput input{};
    input.primitiveId = seed;
    input.flipBitInstanceId = seed * 17u;
    input.barycentric2x16U = pack_unorm2(0.2f, 0.3f);
    input.scatteredDir2x16U = pack_unorm2(0.4f, 0.6f);
    return input;
}

std::vector<float> make_constant_prediction_weights() {
    std::vector<float> weights(nrc::weight_count(), 0.0f);

    for (std::uint32_t layer = 0; layer < nrc::kNnHiddenLayerCount; ++layer) {
        const std::size_t offset = nrc::layer_weight_offset(layer);
        for (std::uint32_t i = 0; i < nrc::kNnHiddenWidth; ++i) {
            weights[offset + i * nrc::kNnHiddenWidth + i] = 1.0f;
        }
    }

    const std::size_t outputOffset = nrc::layer_weight_offset(nrc::kNnHiddenLayerCount);
    weights[outputOffset + 0 * nrc::kNnHiddenWidth + 62] = 0.25f;
    weights[outputOffset + 1 * nrc::kNnHiddenWidth + 62] = 0.5f;
    weights[outputOffset + 2 * nrc::kNnHiddenWidth + 63] = 0.75f;
    return weights;
}

std::vector<nrc::NrcEvalRecord> make_eval_records() {
    std::vector<nrc::NrcEvalRecord> records(kEvalCount);
    records[0].dst = encode_screen(1, 1);
    records[0].packedInput = make_packed_input(3);
    records[1].dst = encode_train(0, 2, 4);
    records[1].packedInput = make_packed_input(7);
    records[2].dst = 0xffffffffu;
    records[2].packedInput = make_packed_input(11);
    records[3].dst = encode_screen(2, 0);
    records[3].packedInput = make_packed_input(13);
    return records;
}

std::vector<Vec4> make_bias_factor_r() {
    std::vector<Vec4> pixels(kScreenPixelCount);
    for (std::uint32_t y = 0; y < kScreenHeight; ++y) {
        for (std::uint32_t x = 0; x < kScreenWidth; ++x) {
            const std::uint32_t index = y * kScreenWidth + x;
            pixels[index] = {
                10.0f + static_cast<float>(x),
                20.0f + static_cast<float>(y),
                30.0f + static_cast<float>(x + y),
                1.0f + static_cast<float>(x) * 0.25f,
            };
        }
    }
    return pixels;
}

std::vector<Vec2> make_factor_gb() {
    std::vector<Vec2> pixels(kScreenPixelCount);
    for (std::uint32_t y = 0; y < kScreenHeight; ++y) {
        for (std::uint32_t x = 0; x < kScreenWidth; ++x) {
            const std::uint32_t index = y * kScreenWidth + x;
            pixels[index] = {
                2.0f + static_cast<float>(y) * 0.5f,
                3.0f + static_cast<float>(x + y) * 0.25f,
            };
        }
    }
    return pixels;
}

std::vector<nrc::NrcTrainRecord> make_train_records() {
    std::vector<nrc::NrcTrainRecord> records(nrc::kTrainBatchCount * nrc::kTrainBatchSize);
    for (std::uint32_t i = 0; i < 8; ++i) {
        records[i].biasR = 1.0f + static_cast<float>(i);
        records[i].biasG = 2.0f + static_cast<float>(i) * 0.5f;
        records[i].biasB = 3.0f + static_cast<float>(i) * 0.25f;
        records[i].factorR = 0.5f + static_cast<float>(i) * 0.125f;
        records[i].factorG = 0.75f + static_cast<float>(i) * 0.25f;
        records[i].factorB = 1.0f + static_cast<float>(i) * 0.375f;
        records[i].packedInput = make_packed_input(i + 23u);
    }
    return records;
}

bool near(float lhs, float rhs) {
    return std::fabs(lhs - rhs) <= 1.0e-5f;
}

Vec3 prediction() {
    return {0.25f, 0.5f, 0.75f};
}

class VulkanInferenceScalarRunner {
public:
    explicit VulkanInferenceScalarRunner(const char *spirvPath) {
        create_instance();
        pick_physical_device();
        check_formats();
        create_device();
        create_descriptor_set_layout();
        create_pipeline_layout();
        create_pipeline(spirvPath);
        create_command_pool();
    }

    VulkanInferenceScalarRunner(const VulkanInferenceScalarRunner &) = delete;
    VulkanInferenceScalarRunner &operator=(const VulkanInferenceScalarRunner &) = delete;

    ~VulkanInferenceScalarRunner() {
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

    void run(std::span<const nrc::PackedFp16x8> packedWeights,
             std::span<const nrc::NrcEvalRecord> evalRecords,
             std::uint32_t evalCount,
             std::span<const Vec4> biasFactorR,
             std::span<const Vec2> factorGb,
             std::span<const nrc::NrcTrainRecord> trainRecords,
             std::vector<Vec4> &outBiasFactorR,
             std::vector<nrc::NrcTrainRecord> &outTrainRecords) {
        Buffer weights = create_host_buffer(static_cast<VkDeviceSize>(packedWeights.size_bytes()),
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer evals = create_host_buffer(static_cast<VkDeviceSize>(evalRecords.size_bytes()),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer evalCountBuffer = create_host_buffer(sizeof(std::uint32_t),
                                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer trains = create_host_buffer(static_cast<VkDeviceSize>(trainRecords.size_bytes()),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer biasUpload = create_host_buffer(static_cast<VkDeviceSize>(biasFactorR.size_bytes()),
                                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        Buffer factorUpload = create_host_buffer(static_cast<VkDeviceSize>(factorGb.size_bytes()),
                                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        Buffer biasReadback = create_host_buffer(static_cast<VkDeviceSize>(biasFactorR.size_bytes()),
                                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT);

        upload(weights, packedWeights.data(), static_cast<VkDeviceSize>(packedWeights.size_bytes()));
        upload(evals, evalRecords.data(), static_cast<VkDeviceSize>(evalRecords.size_bytes()));
        upload(evalCountBuffer, &evalCount, sizeof(evalCount));
        upload(trains, trainRecords.data(), static_cast<VkDeviceSize>(trainRecords.size_bytes()));
        upload(biasUpload, biasFactorR.data(), static_cast<VkDeviceSize>(biasFactorR.size_bytes()));
        upload(factorUpload, factorGb.data(), static_cast<VkDeviceSize>(factorGb.size_bytes()));

        Image biasImage = create_image(VK_FORMAT_R32G32B32A32_SFLOAT);
        Image factorImage = create_image(VK_FORMAT_R32G32_SFLOAT);

        VkDescriptorPool descriptorPool = create_descriptor_pool();
        VkDescriptorSet descriptorSet = allocate_descriptor_set(descriptorPool);
        update_descriptor_set(descriptorSet, weights, evals, evalCountBuffer, biasImage, factorImage, trains);

        VkCommandBuffer commandBuffer = allocate_command_buffer();
        VkFence fence = VK_NULL_HANDLE;

        try {
            VkFenceCreateInfo fenceInfo{};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            check_vk(vkCreateFence(device_, &fenceInfo, nullptr, &fence), "vkCreateFence");

            record_dispatch(commandBuffer,
                            descriptorSet,
                            biasUpload,
                            factorUpload,
                            biasImage,
                            factorImage,
                            biasReadback,
                            evalCount);

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

        outBiasFactorR.resize(biasFactorR.size());
        outTrainRecords.resize(trainRecords.size());
        download(biasReadback, outBiasFactorR.data(), static_cast<VkDeviceSize>(biasFactorR.size_bytes()));
        download(trains, outTrainRecords.data(), static_cast<VkDeviceSize>(trainRecords.size_bytes()));
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

    struct Image {
        VkDevice device = VK_NULL_HANDLE;
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;

        Image() = default;
        Image(const Image &) = delete;
        Image &operator=(const Image &) = delete;

        Image(Image &&other) noexcept {
            *this = std::move(other);
        }

        Image &operator=(Image &&other) noexcept {
            if (this != &other) {
                destroy();
                device = other.device;
                image = other.image;
                view = other.view;
                memory = other.memory;
                format = other.format;
                other.device = VK_NULL_HANDLE;
                other.image = VK_NULL_HANDLE;
                other.view = VK_NULL_HANDLE;
                other.memory = VK_NULL_HANDLE;
                other.format = VK_FORMAT_UNDEFINED;
            }
            return *this;
        }

        ~Image() {
            destroy();
        }

        void destroy() {
            if (device != VK_NULL_HANDLE) {
                if (view != VK_NULL_HANDLE) {
                    vkDestroyImageView(device, view, nullptr);
                }
                if (image != VK_NULL_HANDLE) {
                    vkDestroyImage(device, image, nullptr);
                }
                if (memory != VK_NULL_HANDLE) {
                    vkFreeMemory(device, memory, nullptr);
                }
            }
            image = VK_NULL_HANDLE;
            view = VK_NULL_HANDLE;
            memory = VK_NULL_HANDLE;
            format = VK_FORMAT_UNDEFINED;
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
        appInfo.pApplicationName = "nrc_vulkan_inference_scalar_test";
        appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        appInfo.pEngineName = "NRC";
        appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
        appInfo.apiVersion = loaderVersion < VK_API_VERSION_1_1 ? loaderVersion : VK_API_VERSION_1_1;

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
            if (properties.apiVersion < VK_API_VERSION_1_1) {
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

        throw SkipTest("no Vulkan 1.1 compute-capable device was found");
    }

    void check_format(VkFormat format, const char *name) const {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice_, format, &properties);
        const VkFormatFeatureFlags required =
            VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
            VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
            VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        if ((properties.optimalTilingFeatures & required) != required) {
            throw SkipTest(std::string("format does not support storage image transfer usage: ") + name);
        }
    }

    void check_formats() const {
        check_format(VK_FORMAT_R32G32B32A32_SFLOAT, "VK_FORMAT_R32G32B32A32_SFLOAT");
        check_format(VK_FORMAT_R32G32_SFLOAT, "VK_FORMAT_R32G32_SFLOAT");
    }

    void create_device() {
        float priority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = queueFamilyIndex_;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &priority;

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = 1;
        createInfo.pQueueCreateInfos = &queueInfo;

        check_vk(vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_), "vkCreateDevice");
        vkGetDeviceQueue(device_, queueFamilyIndex_, 0, &queue_);
    }

    void create_descriptor_set_layout() {
        std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
        for (std::uint32_t i = 0; i < bindings.size(); ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[i].descriptorType =
                (i == 3 || i == 4) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }

        VkDescriptorSetLayoutCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        createInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
        createInfo.pBindings = bindings.data();

        check_vk(vkCreateDescriptorSetLayout(device_, &createInfo, nullptr, &descriptorSetLayout_),
                 "vkCreateDescriptorSetLayout");
    }

    void create_pipeline_layout() {
        VkPipelineLayoutCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        createInfo.setLayoutCount = 1;
        createInfo.pSetLayouts = &descriptorSetLayout_;

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

    std::uint32_t find_memory_type(std::uint32_t typeBits,
                                   VkMemoryPropertyFlags required,
                                   VkMemoryPropertyFlags preferred) const {
        std::uint32_t fallback = std::numeric_limits<std::uint32_t>::max();
        for (std::uint32_t i = 0; i < memoryProperties_.memoryTypeCount; ++i) {
            if ((typeBits & (1u << i)) == 0) {
                continue;
            }
            const VkMemoryPropertyFlags flags = memoryProperties_.memoryTypes[i].propertyFlags;
            if ((flags & required) != required) {
                continue;
            }
            if ((flags & preferred) == preferred) {
                return i;
            }
            if (fallback == std::numeric_limits<std::uint32_t>::max()) {
                fallback = i;
            }
        }
        if (fallback != std::numeric_limits<std::uint32_t>::max()) {
            return fallback;
        }
        throw SkipTest("no compatible Vulkan memory type was found");
    }

    Buffer create_host_buffer(VkDeviceSize size, VkBufferUsageFlags usage) {
        Buffer out{};
        out.device = device_;
        out.size = size;

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check_vk(vkCreateBuffer(device_, &bufferInfo, nullptr, &out.buffer), "vkCreateBuffer");

        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, out.buffer, &requirements);

        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = requirements.size;
        allocateInfo.memoryTypeIndex = find_memory_type(
            requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        check_vk(vkAllocateMemory(device_, &allocateInfo, nullptr, &out.memory), "vkAllocateMemory");
        check_vk(vkBindBufferMemory(device_, out.buffer, out.memory, 0), "vkBindBufferMemory");

        return out;
    }

    Image create_image(VkFormat format) {
        Image out{};
        out.device = device_;
        out.format = format;

        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = format;
        imageInfo.extent = {kScreenWidth, kScreenHeight, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        check_vk(vkCreateImage(device_, &imageInfo, nullptr, &out.image), "vkCreateImage");

        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device_, out.image, &requirements);

        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = requirements.size;
        allocateInfo.memoryTypeIndex = find_memory_type(
            requirements.memoryTypeBits,
            0,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check_vk(vkAllocateMemory(device_, &allocateInfo, nullptr, &out.memory), "vkAllocateMemory");
        check_vk(vkBindImageMemory(device_, out.image, out.memory, 0), "vkBindImageMemory");

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = out.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;
        check_vk(vkCreateImageView(device_, &viewInfo, nullptr, &out.view), "vkCreateImageView");

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
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[0].descriptorCount = 4;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        poolSizes[1].descriptorCount = 2;

        VkDescriptorPoolCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        createInfo.maxSets = 1;
        createInfo.poolSizeCount = static_cast<std::uint32_t>(poolSizes.size());
        createInfo.pPoolSizes = poolSizes.data();

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
                               const Buffer &evals,
                               const Buffer &evalCount,
                               const Image &biasImage,
                               const Image &factorImage,
                               const Buffer &trains) const {
        std::array<VkDescriptorBufferInfo, 4> bufferInfos{};
        bufferInfos[0].buffer = weights.buffer;
        bufferInfos[0].range = weights.size;
        bufferInfos[1].buffer = evals.buffer;
        bufferInfos[1].range = evals.size;
        bufferInfos[2].buffer = evalCount.buffer;
        bufferInfos[2].range = evalCount.size;
        bufferInfos[3].buffer = trains.buffer;
        bufferInfos[3].range = trains.size;

        std::array<VkDescriptorImageInfo, 2> imageInfos{};
        imageInfos[0].imageView = biasImage.view;
        imageInfos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageInfos[1].imageView = factorImage.view;
        imageInfos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 6> writes{};
        for (std::uint32_t i = 0; i < writes.size(); ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptorSet;
            writes[i].descriptorCount = 1;
        }

        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &bufferInfos[0];
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &bufferInfos[1];
        writes[2].dstBinding = 2;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &bufferInfos[2];
        writes[3].dstBinding = 3;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[3].pImageInfo = &imageInfos[0];
        writes[4].dstBinding = 4;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[4].pImageInfo = &imageInfos[1];
        writes[5].dstBinding = 5;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[5].pBufferInfo = &bufferInfos[3];

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

    VkImageMemoryBarrier image_barrier(const Image &image,
                                       VkImageLayout oldLayout,
                                       VkImageLayout newLayout,
                                       VkAccessFlags srcAccess,
                                       VkAccessFlags dstAccess) const {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = srcAccess;
        barrier.dstAccessMask = dstAccess;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image.image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        return barrier;
    }

    void copy_buffer_to_image(VkCommandBuffer commandBuffer, const Buffer &buffer, const Image &image) const {
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {kScreenWidth, kScreenHeight, 1};
        vkCmdCopyBufferToImage(commandBuffer,
                               buffer.buffer,
                               image.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               1,
                               &region);
    }

    void copy_image_to_buffer(VkCommandBuffer commandBuffer, const Image &image, const Buffer &buffer) const {
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {kScreenWidth, kScreenHeight, 1};
        vkCmdCopyImageToBuffer(commandBuffer,
                               image.image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               buffer.buffer,
                               1,
                               &region);
    }

    void record_dispatch(VkCommandBuffer commandBuffer,
                         VkDescriptorSet descriptorSet,
                         const Buffer &biasUpload,
                         const Buffer &factorUpload,
                         const Image &biasImage,
                         const Image &factorImage,
                         const Buffer &biasReadback,
                         std::uint32_t evalCount) const {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        check_vk(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");

        VkMemoryBarrier hostToTransfer{};
        hostToTransfer.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        hostToTransfer.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        hostToTransfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_HOST_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0,
                             1,
                             &hostToTransfer,
                             0,
                             nullptr,
                             0,
                             nullptr);

        std::array<VkImageMemoryBarrier, 2> uploadBarriers = {
            image_barrier(biasImage,
                          VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          0,
                          VK_ACCESS_TRANSFER_WRITE_BIT),
            image_barrier(factorImage,
                          VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          0,
                          VK_ACCESS_TRANSFER_WRITE_BIT),
        };
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             static_cast<std::uint32_t>(uploadBarriers.size()),
                             uploadBarriers.data());

        copy_buffer_to_image(commandBuffer, biasUpload, biasImage);
        copy_buffer_to_image(commandBuffer, factorUpload, factorImage);

        std::array<VkImageMemoryBarrier, 2> computeImageBarriers = {
            image_barrier(biasImage,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_GENERAL,
                          VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT),
            image_barrier(factorImage,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_GENERAL,
                          VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT),
        };

        VkMemoryBarrier hostToShader{};
        hostToShader.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        hostToShader.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        hostToShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                             1,
                             &hostToShader,
                             0,
                             nullptr,
                             static_cast<std::uint32_t>(computeImageBarriers.size()),
                             computeImageBarriers.data());

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(commandBuffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipelineLayout_,
                                0,
                                1,
                                &descriptorSet,
                                0,
                                nullptr);

        const std::uint32_t groupCount = (evalCount + kWorkgroupSize - 1u) / kWorkgroupSize;
        vkCmdDispatch(commandBuffer, groupCount, 1, 1);

        VkImageMemoryBarrier readbackBarrier = image_barrier(biasImage,
                                                             VK_IMAGE_LAYOUT_GENERAL,
                                                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                             VK_ACCESS_SHADER_WRITE_BIT,
                                                             VK_ACCESS_TRANSFER_READ_BIT);
        VkMemoryBarrier shaderToHost{};
        shaderToHost.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        shaderToHost.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        shaderToHost.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                             0,
                             1,
                             &shaderToHost,
                             0,
                             nullptr,
                             1,
                             &readbackBarrier);

        copy_image_to_buffer(commandBuffer, biasImage, biasReadback);

        VkMemoryBarrier transferToHost{};
        transferToHost.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        transferToHost.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        transferToHost.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT,
                             0,
                             1,
                             &transferToHost,
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
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
};

bool check_vec4(const Vec4 &actual, const Vec4 &expected, const char *label) {
    const bool ok = near(actual.x, expected.x) &&
                    near(actual.y, expected.y) &&
                    near(actual.z, expected.z) &&
                    near(actual.w, expected.w);
    if (!ok) {
        std::cerr << "FAIL: " << label
                  << " expected (" << expected.x << ", " << expected.y << ", " << expected.z << ", " << expected.w
                  << ") got (" << actual.x << ", " << actual.y << ", " << actual.z << ", " << actual.w << ")\n";
    }
    return ok;
}

bool check_train_record(const nrc::NrcTrainRecord &actual,
                        const nrc::NrcTrainRecord &expected,
                        std::uint32_t index) {
    const bool ok = near(actual.biasR, expected.biasR) &&
                    near(actual.biasG, expected.biasG) &&
                    near(actual.biasB, expected.biasB) &&
                    near(actual.factorR, expected.factorR) &&
                    near(actual.factorG, expected.factorG) &&
                    near(actual.factorB, expected.factorB);
    if (!ok) {
        std::cerr << "FAIL: train record " << index
                  << " expected bias (" << expected.biasR << ", " << expected.biasG << ", " << expected.biasB
                  << ") factor (" << expected.factorR << ", " << expected.factorG << ", " << expected.factorB
                  << ") got bias (" << actual.biasR << ", " << actual.biasG << ", " << actual.biasB
                  << ") factor (" << actual.factorR << ", " << actual.factorG << ", " << actual.factorB << ")\n";
    }
    return ok;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: nrc_vulkan_inference_scalar_test <nrc_inference_scalar_dispatch.spv>\n";
        return EXIT_FAILURE;
    }

    try {
        const std::vector<float> weights = make_constant_prediction_weights();
        const std::vector<nrc::PackedFp16x8> packedWeights = nrc::pack_weights_fp16_uvec4(weights);
        const std::vector<nrc::NrcEvalRecord> evalRecords = make_eval_records();
        const std::vector<Vec4> biasFactorR = make_bias_factor_r();
        const std::vector<Vec2> factorGb = make_factor_gb();
        const std::vector<nrc::NrcTrainRecord> trainRecords = make_train_records();

        VulkanInferenceScalarRunner runner(argv[1]);
        std::vector<Vec4> actualBiasFactorR;
        std::vector<nrc::NrcTrainRecord> actualTrainRecords;
        runner.run(packedWeights,
                   evalRecords,
                   kEvalCount,
                   biasFactorR,
                   factorGb,
                   trainRecords,
                   actualBiasFactorR,
                   actualTrainRecords);

        std::vector<Vec4> expectedBiasFactorR = biasFactorR;
        const Vec3 pred = prediction();
        const auto apply_screen = [&](std::uint32_t x, std::uint32_t y) {
            const std::uint32_t index = y * kScreenWidth + x;
            expectedBiasFactorR[index].x =
                biasFactorR[index].x + biasFactorR[index].w * pred.x;
            expectedBiasFactorR[index].y =
                biasFactorR[index].y + factorGb[index].x * pred.y;
            expectedBiasFactorR[index].z =
                biasFactorR[index].z + factorGb[index].y * pred.z;
            expectedBiasFactorR[index].w = 0.0f;
        };
        apply_screen(1, 1);
        apply_screen(2, 0);

        std::vector<nrc::NrcTrainRecord> expectedTrainRecords = trainRecords;
        for (std::uint32_t i = 2; i <= 4; ++i) {
            expectedTrainRecords[i].biasR = trainRecords[i].biasR + trainRecords[i].factorR * pred.x;
            expectedTrainRecords[i].biasG = trainRecords[i].biasG + trainRecords[i].factorG * pred.y;
            expectedTrainRecords[i].biasB = trainRecords[i].biasB + trainRecords[i].factorB * pred.z;
        }

        bool ok = true;
        for (std::uint32_t i = 0; i < expectedBiasFactorR.size(); ++i) {
            ok &= check_vec4(actualBiasFactorR[i], expectedBiasFactorR[i], "screen pixel");
        }

        for (std::uint32_t i = 0; i < 8; ++i) {
            ok &= check_train_record(actualTrainRecords[i], expectedTrainRecords[i], i);
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
