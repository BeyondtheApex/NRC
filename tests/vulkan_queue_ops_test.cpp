#include "nrc/nrc_constants.hpp"
#include "nrc/nrc_records.hpp"
#include "nrc/vulkan/nrc_vulkan_barriers.hpp"
#include "nrc/vulkan/nrc_vulkan_descriptors.hpp"
#include "nrc/vulkan/nrc_vulkan_pipeline.hpp"
#include "nrc/vulkan/nrc_vulkan_shader.hpp"

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
constexpr std::uint32_t kEvalCapacity = 17;
constexpr std::uint32_t kInvocationCount = 64;
constexpr std::uint32_t kTrainRequestCount = 300;

struct SkipTest : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct PushConstants {
    std::uint32_t mode = 0;
    std::uint32_t evalCapacity = 0;
    std::uint32_t trainRequestCount = 0;
    std::uint32_t invocationCount = 0;
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

std::uint32_t decode_screen_x(std::uint32_t dst) {
    dst >>= 1u;
    return dst & 0x7fffu;
}

std::uint32_t decode_screen_y(std::uint32_t dst) {
    dst >>= 1u;
    return dst >> 15u;
}

bool near(float lhs, float rhs) {
    return std::fabs(lhs - rhs) <= 1.0e-6f;
}

class VulkanQueueOpsRunner {
public:
    explicit VulkanQueueOpsRunner(const char *spirvPath) {
        create_instance();
        pick_physical_device();
        create_device();
        create_descriptor_set_layout();
        create_pipeline_layout();
        create_pipeline(spirvPath);
        create_command_pool();
    }

    VulkanQueueOpsRunner(const VulkanQueueOpsRunner &) = delete;
    VulkanQueueOpsRunner &operator=(const VulkanQueueOpsRunner &) = delete;

    ~VulkanQueueOpsRunner() {
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

    void run(nrc::NrcQueueState &queueState,
             std::vector<nrc::NrcEvalRecord> &evalRecords,
             std::vector<nrc::NrcTrainRecord> &trainRecords) {
        Buffer state = create_host_buffer(sizeof(nrc::NrcQueueState));
        Buffer evals = create_host_buffer(static_cast<VkDeviceSize>(evalRecords.size() * sizeof(nrc::NrcEvalRecord)));
        Buffer trains = create_host_buffer(static_cast<VkDeviceSize>(trainRecords.size() * sizeof(nrc::NrcTrainRecord)));

        nrc::NrcQueueState initialState{};
        initialState.evalCount = 999u;
        initialState.trainCounts = {999u, 999u, 999u, 999u};
        upload(state, &initialState, sizeof(initialState));
        upload(evals, evalRecords.data(), static_cast<VkDeviceSize>(evalRecords.size() * sizeof(nrc::NrcEvalRecord)));
        upload(trains, trainRecords.data(), static_cast<VkDeviceSize>(trainRecords.size() * sizeof(nrc::NrcTrainRecord)));

        VkDescriptorPool descriptorPool = create_descriptor_pool();
        VkDescriptorSet descriptorSet = allocate_descriptor_set(descriptorPool);
        update_descriptor_set(descriptorSet, state, evals, trains);

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

        download(state, &queueState, sizeof(queueState));
        download(evals, evalRecords.data(), static_cast<VkDeviceSize>(evalRecords.size() * sizeof(nrc::NrcEvalRecord)));
        download(trains, trainRecords.data(), static_cast<VkDeviceSize>(trainRecords.size() * sizeof(nrc::NrcTrainRecord)));
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
        appInfo.pApplicationName = "nrc_vulkan_queue_ops_test";
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
        const std::array<nrc::vulkan::VulkanStorageBufferBinding, 3> bindings = {{
            {0, VK_SHADER_STAGE_COMPUTE_BIT},
            {1, VK_SHADER_STAGE_COMPUTE_BIT},
            {2, VK_SHADER_STAGE_COMPUTE_BIT},
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
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        createInfo.setLayoutCount = 1;
        createInfo.pSetLayouts = &descriptorSetLayout_;
        createInfo.pushConstantRangeCount = 1;
        createInfo.pPushConstantRanges = &pushConstantRange;

        check_vk(vkCreatePipelineLayout(device_, &createInfo, nullptr, &pipelineLayout_),
                 "vkCreatePipelineLayout");
    }

    void create_pipeline(const char *spirvPath) {
        const std::vector<std::uint32_t> spirv = read_spirv(spirvPath);

        nrc::vulkan::VulkanShaderModuleCreateInfo shaderCreateInfo{};
        shaderCreateInfo.device = device_;
        shaderCreateInfo.code = std::span<const std::uint32_t>(spirv.data(), spirv.size());
        const nrc::vulkan::VulkanShaderModuleCreateResult shaderResult =
            nrc::vulkan::create_shader_module(shaderCreateInfo);
        check_vk(shaderResult.validation.vkResult, "vkCreateShaderModule");

        nrc::vulkan::VulkanBasicComputePipelineCreateInfo pipelineCreateInfo{};
        pipelineCreateInfo.device = device_;
        pipelineCreateInfo.pipelineLayout = pipelineLayout_;
        pipelineCreateInfo.shaderModule = shaderResult.shaderModule;
        const nrc::vulkan::VulkanComputePipelineCreateResult pipelineResult =
            nrc::vulkan::create_compute_pipeline(pipelineCreateInfo);
        if (pipelineResult.validation.structuredException) {
            vkDestroyShaderModule(device_, shaderResult.shaderModule, nullptr);
            throw SkipTest("vkCreateComputePipelines raised a structured exception");
        }
        pipeline_ = pipelineResult.pipeline;
        vkDestroyShaderModule(device_, shaderResult.shaderModule, nullptr);
        check_vk(pipelineResult.validation.vkResult, "vkCreateComputePipelines");
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
                               const Buffer &state,
                               const Buffer &evals,
                               const Buffer &trains) const {
        const std::array<nrc::vulkan::VulkanStorageBufferDescriptorWrite, 3> writes = {{
            {0, state.buffer, 0, state.size},
            {1, evals.buffer, 0, evals.size},
            {2, trains.buffer, 0, trains.size},
        }};
        nrc::vulkan::VulkanStorageBufferDescriptorUpdate update{};
        update.device = device_;
        update.descriptorSet = descriptorSet;
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

    void dispatch(VkCommandBuffer commandBuffer, VkDescriptorSet descriptorSet, const PushConstants &constants) const {
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
                           sizeof(constants),
                           &constants);
        const std::uint32_t groupCount =
            (std::max(constants.invocationCount, nrc::kTrainBatchCount) + kWorkgroupSize - 1u) / kWorkgroupSize;
        vkCmdDispatch(commandBuffer, groupCount, 1, 1);
    }

    void record_dispatch(VkCommandBuffer commandBuffer, VkDescriptorSet descriptorSet) const {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        check_vk(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");

        nrc::vulkan::VulkanValidationStep barrierValidation =
            nrc::vulkan::cmd_memory_barrier(
                commandBuffer,
                nrc::vulkan::host_write_to_compute_shader_barrier(
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));
        check_vk(barrierValidation.vkResult, "host-to-shader barrier");

        dispatch(commandBuffer, descriptorSet, {0u, kEvalCapacity, kTrainRequestCount, kInvocationCount});

        barrierValidation =
            nrc::vulkan::cmd_memory_barrier(
                commandBuffer,
                nrc::vulkan::compute_shader_write_to_compute_shader_barrier(
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));
        check_vk(barrierValidation.vkResult, "queue reset-to-push barrier");

        dispatch(commandBuffer, descriptorSet, {1u, kEvalCapacity, kTrainRequestCount, kInvocationCount});

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

bool check_eval_record(const nrc::NrcEvalRecord &record, std::uint32_t index) {
    bool ok = true;
    if ((record.dst & 1u) != 0u || decode_screen_x(record.dst) != 7u || decode_screen_y(record.dst) != 3u) {
        std::cerr << "FAIL: eval record " << index << " has unexpected dst " << record.dst << '\n';
        ok = false;
    }
    if (record.packedInput.barycentric2x16U == 0u || record.packedInput.scatteredDir2x16U == 0u) {
        std::cerr << "FAIL: eval record " << index << " did not store packed input\n";
        ok = false;
    }
    return ok;
}

bool check_train_record(const nrc::NrcTrainRecord &record, std::uint32_t index) {
    bool ok = true;
    if (!near(record.biasR, 1.0f) ||
        !near(record.biasG, 2.0f) ||
        !near(record.biasB, 3.0f) ||
        !near(record.factorR, 4.0f) ||
        !near(record.factorG, 5.0f) ||
        !near(record.factorB, 6.0f)) {
        std::cerr << "FAIL: train record " << index << " has unexpected bias/factor\n";
        ok = false;
    }
    if (record.packedInput.barycentric2x16U == 0u || record.packedInput.scatteredDir2x16U == 0u) {
        std::cerr << "FAIL: train record " << index << " did not store packed input\n";
        ok = false;
    }
    return ok;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: nrc_vulkan_queue_ops_test <nrc_queue_ops_dispatch.spv>\n";
        return EXIT_FAILURE;
    }

    try {
        nrc::NrcQueueState queueState{};
        std::vector<nrc::NrcEvalRecord> evalRecords(kEvalCapacity);
        std::vector<nrc::NrcTrainRecord> trainRecords(nrc::kTrainBatchCount * nrc::kTrainBatchSize);

        VulkanQueueOpsRunner runner(argv[1]);
        runner.run(queueState, evalRecords, trainRecords);

        bool ok = true;
        ok &= queueState.evalCount == kEvalCapacity;
        if (queueState.evalCount != kEvalCapacity) {
            std::cerr << "FAIL: eval count expected " << kEvalCapacity
                      << " got " << queueState.evalCount << '\n';
        }

        ok &= queueState.trainCounts[0] == nrc::kTrainBatchSize;
        if (queueState.trainCounts[0] != nrc::kTrainBatchSize) {
            std::cerr << "FAIL: train batch 0 count expected " << nrc::kTrainBatchSize
                      << " got " << queueState.trainCounts[0] << '\n';
        }
        for (std::uint32_t batch = 1; batch < nrc::kTrainBatchCount; ++batch) {
            ok &= queueState.trainCounts[batch] == 0u;
            if (queueState.trainCounts[batch] != 0u) {
                std::cerr << "FAIL: train batch " << batch
                          << " count expected 0 got " << queueState.trainCounts[batch] << '\n';
            }
        }

        for (std::uint32_t i = 0; i < evalRecords.size(); ++i) {
            ok &= check_eval_record(evalRecords[i], i);
        }
        for (std::uint32_t i = 0; i < nrc::kTrainBatchSize; ++i) {
            ok &= check_train_record(trainRecords[i], i);
        }
        const std::uint32_t batch1First = nrc::kTrainBatchSize;
        ok &= near(trainRecords[batch1First].biasR, 0.0f) &&
              near(trainRecords[batch1First].factorR, 0.0f);
        if (!near(trainRecords[batch1First].biasR, 0.0f) ||
            !near(trainRecords[batch1First].factorR, 0.0f)) {
            std::cerr << "FAIL: train batch 1 should remain untouched\n";
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
