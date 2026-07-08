#include "nrc/nrc_constants.hpp"
#include "nrc/nrc_records.hpp"
#include "nrc/nrc_reference_nn.hpp"
#include "nrc/nrc_weights.hpp"
#include "sample_config.hpp"

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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kWidth = 4;
constexpr std::uint32_t kHeight = 4;
constexpr std::uint32_t kPixelCount = kWidth * kHeight;
constexpr std::uint32_t kTrainRecordCount = 2;
constexpr std::uint32_t kEvalCapacity = kPixelCount + 1;
constexpr std::uint32_t kWorkgroupSize = 64;
constexpr std::uint32_t kSceneTriangleCount = 2;

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

struct SampleSceneVertex {
    Vec4 position;
    Vec4 normalRoughness;
    Vec4 diffuse;
    Vec4 specular;
};

struct SampleSceneTriangle {
    SampleSceneVertex v0;
    SampleSceneVertex v1;
    SampleSceneVertex v2;
};

struct ExpectedTrainPayload {
    Vec3 bias;
    Vec3 factor;
};

static_assert(sizeof(SampleSceneVertex) == 64);
static_assert(sizeof(SampleSceneTriangle) == 192);

struct ProducerConstants {
    std::uint32_t mode = 0;
    std::uint32_t evalCapacity = 0;
    std::uint32_t pixelCount = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t trainRecordCount = 0;
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

bool near(float lhs, float rhs) {
    return std::fabs(lhs - rhs) <= 1.0e-5f;
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

std::vector<Vec4> make_bias_factor_r() {
    std::vector<Vec4> pixels(kPixelCount);
    for (std::uint32_t y = 0; y < kHeight; ++y) {
        for (std::uint32_t x = 0; x < kWidth; ++x) {
            const std::uint32_t index = y * kWidth + x;
            pixels[index] = {
                0.1f * static_cast<float>(x),
                0.2f * static_cast<float>(y),
                0.3f * static_cast<float>(x + y),
                1.0f,
            };
        }
    }
    return pixels;
}

std::vector<Vec2> make_factor_gb() {
    std::vector<Vec2> pixels(kPixelCount);
    for (std::uint32_t i = 0; i < kPixelCount; ++i) {
        pixels[i] = {2.0f, 3.0f};
    }
    return pixels;
}

SampleSceneVertex make_scene_vertex(Vec4 position,
                                    Vec4 normalRoughness,
                                    Vec4 diffuse,
                                    Vec4 specular) {
    return {position, normalRoughness, diffuse, specular};
}

std::array<SampleSceneTriangle, kSceneTriangleCount> make_sample_scene() {
    const Vec4 normalA{0.0f, 0.0f, 1.0f, 0.25f};
    const Vec4 normalB{0.0f, 0.2f, 0.98f, 0.55f};
    const Vec4 specularA{0.04f, 0.05f, 0.06f, 1.0f};
    const Vec4 specularB{0.12f, 0.10f, 0.08f, 1.0f};

    return {
        SampleSceneTriangle{
            make_scene_vertex({-1.0f, -1.0f, 0.0f, 1.0f}, normalA, {0.75f, 0.18f, 0.12f, 1.0f}, specularA),
            make_scene_vertex({1.0f, -1.0f, 0.0f, 1.0f}, normalA, {0.18f, 0.70f, 0.24f, 1.0f}, specularA),
            make_scene_vertex({-1.0f, 1.0f, 0.0f, 1.0f}, normalA, {0.18f, 0.30f, 0.82f, 1.0f}, specularA),
        },
        SampleSceneTriangle{
            make_scene_vertex({1.0f, 1.0f, 0.0f, 1.0f}, normalB, {0.82f, 0.72f, 0.18f, 1.0f}, specularB),
            make_scene_vertex({-1.0f, 1.0f, 0.0f, 1.0f}, normalB, {0.18f, 0.64f, 0.78f, 1.0f}, specularB),
            make_scene_vertex({1.0f, -1.0f, 0.0f, 1.0f}, normalB, {0.70f, 0.28f, 0.78f, 1.0f}, specularB),
        },
    };
}

std::uint32_t pack_unorm2x16(float x, float y) {
    const auto pack = [](float value) {
        const float clamped = std::clamp(value, 0.0f, 1.0f);
        return static_cast<std::uint32_t>(std::round(clamped * 65535.0f)) & 0xffffu;
    };
    return pack(x) | (pack(y) << 16u);
}

Vec2 unpack_unorm2x16(std::uint32_t packedValue) {
    constexpr float kInvUnorm16 = 1.0f / 65535.0f;
    return {
        static_cast<float>(packedValue & 0xffffu) * kInvUnorm16,
        static_cast<float>((packedValue >> 16u) & 0xffffu) * kInvUnorm16,
    };
}

Vec4 mix_vec4(Vec4 v0, Vec4 v1, Vec4 v2, Vec2 barycentric2) {
    const float w1 = barycentric2.x;
    const float w2 = barycentric2.y;
    const float w0 = 1.0f - w1 - w2;
    return {
        v0.x * w0 + v1.x * w1 + v2.x * w2,
        v0.y * w0 + v1.y * w1 + v2.y * w2,
        v0.z * w0 + v1.z * w1 + v2.z * w2,
        v0.w * w0 + v1.w * w1 + v2.w * w2,
    };
}

Vec3 normalize_vec3(Vec3 value) {
    const float length = std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
    if (length <= 0.0001f) {
        return {0.0f, 0.0f, 1.0f};
    }
    return {value.x / length, value.y / length, value.z / length};
}

struct SampleRay {
    Vec3 origin;
    Vec3 direction;
};

struct SampleHit {
    std::uint32_t primitiveId = 0;
    Vec2 barycentric2;
    float t = std::numeric_limits<float>::max();
};

Vec3 sub_vec3(Vec3 lhs, Vec3 rhs) {
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

Vec3 vec4_xyz(Vec4 value) {
    return {value.x, value.y, value.z};
}

Vec3 cross_vec3(Vec3 lhs, Vec3 rhs) {
    return {
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x,
    };
}

float dot_vec3(Vec3 lhs, Vec3 rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

SampleRay make_primary_ray(std::uint32_t x, std::uint32_t y) {
    const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(kWidth);
    const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(kHeight);
    const Vec3 origin{0.0f, 0.0f, -2.0f};
    const Vec3 target{u * 2.0f - 1.0f, v * 2.0f - 1.0f, 0.0f};
    return {origin, normalize_vec3(sub_vec3(target, origin))};
}

bool intersect_triangle(const SampleRay &ray,
                        std::uint32_t primitiveId,
                        const SampleSceneTriangle &triangle,
                        SampleHit &hit) {
    const Vec3 p0 = vec4_xyz(triangle.v0.position);
    const Vec3 p1 = vec4_xyz(triangle.v1.position);
    const Vec3 p2 = vec4_xyz(triangle.v2.position);
    const Vec3 edge1 = sub_vec3(p1, p0);
    const Vec3 edge2 = sub_vec3(p2, p0);
    const Vec3 pvec = cross_vec3(ray.direction, edge2);
    const float det = dot_vec3(edge1, pvec);
    if (std::fabs(det) < 0.000001f) {
        return false;
    }

    const float invDet = 1.0f / det;
    const Vec3 tvec = sub_vec3(ray.origin, p0);
    const float u = dot_vec3(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    const Vec3 qvec = cross_vec3(tvec, edge1);
    const float v = dot_vec3(ray.direction, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    const float t = dot_vec3(edge2, qvec) * invDet;
    if (t <= 0.0f) {
        return false;
    }

    hit.primitiveId = primitiveId;
    hit.barycentric2 = {u, v};
    hit.t = t;
    return true;
}

bool trace_primary_hit(std::uint32_t x, std::uint32_t y, SampleHit &hit) {
    const std::array<SampleSceneTriangle, kSceneTriangleCount> sceneTriangles = make_sample_scene();
    const SampleRay ray = make_primary_ray(x, y);
    bool found = false;
    hit.t = std::numeric_limits<float>::max();

    for (std::uint32_t primitiveId = 0; primitiveId < kSceneTriangleCount; ++primitiveId) {
        SampleHit candidate{};
        if (intersect_triangle(ray, primitiveId, sceneTriangles[primitiveId], candidate) &&
            candidate.t < hit.t) {
            hit = candidate;
            found = true;
        }
    }

    return found;
}

nrc::PackedNrcInput expected_packed_input(std::uint32_t x, std::uint32_t y) {
    SampleHit hit{};
    if (!trace_primary_hit(x, y, hit)) {
        throw std::runtime_error("sample CPU reference ray missed the scene");
    }

    const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(kWidth);
    const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(kHeight);

    nrc::PackedNrcInput packedInput{};
    packedInput.primitiveId = hit.primitiveId;
    packedInput.flipBitInstanceId = hit.primitiveId;
    packedInput.barycentric2x16U = pack_unorm2x16(hit.barycentric2.x, hit.barycentric2.y);
    packedInput.scatteredDir2x16U = pack_unorm2x16(u, v);
    return packedInput;
}

ExpectedTrainPayload expected_train_payload(const nrc::PackedNrcInput &packedInput) {
    const std::array<SampleSceneTriangle, kSceneTriangleCount> sceneTriangles = make_sample_scene();
    const std::uint32_t primitiveId = std::min(packedInput.primitiveId, kSceneTriangleCount - 1u);
    const SampleSceneTriangle &triangle = sceneTriangles[primitiveId];
    const Vec2 barycentric2 = unpack_unorm2x16(packedInput.barycentric2x16U);

    const Vec4 normalRoughness = mix_vec4(triangle.v0.normalRoughness,
                                          triangle.v1.normalRoughness,
                                          triangle.v2.normalRoughness,
                                          barycentric2);
    const Vec4 diffuse = mix_vec4(triangle.v0.diffuse,
                                  triangle.v1.diffuse,
                                  triangle.v2.diffuse,
                                  barycentric2);
    const Vec4 specular = mix_vec4(triangle.v0.specular,
                                   triangle.v1.specular,
                                   triangle.v2.specular,
                                   barycentric2);

    const Vec3 normal = normalize_vec3({normalRoughness.x, normalRoughness.y, normalRoughness.z});
    const float nDotL = std::max(normal.z, 0.0f);
    const float roughness = std::clamp(normalRoughness.w, 0.0f, 1.0f);
    const float specularScale = 1.0f - roughness;

    ExpectedTrainPayload payload{};
    payload.bias = {
        diffuse.x * 1.4f * nDotL,
        diffuse.y * 1.1f * nDotL,
        diffuse.z * 0.9f * nDotL,
    };
    payload.factor = {
        std::max(diffuse.x + specular.x * specularScale, 0.05f),
        std::max(diffuse.y + specular.y * specularScale, 0.05f),
        std::max(diffuse.z + specular.z * specularScale, 0.05f),
    };
    return payload;
}

bool same_packed_input(const nrc::PackedNrcInput &lhs, const nrc::PackedNrcInput &rhs) {
    return lhs.primitiveId == rhs.primitiveId &&
           lhs.flipBitInstanceId == rhs.flipBitInstanceId &&
           lhs.barycentric2x16U == rhs.barycentric2x16U &&
           lhs.scatteredDir2x16U == rhs.scatteredDir2x16U;
}

void report_packed_input_mismatch(const char *label,
                                  const nrc::PackedNrcInput &expected,
                                  const nrc::PackedNrcInput &actual) {
    std::cerr << "FAIL: " << label
              << " expected packed input (primitive=" << expected.primitiveId
              << ", instance=" << expected.flipBitInstanceId
              << ", bary=0x" << std::hex << expected.barycentric2x16U
              << ", scatter=0x" << expected.scatteredDir2x16U
              << ") got (primitive=" << std::dec << actual.primitiveId
              << ", instance=" << actual.flipBitInstanceId
              << ", bary=0x" << std::hex << actual.barycentric2x16U
              << ", scatter=0x" << actual.scatteredDir2x16U
              << std::dec << ")\n";
}

std::uint32_t decode_screen_x(std::uint32_t dst) {
    dst >>= 1u;
    return dst & 0x7fffu;
}

std::uint32_t decode_screen_y(std::uint32_t dst) {
    dst >>= 1u;
    return dst >> 15u;
}

std::uint32_t decode_train_batch(std::uint32_t dst) {
    dst >>= 1u;
    return dst & 3u;
}

std::uint32_t decode_train_left(std::uint32_t dst) {
    dst >>= 1u;
    return (dst >> 2u) & 0x3fffu;
}

std::uint32_t decode_train_right(std::uint32_t dst) {
    dst >>= 1u;
    return dst >> 16u;
}

class MinimalIntegrationSample {
public:
    MinimalIntegrationSample(const char *producerSpirv, const char *inferenceSpirv) {
        create_instance();
        pick_physical_device();
        check_formats();
        create_device();
        create_command_pool();
        create_producer_pipeline(producerSpirv);
        create_inference_pipeline(inferenceSpirv);
    }

    MinimalIntegrationSample(const MinimalIntegrationSample &) = delete;
    MinimalIntegrationSample &operator=(const MinimalIntegrationSample &) = delete;

    ~MinimalIntegrationSample() {
        if (device_ != VK_NULL_HANDLE) {
            if (commandPool_ != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device_, commandPool_, nullptr);
            }
            if (producerPipeline_ != VK_NULL_HANDLE) {
                vkDestroyPipeline(device_, producerPipeline_, nullptr);
            }
            if (inferencePipeline_ != VK_NULL_HANDLE) {
                vkDestroyPipeline(device_, inferencePipeline_, nullptr);
            }
            if (producerPipelineLayout_ != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device_, producerPipelineLayout_, nullptr);
            }
            if (inferencePipelineLayout_ != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device_, inferencePipelineLayout_, nullptr);
            }
            if (producerSetLayout_ != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device_, producerSetLayout_, nullptr);
            }
            if (inferenceSetLayout_ != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device_, inferenceSetLayout_, nullptr);
            }
            vkDestroyDevice(device_, nullptr);
        }
        if (instance_ != VK_NULL_HANDLE) {
            vkDestroyInstance(instance_, nullptr);
        }
    }

    bool run() {
        const std::vector<float> weights = make_constant_prediction_weights();
        const std::vector<nrc::PackedFp16x8> packedWeights = nrc::pack_weights_fp16_uvec4(weights);
        const std::vector<Vec4> initialBiasFactorR = make_bias_factor_r();
        const std::vector<Vec2> initialFactorGb = make_factor_gb();
        const std::array<SampleSceneTriangle, kSceneTriangleCount> sceneTriangles = make_sample_scene();

        Buffer queueState = create_host_buffer(sizeof(nrc::NrcQueueState), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer evalRecords = create_host_buffer(kEvalCapacity * sizeof(nrc::NrcEvalRecord),
                                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer trainRecords = create_host_buffer(kTrainRecordCount * sizeof(nrc::NrcTrainRecord),
                                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer sceneBuffer = create_host_buffer(static_cast<VkDeviceSize>(sceneTriangles.size() * sizeof(SampleSceneTriangle)),
                                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer weightBuffer = create_host_buffer(static_cast<VkDeviceSize>(packedWeights.size() * sizeof(nrc::PackedFp16x8)),
                                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer biasUpload = create_host_buffer(static_cast<VkDeviceSize>(initialBiasFactorR.size() * sizeof(Vec4)),
                                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        Buffer factorUpload = create_host_buffer(static_cast<VkDeviceSize>(initialFactorGb.size() * sizeof(Vec2)),
                                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        Buffer biasReadback = create_host_buffer(static_cast<VkDeviceSize>(initialBiasFactorR.size() * sizeof(Vec4)),
                                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT);

        nrc::NrcQueueState initialQueueState{};
        std::vector<nrc::NrcEvalRecord> zeroEvalRecords(kEvalCapacity);
        std::vector<nrc::NrcTrainRecord> zeroTrainRecords(kTrainRecordCount);
        upload(queueState, &initialQueueState, sizeof(initialQueueState));
        upload(evalRecords, zeroEvalRecords.data(), static_cast<VkDeviceSize>(zeroEvalRecords.size() * sizeof(nrc::NrcEvalRecord)));
        upload(trainRecords,
               zeroTrainRecords.data(),
               static_cast<VkDeviceSize>(zeroTrainRecords.size() * sizeof(nrc::NrcTrainRecord)));
        upload(sceneBuffer,
               sceneTriangles.data(),
               static_cast<VkDeviceSize>(sceneTriangles.size() * sizeof(SampleSceneTriangle)));
        upload(weightBuffer, packedWeights.data(), static_cast<VkDeviceSize>(packedWeights.size() * sizeof(nrc::PackedFp16x8)));
        upload(biasUpload, initialBiasFactorR.data(), static_cast<VkDeviceSize>(initialBiasFactorR.size() * sizeof(Vec4)));
        upload(factorUpload, initialFactorGb.data(), static_cast<VkDeviceSize>(initialFactorGb.size() * sizeof(Vec2)));

        Image biasImage = create_image(VK_FORMAT_R32G32B32A32_SFLOAT);
        Image factorImage = create_image(VK_FORMAT_R32G32_SFLOAT);

        VkDescriptorPool descriptorPool = create_descriptor_pool();
        VkDescriptorSet producerSet = allocate_descriptor_set(descriptorPool, producerSetLayout_);
        VkDescriptorSet inferenceSet = allocate_descriptor_set(descriptorPool, inferenceSetLayout_);
        update_producer_set(producerSet, queueState, evalRecords, trainRecords, sceneBuffer);
        update_inference_set(inferenceSet, weightBuffer, evalRecords, queueState, biasImage, factorImage, trainRecords, sceneBuffer);

        submit(queueState,
               evalRecords,
               biasUpload,
               factorUpload,
               biasImage,
               factorImage,
               biasReadback,
               producerSet,
               inferenceSet);

        vkDestroyDescriptorPool(device_, descriptorPool, nullptr);

        nrc::NrcQueueState finalQueueState{};
        std::vector<nrc::NrcEvalRecord> finalEvalRecords(kEvalCapacity);
        std::vector<nrc::NrcTrainRecord> finalTrainRecords(kTrainRecordCount);
        std::vector<Vec4> finalBiasFactorR(kPixelCount);
        download(queueState, &finalQueueState, sizeof(finalQueueState));
        download(evalRecords,
                 finalEvalRecords.data(),
                 static_cast<VkDeviceSize>(finalEvalRecords.size() * sizeof(nrc::NrcEvalRecord)));
        download(trainRecords,
                 finalTrainRecords.data(),
                 static_cast<VkDeviceSize>(finalTrainRecords.size() * sizeof(nrc::NrcTrainRecord)));
        download(biasReadback,
                 finalBiasFactorR.data(),
                 static_cast<VkDeviceSize>(finalBiasFactorR.size() * sizeof(Vec4)));

        return verify(finalQueueState,
                      finalEvalRecords,
                      finalTrainRecords,
                      initialBiasFactorR,
                      initialFactorGb,
                      finalBiasFactorR);
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
        appInfo.pApplicationName = "nrc_reference_path_tracer_minimal";
        appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        appInfo.pEngineName = "NRC";
        appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
        appInfo.apiVersion = loaderVersion < VK_API_VERSION_1_1 ? loaderVersion : VK_API_VERSION_1_1;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        check_vk(vkCreateInstance(&createInfo, nullptr, &instance_), "vkCreateInstance");
    }

    void pick_physical_device() {
        std::uint32_t count = 0;
        check_vk(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "vkEnumeratePhysicalDevices");
        if (count == 0) {
            throw std::runtime_error("no Vulkan physical devices were found");
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

        throw std::runtime_error("no Vulkan 1.1 compute-capable device was found");
    }

    void check_format(VkFormat format, const char *name) const {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice_, format, &properties);
        const VkFormatFeatureFlags required =
            VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
            VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
            VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        if ((properties.optimalTilingFeatures & required) != required) {
            throw std::runtime_error(std::string("format does not support required usage: ") + name);
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

    void create_command_pool() {
        VkCommandPoolCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        createInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        createInfo.queueFamilyIndex = queueFamilyIndex_;
        check_vk(vkCreateCommandPool(device_, &createInfo, nullptr, &commandPool_), "vkCreateCommandPool");
    }

    void create_producer_pipeline(const char *producerSpirv) {
        std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
        for (std::uint32_t i = 0; i < bindings.size(); ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        producerSetLayout_ = create_set_layout(bindings);

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(ProducerConstants);
        producerPipelineLayout_ = create_pipeline_layout(producerSetLayout_, &pushRange);
        producerPipeline_ = create_compute_pipeline(producerSpirv, producerPipelineLayout_);
    }

    void create_inference_pipeline(const char *inferenceSpirv) {
        std::array<VkDescriptorSetLayoutBinding, 7> bindings{};
        for (std::uint32_t i = 0; i < bindings.size(); ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[i].descriptorType =
                (i == 3 || i == 4) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }

        inferenceSetLayout_ = create_set_layout(bindings);
        inferencePipelineLayout_ = create_pipeline_layout(inferenceSetLayout_, nullptr);
        inferencePipeline_ = create_compute_pipeline(inferenceSpirv, inferencePipelineLayout_);
    }

    template <std::size_t N>
    VkDescriptorSetLayout create_set_layout(const std::array<VkDescriptorSetLayoutBinding, N> &bindings) {
        VkDescriptorSetLayoutCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        createInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
        createInfo.pBindings = bindings.data();

        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        check_vk(vkCreateDescriptorSetLayout(device_, &createInfo, nullptr, &layout), "vkCreateDescriptorSetLayout");
        return layout;
    }

    VkPipelineLayout create_pipeline_layout(VkDescriptorSetLayout setLayout,
                                            const VkPushConstantRange *pushRange) {
        VkPipelineLayoutCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        createInfo.setLayoutCount = 1;
        createInfo.pSetLayouts = &setLayout;
        if (pushRange != nullptr) {
            createInfo.pushConstantRangeCount = 1;
            createInfo.pPushConstantRanges = pushRange;
        }

        VkPipelineLayout layout = VK_NULL_HANDLE;
        check_vk(vkCreatePipelineLayout(device_, &createInfo, nullptr, &layout), "vkCreatePipelineLayout");
        return layout;
    }

    VkPipeline create_compute_pipeline(const char *spirvPath, VkPipelineLayout layout) {
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
        pipelineInfo.layout = layout;

        VkPipeline pipeline = VK_NULL_HANDLE;
        const VkResult result =
            vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
        vkDestroyShaderModule(device_, shaderModule, nullptr);
        check_vk(result, "vkCreateComputePipelines");
        return pipeline;
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
        throw std::runtime_error("no compatible Vulkan memory type was found");
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
        imageInfo.extent = {kWidth, kHeight, 1};
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
        poolSizes[0].descriptorCount = 9;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        poolSizes[1].descriptorCount = 2;

        VkDescriptorPoolCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        createInfo.maxSets = 2;
        createInfo.poolSizeCount = static_cast<std::uint32_t>(poolSizes.size());
        createInfo.pPoolSizes = poolSizes.data();

        VkDescriptorPool pool = VK_NULL_HANDLE;
        check_vk(vkCreateDescriptorPool(device_, &createInfo, nullptr, &pool), "vkCreateDescriptorPool");
        return pool;
    }

    VkDescriptorSet allocate_descriptor_set(VkDescriptorPool descriptorPool, VkDescriptorSetLayout layout) const {
        VkDescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool = descriptorPool;
        allocateInfo.descriptorSetCount = 1;
        allocateInfo.pSetLayouts = &layout;

        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        check_vk(vkAllocateDescriptorSets(device_, &allocateInfo, &descriptorSet), "vkAllocateDescriptorSets");
        return descriptorSet;
    }

    void update_producer_set(VkDescriptorSet set,
                             const Buffer &queueState,
                             const Buffer &evalRecords,
                             const Buffer &trainRecords,
                             const Buffer &sceneBuffer) const {
        std::array<VkDescriptorBufferInfo, 4> infos{};
        infos[0] = {queueState.buffer, 0, queueState.size};
        infos[1] = {evalRecords.buffer, 0, evalRecords.size};
        infos[2] = {trainRecords.buffer, 0, trainRecords.size};
        infos[3] = {sceneBuffer.buffer, 0, sceneBuffer.size};

        std::array<VkWriteDescriptorSet, 4> writes{};
        for (std::uint32_t i = 0; i < writes.size(); ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    void update_inference_set(VkDescriptorSet set,
                              const Buffer &weights,
                              const Buffer &evalRecords,
                              const Buffer &queueState,
                              const Image &biasImage,
                              const Image &factorImage,
                              const Buffer &trainRecords,
                              const Buffer &sceneBuffer) const {
        std::array<VkDescriptorBufferInfo, 5> bufferInfos{};
        bufferInfos[0] = {weights.buffer, 0, weights.size};
        bufferInfos[1] = {evalRecords.buffer, 0, evalRecords.size};
        bufferInfos[2] = {queueState.buffer, 0, queueState.size};
        bufferInfos[3] = {trainRecords.buffer, 0, trainRecords.size};
        bufferInfos[4] = {sceneBuffer.buffer, 0, sceneBuffer.size};

        std::array<VkDescriptorImageInfo, 2> imageInfos{};
        imageInfos[0].imageView = biasImage.view;
        imageInfos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageInfos[1].imageView = factorImage.view;
        imageInfos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 7> writes{};
        for (auto &write : writes) {
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = set;
            write.descriptorCount = 1;
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
        writes[6].dstBinding = 6;
        writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[6].pBufferInfo = &bufferInfos[4];

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
        region.imageExtent = {kWidth, kHeight, 1};
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
        region.imageExtent = {kWidth, kHeight, 1};
        vkCmdCopyImageToBuffer(commandBuffer,
                               image.image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               buffer.buffer,
                               1,
                               &region);
    }

    void submit(const Buffer &queueState,
                const Buffer &evalRecords,
                const Buffer &biasUpload,
                const Buffer &factorUpload,
                const Image &biasImage,
                const Image &factorImage,
                const Buffer &biasReadback,
                VkDescriptorSet producerSet,
                VkDescriptorSet inferenceSet) const {
        VkCommandBuffer commandBuffer = allocate_command_buffer();
        VkFence fence = VK_NULL_HANDLE;

        try {
            VkFenceCreateInfo fenceInfo{};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            check_vk(vkCreateFence(device_, &fenceInfo, nullptr, &fence), "vkCreateFence");

            record_commands(commandBuffer,
                            queueState,
                            evalRecords,
                            biasUpload,
                            factorUpload,
                            biasImage,
                            factorImage,
                            biasReadback,
                            producerSet,
                            inferenceSet);

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
            throw;
        }

        vkDestroyFence(device_, fence, nullptr);
        vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer);
    }

    void record_commands(VkCommandBuffer commandBuffer,
                         const Buffer &queueState,
                         const Buffer &evalRecords,
                         const Buffer &biasUpload,
                         const Buffer &factorUpload,
                         const Image &biasImage,
                         const Image &factorImage,
                         const Buffer &biasReadback,
                         VkDescriptorSet producerSet,
                         VkDescriptorSet inferenceSet) const {
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

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, producerPipeline_);
        vkCmdBindDescriptorSets(commandBuffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                producerPipelineLayout_,
                                0,
                                1,
                                &producerSet,
                                0,
                                nullptr);
        dispatch_producer(commandBuffer, {0u, kEvalCapacity, kPixelCount, kWidth, kHeight, kTrainRecordCount});

        VkMemoryBarrier resetToProducer{};
        resetToProducer.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        resetToProducer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        resetToProducer.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                             1,
                             &resetToProducer,
                             0,
                             nullptr,
                             0,
                             nullptr);

        dispatch_producer(commandBuffer, {1u, kEvalCapacity, kPixelCount, kWidth, kHeight, kTrainRecordCount});

        VkMemoryBarrier producerToInference{};
        producerToInference.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        producerToInference.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        producerToInference.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                             1,
                             &producerToInference,
                             0,
                             nullptr,
                             0,
                             nullptr);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, inferencePipeline_);
        vkCmdBindDescriptorSets(commandBuffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                inferencePipelineLayout_,
                                0,
                                1,
                                &inferenceSet,
                                0,
                                nullptr);
        vkCmdDispatch(commandBuffer, (kPixelCount + kWorkgroupSize - 1u) / kWorkgroupSize, 1, 1);

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

    void dispatch_producer(VkCommandBuffer commandBuffer, const ProducerConstants &constants) const {
        vkCmdPushConstants(commandBuffer,
                           producerPipelineLayout_,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           sizeof(constants),
                           &constants);
        const std::uint32_t dispatchCount =
            (std::max(constants.pixelCount, nrc::kTrainBatchCount) + kWorkgroupSize - 1u) / kWorkgroupSize;
        vkCmdDispatch(commandBuffer, dispatchCount, 1, 1);
    }

    bool verify(const nrc::NrcQueueState &queueState,
                const std::vector<nrc::NrcEvalRecord> &evalRecords,
                const std::vector<nrc::NrcTrainRecord> &trainRecords,
                const std::vector<Vec4> &initialBiasFactorR,
                const std::vector<Vec2> &initialFactorGb,
                const std::vector<Vec4> &finalBiasFactorR) const {
        bool ok = true;

        if (queueState.evalCount != kEvalCapacity) {
            std::cerr << "FAIL: expected " << kEvalCapacity
                      << " eval records, got " << queueState.evalCount << '\n';
            ok = false;
        }
        if (queueState.trainCounts[0] != kTrainRecordCount) {
            std::cerr << "FAIL: train batch 0 expected " << kTrainRecordCount
                      << " records, got " << queueState.trainCounts[0] << '\n';
            ok = false;
        }
        for (std::uint32_t batch = 1; batch < nrc::kTrainBatchCount; ++batch) {
            if (queueState.trainCounts[batch] != 0u) {
                std::cerr << "FAIL: train batch " << batch
                          << " expected 0 records, got " << queueState.trainCounts[batch] << '\n';
                ok = false;
            }
        }

        std::array<bool, kPixelCount> seen{};
        bool sawTrainEval = false;
        for (std::uint32_t i = 0; i < kEvalCapacity; ++i) {
            if ((evalRecords[i].dst & 1u) == 0u) {
                const std::uint32_t x = decode_screen_x(evalRecords[i].dst);
                const std::uint32_t y = decode_screen_y(evalRecords[i].dst);
                if (x >= kWidth || y >= kHeight) {
                    std::cerr << "FAIL: eval record " << i << " has invalid screen dst "
                              << evalRecords[i].dst << '\n';
                    ok = false;
                    continue;
                }
                const std::uint32_t pixelIndex = y * kWidth + x;
                if (seen[pixelIndex]) {
                    std::cerr << "FAIL: duplicate eval record for pixel " << pixelIndex << '\n';
                    ok = false;
                }
                const nrc::PackedNrcInput expectedInput = expected_packed_input(x, y);
                if (!same_packed_input(evalRecords[i].packedInput, expectedInput)) {
                    report_packed_input_mismatch("screen eval record",
                                                 expectedInput,
                                                 evalRecords[i].packedInput);
                    ok = false;
                }
                seen[pixelIndex] = true;
            } else {
                const std::uint32_t batch = decode_train_batch(evalRecords[i].dst);
                const std::uint32_t left = decode_train_left(evalRecords[i].dst);
                const std::uint32_t right = decode_train_right(evalRecords[i].dst);
                if (batch != 0u || left != 0u || right != kTrainRecordCount - 1u || sawTrainEval) {
                    std::cerr << "FAIL: eval record " << i << " has invalid train dst "
                              << evalRecords[i].dst << '\n';
                    ok = false;
                }
                const nrc::PackedNrcInput expectedInput = expected_packed_input(0u, 0u);
                if (!same_packed_input(evalRecords[i].packedInput, expectedInput)) {
                    report_packed_input_mismatch("train eval record",
                                                 expectedInput,
                                                 evalRecords[i].packedInput);
                    ok = false;
                }
                sawTrainEval = true;
            }
        }

        for (std::uint32_t i = 0; i < kPixelCount; ++i) {
            if (!seen[i]) {
                std::cerr << "FAIL: missing eval record for pixel " << i << '\n';
                ok = false;
            }
        }
        if (!sawTrainEval) {
            std::cerr << "FAIL: missing train eval record\n";
            ok = false;
        }

        for (std::uint32_t i = 0; i < kPixelCount; ++i) {
            const Vec4 expected{
                initialBiasFactorR[i].x + initialBiasFactorR[i].w * 0.25f,
                initialBiasFactorR[i].y + initialFactorGb[i].x * 0.5f,
                initialBiasFactorR[i].z + initialFactorGb[i].y * 0.75f,
                0.0f,
            };
            const Vec4 &actual = finalBiasFactorR[i];
            if (!near(actual.x, expected.x) ||
                !near(actual.y, expected.y) ||
                !near(actual.z, expected.z) ||
                !near(actual.w, expected.w)) {
                std::cerr << "FAIL: pixel " << i
                          << " expected (" << expected.x << ", " << expected.y << ", "
                          << expected.z << ", " << expected.w << ") got ("
                          << actual.x << ", " << actual.y << ", "
                          << actual.z << ", " << actual.w << ")\n";
                ok = false;
            }
        }

        for (std::uint32_t i = 0; i < kTrainRecordCount; ++i) {
            const nrc::PackedNrcInput expectedInput = expected_packed_input(0u, 0u);
            const ExpectedTrainPayload expectedPayload = expected_train_payload(expectedInput);
            const float expectedBiasR = expectedPayload.bias.x + expectedPayload.factor.x * 0.25f;
            const float expectedBiasG = expectedPayload.bias.y + expectedPayload.factor.y * 0.5f;
            const float expectedBiasB = expectedPayload.bias.z + expectedPayload.factor.z * 0.75f;
            if (!near(trainRecords[i].biasR, expectedBiasR) ||
                !near(trainRecords[i].biasG, expectedBiasG) ||
                !near(trainRecords[i].biasB, expectedBiasB) ||
                !near(trainRecords[i].factorR, expectedPayload.factor.x) ||
                !near(trainRecords[i].factorG, expectedPayload.factor.y) ||
                !near(trainRecords[i].factorB, expectedPayload.factor.z)) {
                std::cerr << "FAIL: train record " << i
                          << " expected bias (" << expectedBiasR << ", " << expectedBiasG << ", "
                          << expectedBiasB << ") factor ("
                          << expectedPayload.factor.x << ", "
                          << expectedPayload.factor.y << ", "
                          << expectedPayload.factor.z << ") got bias ("
                          << trainRecords[i].biasR << ", " << trainRecords[i].biasG << ", "
                          << trainRecords[i].biasB << ") factor ("
                          << trainRecords[i].factorR << ", " << trainRecords[i].factorG << ", "
                          << trainRecords[i].factorB << ")\n";
                ok = false;
            }
            if (!same_packed_input(trainRecords[i].packedInput, expectedInput)) {
                report_packed_input_mismatch("train record",
                                             expectedInput,
                                             trainRecords[i].packedInput);
                ok = false;
            }
        }

        return ok;
    }

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memoryProperties_{};
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::uint32_t queueFamilyIndex_ = 0;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout producerSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout inferenceSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout producerPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout inferencePipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline producerPipeline_ = VK_NULL_HANDLE;
    VkPipeline inferencePipeline_ = VK_NULL_HANDLE;
};

} // namespace

int main() {
    try {
        MinimalIntegrationSample sample(NRC_SAMPLE_PRODUCER_SPV, NRC_SAMPLE_INFERENCE_SPV);
        if (!sample.run()) {
            return EXIT_FAILURE;
        }

        std::cout << "NRC reference path tracer minimal integration passed: "
                  << kEvalCapacity << " eval records, scalar inference screen/train resolve." << std::endl;
        std::cout << "This milestone uses primary-ray traced hits and direct-lighting train payloads; full ray tracing is pending." << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "NRC reference path tracer sample failed: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
