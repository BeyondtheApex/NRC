#pragma once

#include <cstddef>
#include <cstdint>

namespace nrc {

inline constexpr std::uint32_t kEncodedInputWidth = 64;
inline constexpr std::uint32_t kEncodedInputUvec4Count = 8;
inline constexpr std::uint32_t kFp16PerUvec4 = 8;

inline constexpr std::uint32_t kNnHiddenLayerCount = 5;
inline constexpr std::uint32_t kNnHiddenWidth = 64;
inline constexpr std::uint32_t kNnOutputWidth = 3;

inline constexpr std::uint32_t kTrainBatchSize = 16u * 1024u;
inline constexpr std::uint32_t kTrainBatchCount = 4;

inline constexpr std::uint32_t kDefaultWorkgroupSize = 128;
inline constexpr float kDefaultTrainProbability = 0.03f;
inline constexpr double kPi = 3.14159265358979323846;
inline constexpr float kLossScale = 1.0f;
inline constexpr float kAdamBeta1 = 0.9f;
inline constexpr float kAdamBeta2 = 0.999f;
inline constexpr float kEmaAlpha = 0.99f;
inline constexpr float kOptimizerLearningRate = 0.002f;
inline constexpr float kOptimizerEpsilon = 1.0e-8f;

inline constexpr std::uint32_t weight_count() {
    return kNnHiddenWidth * kNnHiddenWidth * kNnHiddenLayerCount +
           kNnHiddenWidth * kNnOutputWidth;
}

inline constexpr std::uint32_t hidden_layer_weight_count() {
    return kNnHiddenWidth * kNnHiddenWidth;
}

inline constexpr std::uint32_t output_layer_weight_count() {
    return kNnHiddenWidth * kNnOutputWidth;
}

inline constexpr std::uint32_t weight_uvec4_count() {
    return weight_count() / kFp16PerUvec4;
}

} // namespace nrc
