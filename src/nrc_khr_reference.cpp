#include "nrc/nrc_khr_reference.hpp"

#include "nrc/nrc_weights.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace nrc {
namespace {

using HiddenVector = std::array<float, kNnHiddenWidth>;
using HiddenLayers = std::array<HiddenVector, kNnHiddenLayerCount>;

float fp16_round(float value) {
    return half_bits_to_float(float_to_half_bits(value));
}

HiddenLayers forward_hidden_layers_fp16(std::span<const float> weights,
                                        const ReferenceInput &input) {
    HiddenLayers hiddenLayers{};
    HiddenVector src{};
    std::copy(input.begin(), input.end(), src.begin());

    for (std::uint32_t layer = 0; layer < kNnHiddenLayerCount; ++layer) {
        const std::size_t offset = layer_weight_offset(layer);
        for (std::uint32_t row = 0; row < kNnHiddenWidth; ++row) {
            float value = 0.0f;
            for (std::uint32_t col = 0; col < kNnHiddenWidth; ++col) {
                value += weights[offset + row * kNnHiddenWidth + col] * src[col];
            }
            hiddenLayers[layer][row] = fp16_round(std::max(value, 0.0f));
        }
        src = hiddenLayers[layer];
    }
    return hiddenLayers;
}

} // namespace

void reference_backward_khr_fp16(std::span<const float> weights,
                                 const ReferenceInput &input,
                                 const ReferenceOutput &target,
                                 std::span<float> gradients,
                                 ReferenceOutput *prediction,
                                 float lossScale,
                                 bool accumulate) {
    if (weights.size() != weight_count()) {
        throw std::invalid_argument("NRC KHR reference_backward received an invalid weight count");
    }
    if (gradients.size() != weight_count()) {
        throw std::invalid_argument("NRC KHR reference_backward received an invalid gradient count");
    }
    if (!accumulate) {
        std::fill(gradients.begin(), gradients.end(), 0.0f);
    }

    HiddenVector inputActivation{};
    std::copy(input.begin(), input.end(), inputActivation.begin());
    const HiddenLayers hiddenLayers = forward_hidden_layers_fp16(weights, input);
    const HiddenVector &finalHidden = hiddenLayers[kNnHiddenLayerCount - 1u];

    ReferenceOutput output{};
    const std::size_t outputOffset = layer_weight_offset(kNnHiddenLayerCount);
    for (std::uint32_t row = 0; row < kNnOutputWidth; ++row) {
        float value = 0.0f;
        for (std::uint32_t col = 0; col < kNnHiddenWidth; ++col) {
            value += weights[outputOffset + row * kNnHiddenWidth + col] * finalHidden[col];
        }
        output[row] = fp16_round(value);
    }
    if (prediction != nullptr) {
        *prediction = output;
    }

    const float luminance =
        0.299f * std::max(output[0], 0.0f) +
        0.587f * std::max(output[1], 0.0f) +
        0.114f * std::max(output[2], 0.0f);
    const float denominator = luminance * luminance + 0.01f;

    ReferenceOutput dOutput{};
    for (std::uint32_t row = 0; row < kNnOutputWidth; ++row) {
        dOutput[row] = 2.0f * lossScale * (output[row] - target[row]) / denominator;
    }

    for (std::uint32_t row = 0; row < kNnOutputWidth; ++row) {
        for (std::uint32_t col = 0; col < kNnHiddenWidth; ++col) {
            gradients[outputOffset + row * kNnHiddenWidth + col] +=
                dOutput[row] * finalHidden[col];
        }
    }

    HiddenVector dActivation{};
    for (std::uint32_t col = 0; col < kNnHiddenWidth; ++col) {
        for (std::uint32_t row = 0; row < kNnOutputWidth; ++row) {
            dActivation[col] += weights[outputOffset + row * kNnHiddenWidth + col] *
                                dOutput[row];
        }
    }

    for (std::uint32_t reverseLayer = 0; reverseLayer < kNnHiddenLayerCount; ++reverseLayer) {
        const std::uint32_t layer = kNnHiddenLayerCount - 1u - reverseLayer;
        const std::size_t offset = layer_weight_offset(layer);
        const HiddenVector &activation = hiddenLayers[layer];
        const HiddenVector &sourceActivation = layer == 0u ? inputActivation : hiddenLayers[layer - 1u];

        for (std::uint32_t row = 0; row < kNnHiddenWidth; ++row) {
            const float dPreActivation = activation[row] > 0.0f ? dActivation[row] : 0.0f;
            for (std::uint32_t col = 0; col < kNnHiddenWidth; ++col) {
                gradients[offset + row * kNnHiddenWidth + col] +=
                    dPreActivation * sourceActivation[col];
            }
        }

        if (layer != 0u) {
            HiddenVector dInput{};
            for (std::uint32_t col = 0; col < kNnHiddenWidth; ++col) {
                for (std::uint32_t row = 0; row < kNnHiddenWidth; ++row) {
                    const float dPreActivation = activation[row] > 0.0f ? dActivation[row] : 0.0f;
                    dInput[col] += weights[offset + row * kNnHiddenWidth + col] *
                                   dPreActivation;
                }
            }
            dActivation = dInput;
        }
    }
}

} // namespace nrc
