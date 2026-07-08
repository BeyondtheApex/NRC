#include "nrc/nrc_reference_nn.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace nrc {

std::size_t layer_weight_offset(std::uint32_t layer) {
    if (layer > kNnHiddenLayerCount) {
        throw std::out_of_range("NRC NN layer out of range");
    }
    return static_cast<std::size_t>(layer) * hidden_layer_weight_count();
}

void reference_forward(std::span<const float> weights,
                       const ReferenceInput &input,
                       ReferenceOutput &output) {
    if (weights.size() != weight_count()) {
        throw std::invalid_argument("NRC reference_forward received an invalid weight count");
    }

    std::array<float, kNnHiddenWidth> src{};
    std::array<float, kNnHiddenWidth> dst{};
    std::copy(input.begin(), input.end(), src.begin());

    for (std::uint32_t layer = 0; layer < kNnHiddenLayerCount; ++layer) {
        const std::size_t offset = layer_weight_offset(layer);
        for (std::uint32_t row = 0; row < kNnHiddenWidth; ++row) {
            float value = 0.0f;
            for (std::uint32_t col = 0; col < kNnHiddenWidth; ++col) {
                value += weights[offset + row * kNnHiddenWidth + col] * src[col];
            }
            dst[row] = std::max(value, 0.0f);
        }
        src = dst;
    }

    const std::size_t outputOffset = layer_weight_offset(kNnHiddenLayerCount);
    for (std::uint32_t row = 0; row < kNnOutputWidth; ++row) {
        float value = 0.0f;
        for (std::uint32_t col = 0; col < kNnHiddenWidth; ++col) {
            value += weights[outputOffset + row * kNnHiddenWidth + col] * src[col];
        }
        output[row] = value;
    }
}

void reference_backward(std::span<const float> weights,
                        const ReferenceInput &input,
                        const ReferenceOutput &target,
                        std::span<float> gradients,
                        ReferenceOutput *prediction,
                        ReferenceLoss loss,
                        float lossScale,
                        bool accumulate) {
    if (weights.size() != weight_count()) {
        throw std::invalid_argument("NRC reference_backward received an invalid weight count");
    }
    if (gradients.size() != weight_count()) {
        throw std::invalid_argument("NRC reference_backward received an invalid gradient count");
    }
    if (!accumulate) {
        std::fill(gradients.begin(), gradients.end(), 0.0f);
    }

    std::array<std::array<float, kNnHiddenWidth>, kNnHiddenLayerCount + 1> activations{};
    std::array<std::array<float, kNnHiddenWidth>, kNnHiddenLayerCount> preActivations{};
    std::copy(input.begin(), input.end(), activations[0].begin());

    for (std::uint32_t layer = 0; layer < kNnHiddenLayerCount; ++layer) {
        const std::size_t offset = layer_weight_offset(layer);
        for (std::uint32_t row = 0; row < kNnHiddenWidth; ++row) {
            float value = 0.0f;
            for (std::uint32_t col = 0; col < kNnHiddenWidth; ++col) {
                value += weights[offset + row * kNnHiddenWidth + col] *
                         activations[layer][col];
            }
            preActivations[layer][row] = value;
            activations[layer + 1][row] = std::max(value, 0.0f);
        }
    }

    ReferenceOutput output{};
    const std::size_t outputOffset = layer_weight_offset(kNnHiddenLayerCount);
    for (std::uint32_t row = 0; row < kNnOutputWidth; ++row) {
        float value = 0.0f;
        for (std::uint32_t col = 0; col < kNnHiddenWidth; ++col) {
            value += weights[outputOffset + row * kNnHiddenWidth + col] *
                     activations[kNnHiddenLayerCount][col];
        }
        output[row] = value;
    }
    if (prediction != nullptr) {
        *prediction = output;
    }

    ReferenceOutput dOutput{};
    float lossDenominator = 1.0f;
    if (loss == ReferenceLoss::RelativeL2Luminance) {
        const float luminance =
            0.299f * std::max(output[0], 0.0f) +
            0.587f * std::max(output[1], 0.0f) +
            0.114f * std::max(output[2], 0.0f);
        lossDenominator = luminance * luminance + 0.01f;
    }
    for (std::uint32_t channel = 0; channel < kNnOutputWidth; ++channel) {
        dOutput[channel] = 2.0f * lossScale * (output[channel] - target[channel]) /
                           lossDenominator;
    }

    for (std::uint32_t row = 0; row < kNnOutputWidth; ++row) {
        for (std::uint32_t col = 0; col < kNnHiddenWidth; ++col) {
            gradients[outputOffset + row * kNnHiddenWidth + col] +=
                dOutput[row] * activations[kNnHiddenLayerCount][col];
        }
    }

    std::array<float, kNnHiddenWidth> dActivation{};
    for (std::uint32_t col = 0; col < kNnHiddenWidth; ++col) {
        float value = 0.0f;
        for (std::uint32_t row = 0; row < kNnOutputWidth; ++row) {
            value += weights[outputOffset + row * kNnHiddenWidth + col] *
                     dOutput[row];
        }
        dActivation[col] = value;
    }

    for (std::uint32_t reverseLayer = 0; reverseLayer < kNnHiddenLayerCount; ++reverseLayer) {
        const std::uint32_t layer = kNnHiddenLayerCount - 1u - reverseLayer;
        const std::size_t offset = layer_weight_offset(layer);
        std::array<float, kNnHiddenWidth> dInput{};

        for (std::uint32_t row = 0; row < kNnHiddenWidth; ++row) {
            const float dPreActivation =
                preActivations[layer][row] > 0.0f ? dActivation[row] : 0.0f;
            for (std::uint32_t col = 0; col < kNnHiddenWidth; ++col) {
                gradients[offset + row * kNnHiddenWidth + col] +=
                    dPreActivation * activations[layer][col];
                dInput[col] += weights[offset + row * kNnHiddenWidth + col] *
                               dPreActivation;
            }
        }

        dActivation = dInput;
    }
}

} // namespace nrc
