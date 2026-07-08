#include "nrc/nrc_reference_nn.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

bool require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }
    return true;
}

bool near(float lhs, float rhs, float epsilon) {
    return std::fabs(lhs - rhs) <= epsilon;
}

std::vector<float> make_active_test_weights() {
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

nrc::ReferenceInput make_input() {
    nrc::ReferenceInput input{};
    input[0] = 0.9f;
    input[1] = 0.7f;
    input[2] = 0.5f;
    input[3] = 0.25f;
    return input;
}

float detached_relative_loss(std::span<const float> weights,
                             const nrc::ReferenceInput &input,
                             const nrc::ReferenceOutput &target,
                             float fixedDenominator) {
    nrc::ReferenceOutput output{};
    nrc::reference_forward(weights, input, output);

    float loss = 0.0f;
    for (std::uint32_t i = 0; i < nrc::kNnOutputWidth; ++i) {
        const float delta = output[i] - target[i];
        loss += delta * delta;
    }
    return nrc::kLossScale * loss / fixedDenominator;
}

float finite_difference(std::vector<float> weights,
                        const nrc::ReferenceInput &input,
                        const nrc::ReferenceOutput &target,
                        float fixedDenominator,
                        std::size_t weightIndex) {
    constexpr float kEpsilon = 1.0e-3f;
    weights[weightIndex] += kEpsilon;
    const float plus = detached_relative_loss(weights, input, target, fixedDenominator);
    weights[weightIndex] -= 2.0f * kEpsilon;
    const float minus = detached_relative_loss(weights, input, target, fixedDenominator);
    return (plus - minus) / (2.0f * kEpsilon);
}

} // namespace

int main() {
    bool ok = true;

    const nrc::ReferenceInput input = make_input();
    const nrc::ReferenceOutput target{0.15f, 0.12f, 0.10f};
    const std::vector<float> weights = make_active_test_weights();

    std::vector<float> gradients(nrc::weight_count(), 123.0f);
    nrc::ReferenceOutput prediction{};
    nrc::reference_backward(weights, input, target, gradients, &prediction);

    const float luminance =
        0.299f * std::max(prediction[0], 0.0f) +
        0.587f * std::max(prediction[1], 0.0f) +
        0.114f * std::max(prediction[2], 0.0f);
    const float fixedDenominator = luminance * luminance + 0.01f;

    const std::array<std::size_t, 5> checkedWeights = {
        nrc::layer_weight_offset(0) + 0 * nrc::kNnHiddenWidth + 0,
        nrc::layer_weight_offset(1) + 0 * nrc::kNnHiddenWidth + 0,
        nrc::layer_weight_offset(4) + 2 * nrc::kNnHiddenWidth + 2,
        nrc::layer_weight_offset(nrc::kNnHiddenLayerCount) + 0 * nrc::kNnHiddenWidth + 0,
        nrc::layer_weight_offset(nrc::kNnHiddenLayerCount) + 2 * nrc::kNnHiddenWidth + 2,
    };

    for (std::size_t weightIndex : checkedWeights) {
        const float expected = finite_difference(weights, input, target, fixedDenominator, weightIndex);
        if (!near(gradients[weightIndex], expected, 1.0e-3f)) {
            std::cerr << "FAIL: gradient " << weightIndex
                      << " expected " << expected
                      << " got " << gradients[weightIndex] << '\n';
            ok = false;
        }
    }

    std::vector<float> accumulated(nrc::weight_count(), 1.0f);
    nrc::reference_backward(weights,
                            input,
                            target,
                            accumulated,
                            nullptr,
                            nrc::ReferenceLoss::L2,
                            nrc::kLossScale,
                            true);
    ok &= require(accumulated[checkedWeights[0]] != gradients[checkedWeights[0]],
                  "accumulate mode preserves existing gradient contents");

    bool threw = false;
    try {
        nrc::reference_backward(std::span<const float>{weights.data(), weights.size() - 1},
                                input,
                                target,
                                gradients);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    ok &= require(threw, "invalid backward weight count throws");

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
