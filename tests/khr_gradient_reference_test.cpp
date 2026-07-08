#include "nrc/nrc_khr_reference.hpp"
#include "nrc/nrc_weights.hpp"

#include <algorithm>
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

std::vector<float> make_positive_test_weights() {
    std::vector<float> weights(nrc::weight_count(), 0.0f);
    for (std::uint32_t layer = 0; layer < nrc::kNnHiddenLayerCount; ++layer) {
        const std::size_t offset = nrc::layer_weight_offset(layer);
        for (std::uint32_t row = 0; row < nrc::kNnHiddenWidth; ++row) {
            for (std::uint32_t col = 0; col < nrc::kNnHiddenWidth; ++col) {
                const float diagonal = row == col ? 0.48f : 0.0f;
                const float dense = 0.00035f * static_cast<float>((row + col + layer) % 5u + 1u);
                weights[offset + row * nrc::kNnHiddenWidth + col] = diagonal + dense;
            }
        }
    }

    const std::size_t outputOffset = nrc::layer_weight_offset(nrc::kNnHiddenLayerCount);
    for (std::uint32_t row = 0; row < nrc::kNnOutputWidth; ++row) {
        for (std::uint32_t col = 0; col < nrc::kNnHiddenWidth; ++col) {
            weights[outputOffset + row * nrc::kNnHiddenWidth + col] =
                0.0025f * static_cast<float>(row + 1u) +
                0.0001f * static_cast<float>((col % 7u) + 1u);
        }
    }
    return weights;
}

nrc::ReferenceInput make_input() {
    nrc::ReferenceInput input{};
    for (std::uint32_t i = 0; i < nrc::kEncodedInputWidth; ++i) {
        input[i] = 0.04f + 0.0025f * static_cast<float>((i % 11u) + 1u);
    }
    return input;
}

float max_abs(std::span<const float> values) {
    float result = 0.0f;
    for (float value : values) {
        result = std::max(result, std::fabs(value));
    }
    return result;
}

bool all_finite(std::span<const float> values) {
    return std::all_of(values.begin(), values.end(), [](float value) {
        return std::isfinite(value);
    });
}

} // namespace

int main() {
    bool ok = true;

    const std::vector<float> weights = make_positive_test_weights();
    const std::vector<nrc::PackedFp16x8> packedWeights = nrc::pack_weights_fp16_uvec4(weights);
    const std::vector<float> gpuVisibleWeights =
        nrc::unpack_weights_fp16_uvec4(packedWeights, nrc::weight_count());
    const nrc::ReferenceInput input = make_input();
    const nrc::ReferenceOutput target{0.0f, 0.0f, 0.0f};

    std::vector<float> khrGradients(nrc::weight_count(), -1.0f);
    nrc::ReferenceOutput prediction{};
    nrc::reference_backward_khr_fp16(gpuVisibleWeights,
                                     input,
                                     target,
                                     khrGradients,
                                     &prediction);

    ok &= require(all_finite(khrGradients), "KHR reference gradients are finite");
    ok &= require(all_finite(prediction), "KHR reference prediction is finite");
    ok &= require(max_abs(khrGradients) > 0.0f, "KHR reference produces non-zero gradients");

    std::vector<float> scalarGradients(nrc::weight_count(), 0.0f);
    nrc::reference_backward(gpuVisibleWeights,
                            input,
                            target,
                            scalarGradients);
    ok &= require(max_abs(scalarGradients) > 0.0f, "scalar reference produces non-zero gradients");

    for (std::uint32_t layer = 0; layer < nrc::kNnHiddenLayerCount; ++layer) {
        const std::size_t offset = nrc::layer_weight_offset(layer);
        const std::span<const float> khrLayer{khrGradients.data() + offset,
                                              nrc::hidden_layer_weight_count()};
        const std::span<const float> scalarLayer{scalarGradients.data() + offset,
                                                 nrc::hidden_layer_weight_count()};
        if (max_abs(khrLayer) == 0.0f) {
            std::cerr << "FAIL: KHR reference hidden layer " << layer
                      << " gradients are all zero\n";
            ok = false;
        }
        if (max_abs(scalarLayer) == 0.0f) {
            std::cerr << "FAIL: scalar reference hidden layer " << layer
                      << " gradients are all zero\n";
            ok = false;
        }
    }

    const std::size_t outputOffset = nrc::layer_weight_offset(nrc::kNnHiddenLayerCount);
    ok &= require(max_abs(std::span<const float>{khrGradients.data() + outputOffset,
                                                  nrc::output_layer_weight_count()}) > 0.0f,
                  "KHR reference output gradients are non-zero");
    ok &= require(max_abs(std::span<const float>{scalarGradients.data() + outputOffset,
                                                  nrc::output_layer_weight_count()}) > 0.0f,
                  "scalar reference output gradients are non-zero");

    std::vector<float> accumulated(nrc::weight_count(), 1.0f);
    nrc::reference_backward_khr_fp16(gpuVisibleWeights,
                                     input,
                                     target,
                                     accumulated,
                                     nullptr,
                                     nrc::kLossScale,
                                     true);
    ok &= require(accumulated[0] != khrGradients[0],
                  "KHR reference accumulate mode preserves existing gradient contents");

    bool threw = false;
    try {
        nrc::reference_backward_khr_fp16(std::span<const float>{gpuVisibleWeights.data(),
                                                                gpuVisibleWeights.size() - 1u},
                                         input,
                                         target,
                                         khrGradients);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    ok &= require(threw, "invalid KHR reference weight count throws");

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
