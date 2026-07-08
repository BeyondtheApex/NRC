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

} // namespace nrc
