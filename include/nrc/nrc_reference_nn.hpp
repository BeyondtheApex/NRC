#pragma once

#include "nrc/nrc_constants.hpp"

#include <array>
#include <cstddef>
#include <span>

namespace nrc {

using ReferenceInput = std::array<float, kEncodedInputWidth>;
using ReferenceOutput = std::array<float, kNnOutputWidth>;

enum class ReferenceLoss {
    L2,
    RelativeL2Luminance,
};

std::size_t layer_weight_offset(std::uint32_t layer);
void reference_forward(std::span<const float> weights,
                       const ReferenceInput &input,
                       ReferenceOutput &output);
void reference_backward(std::span<const float> weights,
                        const ReferenceInput &input,
                        const ReferenceOutput &target,
                        std::span<float> gradients,
                        ReferenceOutput *prediction = nullptr,
                        ReferenceLoss loss = ReferenceLoss::RelativeL2Luminance,
                        float lossScale = kLossScale,
                        bool accumulate = false);

} // namespace nrc
