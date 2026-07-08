#pragma once

#include "nrc/nrc_constants.hpp"

#include <array>
#include <cstddef>
#include <span>

namespace nrc {

using ReferenceInput = std::array<float, kEncodedInputWidth>;
using ReferenceOutput = std::array<float, kNnOutputWidth>;

std::size_t layer_weight_offset(std::uint32_t layer);
void reference_forward(std::span<const float> weights,
                       const ReferenceInput &input,
                       ReferenceOutput &output);

} // namespace nrc
