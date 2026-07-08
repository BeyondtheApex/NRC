#pragma once

#include "nrc/nrc_reference_nn.hpp"

#include <span>

namespace nrc {

void reference_backward_khr_fp16(std::span<const float> weights,
                                 const ReferenceInput &input,
                                 const ReferenceOutput &target,
                                 std::span<float> gradients,
                                 ReferenceOutput *prediction = nullptr,
                                 float lossScale = kLossScale,
                                 bool accumulate = false);

} // namespace nrc
