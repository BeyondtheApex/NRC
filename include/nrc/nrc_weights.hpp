#pragma once

#include "nrc/nrc_constants.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace nrc {

using PackedFp16x8 = std::array<std::uint32_t, 4>;

std::uint16_t float_to_half_bits(float value);
float half_bits_to_float(std::uint16_t bits);

std::vector<float> make_kaiming_weights(std::uint32_t seed);
std::vector<PackedFp16x8> pack_weights_fp16_uvec4(std::span<const float> weights);
std::vector<float> unpack_weights_fp16_uvec4(std::span<const PackedFp16x8> packed,
                                             std::uint32_t valueCount);

} // namespace nrc
