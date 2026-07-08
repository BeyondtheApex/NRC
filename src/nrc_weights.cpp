#include "nrc/nrc_weights.hpp"

#include <bit>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace nrc {

std::uint16_t float_to_half_bits(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = (bits >> 16u) & 0x8000u;
    std::int32_t exponent = static_cast<std::int32_t>((bits >> 23u) & 0xffu) - 127 + 15;
    std::uint32_t mantissa = bits & 0x007fffffu;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        mantissa |= 0x00800000u;
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - exponent);
        std::uint32_t rounded = mantissa >> shift;
        const std::uint32_t roundBit = (mantissa >> (shift - 1u)) & 1u;
        rounded += roundBit;
        return static_cast<std::uint16_t>(sign | rounded);
    }

    if (exponent >= 31) {
        if (mantissa == 0) {
            return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
        return static_cast<std::uint16_t>(sign | 0x7c00u | (mantissa >> 13u) | 1u);
    }

    mantissa += 0x00001000u;
    if (mantissa & 0x00800000u) {
        mantissa = 0;
        ++exponent;
        if (exponent >= 31) {
            return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
    }

    return static_cast<std::uint16_t>(sign |
                                      (static_cast<std::uint32_t>(exponent) << 10u) |
                                      (mantissa >> 13u));
}

float half_bits_to_float(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16u;
    std::uint32_t exponent = (bits >> 10u) & 0x1fu;
    std::uint32_t mantissa = bits & 0x03ffu;

    std::uint32_t out = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            out = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1u;
                --exponent;
            }
            mantissa &= 0x03ffu;
            out = sign | ((exponent + (127u - 15u)) << 23u) | (mantissa << 13u);
        }
    } else if (exponent == 31) {
        out = sign | 0x7f800000u | (mantissa << 13u);
    } else {
        out = sign | ((exponent + (127u - 15u)) << 23u) | (mantissa << 13u);
    }

    return std::bit_cast<float>(out);
}

std::vector<float> make_kaiming_weights(std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> distribution(0.0f, std::sqrt(2.0f / static_cast<float>(kNnHiddenWidth)));

    std::vector<float> weights(weight_count());
    for (float &weight : weights) {
        weight = distribution(rng);
    }
    return weights;
}

std::vector<PackedFp16x8> pack_weights_fp16_uvec4(std::span<const float> weights) {
    if (weights.size() != weight_count()) {
        throw std::invalid_argument("NRC pack_weights_fp16_uvec4 received an invalid weight count");
    }

    std::vector<PackedFp16x8> packed((weights.size() + kFp16PerUvec4 - 1u) / kFp16PerUvec4);
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const std::uint16_t half = float_to_half_bits(weights[i]);
        PackedFp16x8 &block = packed[i / kFp16PerUvec4];
        const std::uint32_t lane = static_cast<std::uint32_t>(i % kFp16PerUvec4);
        const std::uint32_t component = lane >> 1u;
        const std::uint32_t shift = (lane & 1u) == 0u ? 0u : 16u;
        block[component] |= static_cast<std::uint32_t>(half) << shift;
    }
    return packed;
}

std::vector<float> unpack_weights_fp16_uvec4(std::span<const PackedFp16x8> packed,
                                             std::uint32_t valueCount) {
    if (packed.size() * kFp16PerUvec4 < valueCount) {
        throw std::invalid_argument("NRC unpack_weights_fp16_uvec4 received too few packed blocks");
    }

    std::vector<float> weights(valueCount);
    for (std::uint32_t i = 0; i < valueCount; ++i) {
        const PackedFp16x8 &block = packed[i / kFp16PerUvec4];
        const std::uint32_t lane = i % kFp16PerUvec4;
        const std::uint32_t component = lane >> 1u;
        const std::uint32_t shift = (lane & 1u) == 0u ? 0u : 16u;
        const auto half = static_cast<std::uint16_t>((block[component] >> shift) & 0xffffu);
        weights[i] = half_bits_to_float(half);
    }
    return weights;
}

} // namespace nrc
