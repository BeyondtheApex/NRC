#include "nrc/nrc_encoding.hpp"

#include "nrc/nrc_weights.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace nrc {
namespace {

using Vec4 = std::array<float, 4>;

float half_round(float value) {
    return half_bits_to_float(float_to_half_bits(value));
}

float glsl_mod(float x, float y) {
    return x - y * std::floor(x / y);
}

Vec4 quartic_cdf(Vec4 x, float invRadius) {
    Vec4 out{};
    for (std::size_t i = 0; i < out.size(); ++i) {
        const float u = x[i] * invRadius;
        const float u2 = u * u;
        const float u4 = u2 * u2;
        out[i] = std::clamp((15.0f / 16.0f) *
                                u *
                                (1.0f - (2.0f / 3.0f) * u2 + (1.0f / 5.0f) * u4) +
                                0.5f,
                            0.0f,
                            1.0f);
    }
    return out;
}

Vec4 one_blob4_encode(float x) {
    constexpr Vec4 kLeft{0.0f, 0.25f, 0.5f, 0.75f};
    constexpr Vec4 kRight{0.25f, 0.5f, 0.75f, 1.0f};
    Vec4 leftDelta{};
    Vec4 rightDelta{};
    for (std::size_t i = 0; i < leftDelta.size(); ++i) {
        leftDelta[i] = kLeft[i] - x;
        rightDelta[i] = kRight[i] - x;
    }

    const Vec4 leftCdf = quartic_cdf(leftDelta, 4.0f);
    const Vec4 rightCdf = quartic_cdf(rightDelta, 4.0f);
    Vec4 out{};
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = rightCdf[i] - leftCdf[i];
    }
    return out;
}

Vec4 tri_wave(Vec4 x) {
    Vec4 out{};
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = 2.0f * std::fabs(glsl_mod(x[i] - 0.5f, 2.0f) - 1.0f) - 1.0f;
    }
    return out;
}

std::array<Vec4, 3> frequency_encode(float p) {
    constexpr std::array<Vec4, 3> kFrequencies{{
        {1.0f, 2.0f, 4.0f, 8.0f},
        {16.0f, 32.0f, 64.0f, 128.0f},
        {256.0f, 512.0f, 1024.0f, 2048.0f},
    }};

    std::array<Vec4, 3> out{};
    for (std::size_t row = 0; row < out.size(); ++row) {
        Vec4 scaled{};
        for (std::size_t component = 0; component < scaled.size(); ++component) {
            scaled[component] = kFrequencies[row][component] * p;
        }
        out[row] = tri_wave(scaled);
    }
    return out;
}

void append_half_rounded(EncodedInput &encoded, std::uint32_t &offset, Vec4 values) {
    for (float value : values) {
        encoded[offset++] = half_round(value);
    }
}

} // namespace

Vec2 spherical_encode(Vec3 direction) {
    const bool zeroXy = direction.x == 0.0f && direction.y == 0.0f;
    return {
        zeroXy ? 0.5f : 0.5f + std::atan2(direction.y, direction.x) / (2.0f * static_cast<float>(kPi)),
        std::acos(std::clamp(direction.z, -1.0f, 1.0f)) / static_cast<float>(kPi),
    };
}

void encode_input(const UnpackedNrcInput &input, EncodedInput &encoded) {
    const auto posFreq0 = frequency_encode(input.position.x);
    const auto posFreq1 = frequency_encode(input.position.y);
    const auto posFreq2 = frequency_encode(input.position.z);
    const Vec4 scattered0 = one_blob4_encode(input.scatteredDir.x);
    const Vec4 scattered1 = one_blob4_encode(input.scatteredDir.y);
    const Vec4 normal0 = one_blob4_encode(input.normal.x);
    const Vec4 normal1 = one_blob4_encode(input.normal.y);
    const Vec4 roughness = one_blob4_encode(1.0f - std::exp(-input.roughness));

    std::uint32_t offset = 0;
    append_half_rounded(encoded, offset, posFreq0[0]);
    append_half_rounded(encoded, offset, posFreq0[1]);
    append_half_rounded(encoded, offset, posFreq0[2]);
    append_half_rounded(encoded, offset, posFreq1[0]);
    append_half_rounded(encoded, offset, posFreq1[1]);
    append_half_rounded(encoded, offset, posFreq1[2]);
    append_half_rounded(encoded, offset, posFreq2[0]);
    append_half_rounded(encoded, offset, posFreq2[1]);
    append_half_rounded(encoded, offset, posFreq2[2]);
    append_half_rounded(encoded, offset, scattered0);
    append_half_rounded(encoded, offset, scattered1);
    append_half_rounded(encoded, offset, normal0);
    append_half_rounded(encoded, offset, normal1);
    append_half_rounded(encoded, offset, roughness);
    append_half_rounded(encoded, offset, {input.diffuse.x, input.diffuse.y, input.diffuse.z, input.specular.x});
    append_half_rounded(encoded, offset, {input.specular.y, input.specular.z, 1.0f, 1.0f});
}

} // namespace nrc
