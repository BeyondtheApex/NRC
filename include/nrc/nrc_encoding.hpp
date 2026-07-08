#pragma once

#include "nrc/nrc_constants.hpp"

#include <array>

namespace nrc {

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct UnpackedNrcInput {
    Vec3 position;
    Vec2 scatteredDir;
    Vec2 normal;
    float roughness = 0.0f;
    Vec3 diffuse;
    Vec3 specular;
};

using EncodedInput = std::array<float, kEncodedInputWidth>;

Vec2 spherical_encode(Vec3 direction);
void encode_input(const UnpackedNrcInput &input, EncodedInput &encoded);

} // namespace nrc
