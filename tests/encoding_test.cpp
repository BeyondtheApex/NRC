#include "nrc/nrc_encoding.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

bool require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }
    return true;
}

bool near(float lhs, float rhs, float epsilon = 1.0e-5f) {
    return std::fabs(lhs - rhs) <= epsilon;
}

} // namespace

int main() {
    bool ok = true;

    const nrc::Vec2 encodedZ = nrc::spherical_encode({0.0f, 0.0f, 1.0f});
    ok &= require(near(encodedZ.x, 0.5f), "spherical z azimuth");
    ok &= require(near(encodedZ.y, 0.0f), "spherical z elevation");

    const nrc::Vec2 encodedX = nrc::spherical_encode({1.0f, 0.0f, 0.0f});
    ok &= require(near(encodedX.x, 0.5f), "spherical x azimuth");
    ok &= require(near(encodedX.y, 0.5f), "spherical x elevation");

    nrc::UnpackedNrcInput input{};
    input.position = {0.0f, 0.0f, 0.0f};
    input.scatteredDir = {0.5f, 0.5f};
    input.normal = {0.5f, 0.5f};
    input.roughness = 0.0f;
    input.diffuse = {0.25f, 0.5f, 0.75f};
    input.specular = {0.125f, 0.375f, 0.625f};

    nrc::EncodedInput encoded{};
    nrc::encode_input(input, encoded);

    ok &= require(near(encoded[0], 0.0f), "zero position frequency lane 0");
    ok &= require(near(encoded[1], 0.0f), "zero position frequency lane 1");
    ok &= require(near(encoded[2], 0.0f), "zero position frequency lane 2");
    ok &= require(near(encoded[3], 0.0f), "zero position frequency lane 3");
    ok &= require(near(encoded[56], 0.25f), "diffuse r lane");
    ok &= require(near(encoded[57], 0.5f), "diffuse g lane");
    ok &= require(near(encoded[58], 0.75f), "diffuse b lane");
    ok &= require(near(encoded[59], 0.125f), "specular r lane");
    ok &= require(near(encoded[60], 0.375f), "specular g lane");
    ok &= require(near(encoded[61], 0.625f), "specular b lane");
    ok &= require(near(encoded[62], 1.0f), "constant lane 62");
    ok &= require(near(encoded[63], 1.0f), "constant lane 63");

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
