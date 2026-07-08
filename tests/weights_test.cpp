#include "nrc/nrc_weights.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

bool require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }
    return true;
}

bool near(float lhs, float rhs, float epsilon = 1.0e-3f) {
    return std::fabs(lhs - rhs) <= epsilon;
}

} // namespace

int main() {
    bool ok = true;

    ok &= require(nrc::float_to_half_bits(0.0f) == 0x0000u, "half zero");
    ok &= require(nrc::float_to_half_bits(1.0f) == 0x3c00u, "half one");
    ok &= require(nrc::float_to_half_bits(-2.0f) == 0xc000u, "half minus two");
    ok &= require(near(nrc::half_bits_to_float(0x3800u), 0.5f), "half to float 0.5");

    auto weights = nrc::make_kaiming_weights(1234u);
    auto weightsAgain = nrc::make_kaiming_weights(1234u);
    auto differentWeights = nrc::make_kaiming_weights(1235u);

    ok &= require(weights.size() == nrc::weight_count(), "kaiming weight count");
    ok &= require(weights == weightsAgain, "kaiming initialization is deterministic by seed");
    ok &= require(weights != differentWeights, "different seed changes initialization");

    std::fill(weights.begin(), weights.end(), 0.0f);
    weights[0] = 1.0f;
    weights[1] = -2.0f;
    weights[2] = 0.5f;
    weights[7] = 3.0f;
    weights[8] = 4.0f;

    const auto packed = nrc::pack_weights_fp16_uvec4(weights);
    ok &= require(packed.size() == 2584u, "packed uvec4 block count");
    ok &= require(packed[0][0] == 0xc0003c00u, "first packed fp16 pair");
    ok &= require(packed[0][1] == 0x00003800u, "second packed fp16 pair");
    ok &= require(packed[0][3] == 0x42000000u, "fourth packed fp16 pair");
    ok &= require(packed[1][0] == 0x00004400u, "next block first fp16 value");

    const auto unpacked = nrc::unpack_weights_fp16_uvec4(packed, nrc::weight_count());
    ok &= require(near(unpacked[0], 1.0f), "unpack weight 0");
    ok &= require(near(unpacked[1], -2.0f), "unpack weight 1");
    ok &= require(near(unpacked[2], 0.5f), "unpack weight 2");
    ok &= require(near(unpacked[7], 3.0f), "unpack weight 7");
    ok &= require(near(unpacked[8], 4.0f), "unpack weight 8");

    bool packThrew = false;
    try {
        (void)nrc::pack_weights_fp16_uvec4({weights.data(), weights.size() - 1});
    } catch (const std::invalid_argument &) {
        packThrew = true;
    }
    ok &= require(packThrew, "invalid pack count throws");

    bool unpackThrew = false;
    try {
        (void)nrc::unpack_weights_fp16_uvec4({packed.data(), packed.size() - 1}, nrc::weight_count());
    } catch (const std::invalid_argument &) {
        unpackThrew = true;
    }
    ok &= require(unpackThrew, "invalid unpack count throws");

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
