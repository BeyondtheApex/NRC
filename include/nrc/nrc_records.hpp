#pragma once

#include <cstdint>

namespace nrc {

struct PackedNrcInput {
    std::uint32_t primitiveId = 0;
    std::uint32_t flipBitInstanceId = 0;
    std::uint32_t barycentric2x16U = 0;
    std::uint32_t scatteredDir2x16U = 0;
};

struct NrcEvalRecord {
    std::uint32_t dst = 0xffffffffu;
    PackedNrcInput packedInput = {};
};

struct NrcTrainRecord {
    float biasR = 0.0f;
    float biasG = 0.0f;
    float biasB = 0.0f;
    float factorR = 0.0f;
    float factorG = 0.0f;
    float factorB = 0.0f;
    PackedNrcInput packedInput = {};
};

struct NrcOptimizerState {
    std::uint32_t step = 0;
    float beta1T = 1.0f;
    float beta2T = 1.0f;
    float alphaT = 1.0f;
    float alphaTMinus1 = 0.0f;
};

struct NrcOptimizerEntry {
    float moment1 = 0.0f;
    float moment2 = 0.0f;
    float weight = 0.0f;
    float emaWeight = 0.0f;
};

static_assert(sizeof(PackedNrcInput) == 16);
static_assert(sizeof(NrcEvalRecord) == 20);
static_assert(sizeof(NrcTrainRecord) == 40);
static_assert(sizeof(NrcOptimizerState) == 20);
static_assert(sizeof(NrcOptimizerEntry) == 16);

} // namespace nrc
