#ifndef NRC_TYPES_GLSL
#define NRC_TYPES_GLSL

#include "NrcConstants.glsl"

struct PackedNrcInput {
    uint primitiveId;
    uint flipBitInstanceId;
    uint barycentric2x16U;
    uint scatteredDir2x16U;
};

struct NrcEvalRecord {
    uint dst;
    PackedNrcInput packedInput;
};

struct NrcTrainRecord {
    float biasR;
    float biasG;
    float biasB;
    float factorR;
    float factorG;
    float factorB;
    PackedNrcInput packedInput;
};

struct NrcOptimizerState {
    uint step;
    float beta1T;
    float beta2T;
    float alphaT;
    float alphaTMinus1;
};

struct NrcOptimizerEntry {
    float moment1;
    float moment2;
    float weight;
    float emaWeight;
};

#define NRC_EVAL_INVALID_DST 0xffffffffu
#define NRC_EVAL_DST_SCREEN 0u
#define NRC_EVAL_DST_TRAIN 1u

uint NrcEncodeEvalDstScreen(uvec2 xy15) {
    return (xy15.x | (xy15.y << 15u)) << 1u;
}

uint NrcEncodeEvalDstTrain(uint batch2, uint left14, uint right14) {
    return (batch2 | (left14 << 2u) | (right14 << 16u)) << 1u | 1u;
}

uint NrcEvalDstType(uint encodedDst) {
    return encodedDst & 1u;
}

uvec2 NrcDecodeEvalDstScreen(uint encodedDst) {
    encodedDst >>= 1u;
    return uvec2(encodedDst & 0x7fffu, encodedDst >> 15u);
}

void NrcDecodeEvalDstTrain(uint encodedDst, out uint batch2, out uint left14, out uint right14) {
    encodedDst >>= 1u;
    batch2 = encodedDst & 3u;
    left14 = (encodedDst >> 2u) & 0x3fffu;
    right14 = encodedDst >> 16u;
}

#endif
