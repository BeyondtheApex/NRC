#ifndef NRC_NN_SCALAR_GLSL
#define NRC_NN_SCALAR_GLSL

#include "../../common/NrcEncoding.glsl"

#ifndef NRC_NN_SET
#define NRC_NN_SET 0
#endif

#ifndef NRC_NN_WEIGHTS_BINDING
#define NRC_NN_WEIGHTS_BINDING 0
#endif

layout(std430, set = NRC_NN_SET, binding = NRC_NN_WEIGHTS_BINDING) readonly buffer NrcScalarWeightsBuffer {
    uvec4 nrcScalarWeights[];
};

float NrcScalarLoadWeight(uint index) {
    uvec4 packed = nrcScalarWeights[index / NRC_FP16_PER_UVEC4];
    uint lane = index & 7u;
    uint pairIndex = lane >> 1u;
    vec2 pairValue = unpackHalf2x16(packed[pairIndex]);
    return (lane & 1u) == 0u ? pairValue.x : pairValue.y;
}

void NrcScalarUnpackInput(uvec4 encoded[NRC_ENCODED_INPUT_UVEC4_COUNT], out float activations[NRC_NN_HIDDEN_WIDTH]) {
    for (uint block = 0u; block < NRC_ENCODED_INPUT_UVEC4_COUNT; ++block) {
        for (uint component = 0u; component < 4u; ++component) {
            vec2 pairValue = unpackHalf2x16(encoded[block][component]);
            uint base = block * NRC_FP16_PER_UVEC4 + component * 2u;
            activations[base] = pairValue.x;
            activations[base + 1u] = pairValue.y;
        }
    }
}

uint NrcScalarLayerWeightOffset(uint layer) {
    return layer * NRC_NN_HIDDEN_WIDTH * NRC_NN_HIDDEN_WIDTH;
}

void NrcScalarForward(uvec4 encoded[NRC_ENCODED_INPUT_UVEC4_COUNT], out vec3 outputValue) {
    float src[NRC_NN_HIDDEN_WIDTH];
    float dst[NRC_NN_HIDDEN_WIDTH];
    NrcScalarUnpackInput(encoded, src);

    for (uint layer = 0u; layer < NRC_NN_HIDDEN_LAYER_COUNT; ++layer) {
        uint offset = NrcScalarLayerWeightOffset(layer);
        for (uint row = 0u; row < NRC_NN_HIDDEN_WIDTH; ++row) {
            float value = 0.0;
            for (uint col = 0u; col < NRC_NN_HIDDEN_WIDTH; ++col) {
                value += NrcScalarLoadWeight(offset + row * NRC_NN_HIDDEN_WIDTH + col) * src[col];
            }
            dst[row] = max(value, 0.0);
        }

        for (uint i = 0u; i < NRC_NN_HIDDEN_WIDTH; ++i) {
            src[i] = dst[i];
        }
    }

    uint outputOffset = NrcScalarLayerWeightOffset(NRC_NN_HIDDEN_LAYER_COUNT);
    for (uint row = 0u; row < NRC_NN_OUTPUT_WIDTH; ++row) {
        float value = 0.0;
        for (uint col = 0u; col < NRC_NN_HIDDEN_WIDTH; ++col) {
            value += NrcScalarLoadWeight(outputOffset + row * NRC_NN_HIDDEN_WIDTH + col) * src[col];
        }
        outputValue[row] = value;
    }
}

#endif
