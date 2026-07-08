#ifndef NRC_NN_KHR_GLSL
#define NRC_NN_KHR_GLSL

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_memory_scope_semantics : require

#include "../../common/NrcEncoding.glsl"

#ifndef NRC_NN_SET
#define NRC_NN_SET 0
#endif

#ifndef NRC_NN_WEIGHTS_BINDING
#define NRC_NN_WEIGHTS_BINDING 0
#endif

#ifndef NRC_KHR_WORKGROUP_SIZE
#define NRC_KHR_WORKGROUP_SIZE 128
#endif

#ifndef NRC_KHR_SUBGROUP_SIZE
#define NRC_KHR_SUBGROUP_SIZE 32
#endif

#if NRC_KHR_WORKGROUP_SIZE != 128
#error NRC KHR backend currently requires workgroup size 128.
#endif

#if NRC_KHR_SUBGROUP_SIZE != 16 && NRC_KHR_SUBGROUP_SIZE != 32
#error NRC KHR backend currently supports subgroup size 16 or 32.
#endif

#define NRC_KHR_SUBGROUP_COUNT (NRC_KHR_WORKGROUP_SIZE / NRC_KHR_SUBGROUP_SIZE)
#define NRC_KHR_COOPMAT_M 16
#define NRC_KHR_COOPMAT_N 16
#define NRC_KHR_COOPMAT_K 16
#define NRC_KHR_COOPMAT_X (NRC_NN_HIDDEN_WIDTH / NRC_KHR_COOPMAT_K)
#define NRC_KHR_ACT_Y NRC_KHR_WORKGROUP_SIZE
#define NRC_KHR_ACT_UVEC4_COUNT ((NRC_NN_HIDDEN_WIDTH * NRC_KHR_ACT_Y) / NRC_FP16_PER_UVEC4)
#define NRC_KHR_ACT_COOPMAT_Y (NRC_KHR_ACT_Y / NRC_KHR_COOPMAT_N)
#define NRC_KHR_SUBGROUP_ACT_COOPMAT_Y (NRC_KHR_ACT_COOPMAT_Y / NRC_KHR_SUBGROUP_COUNT)
#define NRC_KHR_WEIGHT_64_UVEC4_COUNT ((NRC_NN_HIDDEN_WIDTH * NRC_NN_HIDDEN_WIDTH) / NRC_FP16_PER_UVEC4)
#define NRC_KHR_OUTPUT_WEIGHT_UVEC4_COUNT ((NRC_NN_HIDDEN_WIDTH * NRC_NN_OUTPUT_WIDTH) / NRC_FP16_PER_UVEC4)
#define NRC_KHR_THREAD_WEIGHT_64_UVEC4_COUNT (NRC_KHR_WEIGHT_64_UVEC4_COUNT / NRC_KHR_WORKGROUP_SIZE)
#define NRC_KHR_MAT64_COOPMAT_STRIDE (NRC_NN_HIDDEN_WIDTH / NRC_FP16_PER_UVEC4)
#define NRC_KHR_MAT64_COOPMAT_ELEMENT(X, Y) \
    ((X) * (NRC_KHR_COOPMAT_K / NRC_FP16_PER_UVEC4) + (Y) * NRC_KHR_COOPMAT_N * NRC_KHR_MAT64_COOPMAT_STRIDE)
#define NRC_KHR_WEIGHT_COOPMAT_MAJOR gl_CooperativeMatrixLayoutRowMajor
#define NRC_KHR_ACT_COOPMAT_MAJOR gl_CooperativeMatrixLayoutColumnMajor

layout(std430, set = NRC_NN_SET, binding = NRC_NN_WEIGHTS_BINDING) readonly buffer NrcKhrWeightsBuffer {
    uvec4 nrcKhrWeights[];
};

shared uvec4 nrcKhrSharedBuffer[NRC_KHR_ACT_UVEC4_COUNT];
coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>
    nrcKhrActCoopmats[NRC_KHR_SUBGROUP_ACT_COOPMAT_Y][NRC_KHR_COOPMAT_X];

uint NrcKhrLayerWeightUvec4Offset(uint layer) {
    return layer * NRC_KHR_WEIGHT_64_UVEC4_COUNT;
}

void NrcKhrLoadInput(uvec4 encoded[NRC_ENCODED_INPUT_UVEC4_COUNT]) {
    uint localBase = gl_LocalInvocationID.x * NRC_ENCODED_INPUT_UVEC4_COUNT;
    for (uint x = 0u; x < NRC_ENCODED_INPUT_UVEC4_COUNT; ++x) {
        nrcKhrSharedBuffer[localBase + x] = encoded[x];
    }
    barrier();

    for (uint y = 0u; y < NRC_KHR_SUBGROUP_ACT_COOPMAT_Y; ++y) {
        uint workgroupY = gl_SubgroupID * NRC_KHR_SUBGROUP_ACT_COOPMAT_Y + y;
        for (uint x = 0u; x < NRC_KHR_COOPMAT_X; ++x) {
            coopMatLoad(nrcKhrActCoopmats[y][x],
                        nrcKhrSharedBuffer,
                        NRC_KHR_MAT64_COOPMAT_ELEMENT(x, workgroupY),
                        NRC_KHR_MAT64_COOPMAT_STRIDE,
                        NRC_KHR_ACT_COOPMAT_MAJOR);
        }
    }
    barrier();
}

void NrcKhrLoadHiddenWeights(uint layer) {
    uint sharedBase = gl_LocalInvocationID.x * NRC_KHR_THREAD_WEIGHT_64_UVEC4_COUNT;
    uint weightBase = NrcKhrLayerWeightUvec4Offset(layer) + sharedBase;
    for (uint i = 0u; i < NRC_KHR_THREAD_WEIGHT_64_UVEC4_COUNT; ++i) {
        nrcKhrSharedBuffer[sharedBase + i] = nrcKhrWeights[weightBase + i];
    }
    barrier();
}

void NrcKhrLoadOutputWeights(uint layer) {
    uint weightBase = NrcKhrLayerWeightUvec4Offset(layer);
    nrcKhrSharedBuffer[gl_LocalInvocationID.x] =
        gl_LocalInvocationID.x < NRC_KHR_OUTPUT_WEIGHT_UVEC4_COUNT
            ? nrcKhrWeights[weightBase + gl_LocalInvocationID.x]
            : uvec4(0u);
    barrier();
}

void NrcKhrForwardHiddenRelu() {
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>
        dst[NRC_KHR_SUBGROUP_ACT_COOPMAT_Y][NRC_KHR_COOPMAT_X];

    for (uint y = 0u; y < NRC_KHR_SUBGROUP_ACT_COOPMAT_Y; ++y) {
        for (uint x = 0u; x < NRC_KHR_COOPMAT_X; ++x) {
            dst[y][x] = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0);
        }
    }

    for (uint weightY = 0u; weightY < NRC_KHR_COOPMAT_X; ++weightY) {
        for (uint x = 0u; x < NRC_KHR_COOPMAT_X; ++x) {
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> weightCoopmat;
            coopMatLoad(weightCoopmat,
                        nrcKhrSharedBuffer,
                        NRC_KHR_MAT64_COOPMAT_ELEMENT(x, weightY),
                        NRC_KHR_MAT64_COOPMAT_STRIDE,
                        NRC_KHR_WEIGHT_COOPMAT_MAJOR);

            for (uint activationY = 0u; activationY < NRC_KHR_SUBGROUP_ACT_COOPMAT_Y; ++activationY) {
                dst[activationY][weightY] =
                    coopMatMulAdd(weightCoopmat,
                                  nrcKhrActCoopmats[activationY][x],
                                  dst[activationY][weightY]);
            }
        }
    }
    barrier();

    for (uint y = 0u; y < NRC_KHR_SUBGROUP_ACT_COOPMAT_Y; ++y) {
        for (uint x = 0u; x < NRC_KHR_COOPMAT_X; ++x) {
            for (uint k = 0u; k < (256u / NRC_KHR_SUBGROUP_SIZE); ++k) {
                nrcKhrActCoopmats[y][x][k] = max(dst[y][x][k], float16_t(0));
            }
        }
    }
}

void NrcKhrForwardOutputStore() {
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>
        dst[NRC_KHR_SUBGROUP_ACT_COOPMAT_Y];

    for (uint y = 0u; y < NRC_KHR_SUBGROUP_ACT_COOPMAT_Y; ++y) {
        dst[y] = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0);
    }

    for (uint x = 0u; x < NRC_KHR_COOPMAT_X; ++x) {
        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> weightCoopmat;
        coopMatLoad(weightCoopmat,
                    nrcKhrSharedBuffer,
                    NRC_KHR_MAT64_COOPMAT_ELEMENT(x, 0u),
                    NRC_KHR_MAT64_COOPMAT_STRIDE,
                    NRC_KHR_WEIGHT_COOPMAT_MAJOR);

        for (uint y = 0u; y < NRC_KHR_SUBGROUP_ACT_COOPMAT_Y; ++y) {
            dst[y] = coopMatMulAdd(weightCoopmat, nrcKhrActCoopmats[y][x], dst[y]);
        }
    }
    barrier();

    for (uint y = 0u; y < NRC_KHR_SUBGROUP_ACT_COOPMAT_Y; ++y) {
        uint workgroupY = gl_SubgroupID * NRC_KHR_SUBGROUP_ACT_COOPMAT_Y + y;
        coopMatStore(dst[y],
                     nrcKhrSharedBuffer,
                     NRC_KHR_MAT64_COOPMAT_ELEMENT(0u, workgroupY),
                     NRC_KHR_MAT64_COOPMAT_STRIDE,
                     NRC_KHR_ACT_COOPMAT_MAJOR);
    }
    barrier();
}

vec3 NrcKhrReadInvocationOutput() {
    uvec2 packed = nrcKhrSharedBuffer[gl_LocalInvocationID.x * NRC_ENCODED_INPUT_UVEC4_COUNT].rg;
    return vec3(unpackHalf2x16(packed.x), unpackHalf2x16(packed.y).x);
}

void NrcKhrForward(uvec4 encoded[NRC_ENCODED_INPUT_UVEC4_COUNT], out vec3 outputValue) {
    NrcKhrLoadInput(encoded);

    for (uint layer = 0u; layer < NRC_NN_HIDDEN_LAYER_COUNT; ++layer) {
        NrcKhrLoadHiddenWeights(layer);
        NrcKhrForwardHiddenRelu();
    }

    NrcKhrLoadOutputWeights(NRC_NN_HIDDEN_LAYER_COUNT);
    NrcKhrForwardOutputStore();
    outputValue = NrcKhrReadInvocationOutput();
}

#endif
