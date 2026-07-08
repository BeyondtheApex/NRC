#ifndef NRC_NN_KHR_GLSL
#define NRC_NN_KHR_GLSL

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_memory_scope_semantics : require
#ifdef NRC_NN_BACKPROPAGATION
#extension GL_EXT_shader_atomic_float : require
#endif

#include "../../common/NrcEncoding.glsl"

#ifndef NRC_NN_SET
#define NRC_NN_SET 0
#endif

#ifndef NRC_NN_WEIGHTS_BINDING
#define NRC_NN_WEIGHTS_BINDING 0
#endif

#ifdef NRC_NN_BACKPROPAGATION
#ifndef NRC_NN_DWEIGHTS_BINDING
#define NRC_NN_DWEIGHTS_BINDING 1
#endif
#endif

#ifndef NRC_KHR_WORKGROUP_SIZE
#define NRC_KHR_WORKGROUP_SIZE 128
#endif

#ifndef NRC_KHR_SUBGROUP_SIZE
#define NRC_KHR_SUBGROUP_SIZE 32
#endif

#ifndef NRC_KHR_COOPMAT_M
#define NRC_KHR_COOPMAT_M 16
#endif

#ifndef NRC_KHR_COOPMAT_N
#define NRC_KHR_COOPMAT_N 16
#endif

#ifndef NRC_KHR_COOPMAT_K
#define NRC_KHR_COOPMAT_K 16
#endif

#if NRC_KHR_WORKGROUP_SIZE != 128
#error NRC KHR backend currently requires workgroup size 128.
#endif

#if NRC_KHR_SUBGROUP_SIZE != 16 && NRC_KHR_SUBGROUP_SIZE != 32
#error NRC KHR backend currently supports subgroup size 16 or 32.
#endif

#if NRC_KHR_COOPMAT_M != 8 && NRC_KHR_COOPMAT_M != 16
#error NRC KHR backend currently supports cooperative matrix M size 8 or 16.
#endif

#if NRC_KHR_COOPMAT_N != 16 || NRC_KHR_COOPMAT_K != 16
#error NRC KHR backend currently requires cooperative matrix N=16 and K=16.
#endif

#define NRC_KHR_SUBGROUP_COUNT (NRC_KHR_WORKGROUP_SIZE / NRC_KHR_SUBGROUP_SIZE)
#define NRC_KHR_COOPMAT_K_BLOCKS (NRC_NN_HIDDEN_WIDTH / NRC_KHR_COOPMAT_K)
#define NRC_KHR_COOPMAT_M_BLOCKS (NRC_NN_HIDDEN_WIDTH / NRC_KHR_COOPMAT_M)
#define NRC_KHR_ACT_Y NRC_KHR_WORKGROUP_SIZE
#define NRC_KHR_ACT_UVEC4_COUNT ((NRC_NN_HIDDEN_WIDTH * NRC_KHR_ACT_Y) / NRC_FP16_PER_UVEC4)
#define NRC_KHR_ACT_COOPMAT_Y (NRC_KHR_ACT_Y / NRC_KHR_COOPMAT_N)
#define NRC_KHR_SUBGROUP_ACT_COOPMAT_Y (NRC_KHR_ACT_COOPMAT_Y / NRC_KHR_SUBGROUP_COUNT)
#define NRC_KHR_WEIGHT_64_UVEC4_COUNT ((NRC_NN_HIDDEN_WIDTH * NRC_NN_HIDDEN_WIDTH) / NRC_FP16_PER_UVEC4)
#define NRC_KHR_OUTPUT_WEIGHT_UVEC4_COUNT ((NRC_NN_HIDDEN_WIDTH * NRC_NN_OUTPUT_WIDTH) / NRC_FP16_PER_UVEC4)
#define NRC_KHR_THREAD_WEIGHT_64_UVEC4_COUNT (NRC_KHR_WEIGHT_64_UVEC4_COUNT / NRC_KHR_WORKGROUP_SIZE)
#define NRC_KHR_BACKPROP_SHARED_UVEC4_COUNT (NRC_KHR_WEIGHT_64_UVEC4_COUNT * NRC_KHR_SUBGROUP_COUNT)
#define NRC_KHR_MAT64_COOPMAT_STRIDE (NRC_NN_HIDDEN_WIDTH / NRC_FP16_PER_UVEC4)
#define NRC_KHR_ACC_COMPONENT_COUNT ((NRC_KHR_COOPMAT_M * NRC_KHR_COOPMAT_N) / NRC_KHR_SUBGROUP_SIZE)
#define NRC_KHR_WEIGHT_COOPMAT_ELEMENT(COL_BLOCK, ROW_BLOCK) \
    ((COL_BLOCK) * (NRC_KHR_COOPMAT_K / NRC_FP16_PER_UVEC4) + \
     (ROW_BLOCK) * NRC_KHR_COOPMAT_M * NRC_KHR_MAT64_COOPMAT_STRIDE)
#define NRC_KHR_ACT_COOPMAT_ELEMENT(ROW_BLOCK, COL_BLOCK) \
    ((ROW_BLOCK) * (NRC_KHR_COOPMAT_K / NRC_FP16_PER_UVEC4) + \
     (COL_BLOCK) * NRC_KHR_COOPMAT_N * NRC_KHR_MAT64_COOPMAT_STRIDE)
#define NRC_KHR_ACC_COOPMAT_ELEMENT(ROW_BLOCK, COL_BLOCK) \
    ((ROW_BLOCK) * (NRC_KHR_COOPMAT_M / NRC_FP16_PER_UVEC4) + \
     (COL_BLOCK) * NRC_KHR_COOPMAT_N * NRC_KHR_MAT64_COOPMAT_STRIDE)
#define NRC_KHR_WEIGHT_COOPMAT_MAJOR gl_CooperativeMatrixLayoutRowMajor
#define NRC_KHR_ACT_COOPMAT_MAJOR gl_CooperativeMatrixLayoutColumnMajor

#ifdef NRC_NN_BACKPROPAGATION
#if NRC_KHR_BACKPROP_SHARED_UVEC4_COUNT > NRC_KHR_ACT_UVEC4_COUNT
#define NRC_KHR_SHARED_BUFFER_UVEC4_COUNT NRC_KHR_BACKPROP_SHARED_UVEC4_COUNT
#else
#define NRC_KHR_SHARED_BUFFER_UVEC4_COUNT NRC_KHR_ACT_UVEC4_COUNT
#endif
#else
#define NRC_KHR_SHARED_BUFFER_UVEC4_COUNT NRC_KHR_ACT_UVEC4_COUNT
#endif

layout(std430, set = NRC_NN_SET, binding = NRC_NN_WEIGHTS_BINDING) readonly buffer NrcKhrWeightsBuffer {
    uvec4 nrcKhrWeights[];
};

#ifdef NRC_NN_BACKPROPAGATION
layout(std430, set = NRC_NN_SET, binding = NRC_NN_DWEIGHTS_BINDING) buffer NrcKhrDWeightsBuffer {
    float nrcKhrDWeights[];
};
#endif

shared uvec4 nrcKhrSharedBuffer[NRC_KHR_SHARED_BUFFER_UVEC4_COUNT];
coopmat<float16_t, gl_ScopeSubgroup, NRC_KHR_COOPMAT_K, NRC_KHR_COOPMAT_N, gl_MatrixUseB>
    nrcKhrActCoopmats[NRC_KHR_SUBGROUP_ACT_COOPMAT_Y][NRC_KHR_COOPMAT_K_BLOCKS];

uint NrcKhrLayerWeightUvec4Offset(uint layer) {
    return layer * NRC_KHR_WEIGHT_64_UVEC4_COUNT;
}

void NrcKhrLoadActivationsFromShared() {
    for (uint y = 0u; y < NRC_KHR_SUBGROUP_ACT_COOPMAT_Y; ++y) {
        uint workgroupY = gl_SubgroupID * NRC_KHR_SUBGROUP_ACT_COOPMAT_Y + y;
        for (uint kBlock = 0u; kBlock < NRC_KHR_COOPMAT_K_BLOCKS; ++kBlock) {
            coopMatLoad(nrcKhrActCoopmats[y][kBlock],
                        nrcKhrSharedBuffer,
                        NRC_KHR_ACT_COOPMAT_ELEMENT(kBlock, workgroupY),
                        NRC_KHR_MAT64_COOPMAT_STRIDE,
                        NRC_KHR_ACT_COOPMAT_MAJOR);
        }
    }
    barrier();
}

void NrcKhrLoadInput(uvec4 encoded[NRC_ENCODED_INPUT_UVEC4_COUNT]) {
    uint localBase = gl_LocalInvocationID.x * NRC_ENCODED_INPUT_UVEC4_COUNT;
    for (uint x = 0u; x < NRC_ENCODED_INPUT_UVEC4_COUNT; ++x) {
        nrcKhrSharedBuffer[localBase + x] = encoded[x];
    }
    barrier();
    NrcKhrLoadActivationsFromShared();
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
    coopmat<float16_t, gl_ScopeSubgroup, NRC_KHR_COOPMAT_M, NRC_KHR_COOPMAT_N, gl_MatrixUseAccumulator>
        dst[NRC_KHR_SUBGROUP_ACT_COOPMAT_Y][NRC_KHR_COOPMAT_M_BLOCKS];

    for (uint y = 0u; y < NRC_KHR_SUBGROUP_ACT_COOPMAT_Y; ++y) {
        for (uint rowBlock = 0u; rowBlock < NRC_KHR_COOPMAT_M_BLOCKS; ++rowBlock) {
            dst[y][rowBlock] =
                coopmat<float16_t, gl_ScopeSubgroup, NRC_KHR_COOPMAT_M, NRC_KHR_COOPMAT_N, gl_MatrixUseAccumulator>(0);
        }
    }

    for (uint rowBlock = 0u; rowBlock < NRC_KHR_COOPMAT_M_BLOCKS; ++rowBlock) {
        for (uint kBlock = 0u; kBlock < NRC_KHR_COOPMAT_K_BLOCKS; ++kBlock) {
            coopmat<float16_t, gl_ScopeSubgroup, NRC_KHR_COOPMAT_M, NRC_KHR_COOPMAT_K, gl_MatrixUseA> weightCoopmat;
            coopMatLoad(weightCoopmat,
                        nrcKhrSharedBuffer,
                        NRC_KHR_WEIGHT_COOPMAT_ELEMENT(kBlock, rowBlock),
                        NRC_KHR_MAT64_COOPMAT_STRIDE,
                        NRC_KHR_WEIGHT_COOPMAT_MAJOR);

            for (uint activationY = 0u; activationY < NRC_KHR_SUBGROUP_ACT_COOPMAT_Y; ++activationY) {
                dst[activationY][rowBlock] =
                    coopMatMulAdd(weightCoopmat,
                                  nrcKhrActCoopmats[activationY][kBlock],
                                  dst[activationY][rowBlock]);
            }
        }
    }
    barrier();

    for (uint y = 0u; y < NRC_KHR_SUBGROUP_ACT_COOPMAT_Y; ++y) {
        uint workgroupY = gl_SubgroupID * NRC_KHR_SUBGROUP_ACT_COOPMAT_Y + y;
        for (uint rowBlock = 0u; rowBlock < NRC_KHR_COOPMAT_M_BLOCKS; ++rowBlock) {
            for (uint element = 0u; element < NRC_KHR_ACC_COMPONENT_COUNT; ++element) {
                dst[y][rowBlock][element] = max(dst[y][rowBlock][element], float16_t(0));
            }
            coopMatStore(dst[y][rowBlock],
                         nrcKhrSharedBuffer,
                         NRC_KHR_ACC_COOPMAT_ELEMENT(rowBlock, workgroupY),
                         NRC_KHR_MAT64_COOPMAT_STRIDE,
                         NRC_KHR_ACT_COOPMAT_MAJOR);
        }
    }
    barrier();
    NrcKhrLoadActivationsFromShared();
}

void NrcKhrForwardOutputStore() {
    coopmat<float16_t, gl_ScopeSubgroup, NRC_KHR_COOPMAT_M, NRC_KHR_COOPMAT_N, gl_MatrixUseAccumulator>
        dst[NRC_KHR_SUBGROUP_ACT_COOPMAT_Y];

    for (uint y = 0u; y < NRC_KHR_SUBGROUP_ACT_COOPMAT_Y; ++y) {
        dst[y] =
            coopmat<float16_t, gl_ScopeSubgroup, NRC_KHR_COOPMAT_M, NRC_KHR_COOPMAT_N, gl_MatrixUseAccumulator>(0);
    }

    for (uint kBlock = 0u; kBlock < NRC_KHR_COOPMAT_K_BLOCKS; ++kBlock) {
        coopmat<float16_t, gl_ScopeSubgroup, NRC_KHR_COOPMAT_M, NRC_KHR_COOPMAT_K, gl_MatrixUseA> weightCoopmat;
        coopMatLoad(weightCoopmat,
                    nrcKhrSharedBuffer,
                    NRC_KHR_WEIGHT_COOPMAT_ELEMENT(kBlock, 0u),
                    NRC_KHR_MAT64_COOPMAT_STRIDE,
                    NRC_KHR_WEIGHT_COOPMAT_MAJOR);

        for (uint y = 0u; y < NRC_KHR_SUBGROUP_ACT_COOPMAT_Y; ++y) {
            dst[y] = coopMatMulAdd(weightCoopmat, nrcKhrActCoopmats[y][kBlock], dst[y]);
        }
    }
    barrier();

    for (uint y = 0u; y < NRC_KHR_SUBGROUP_ACT_COOPMAT_Y; ++y) {
        uint workgroupY = gl_SubgroupID * NRC_KHR_SUBGROUP_ACT_COOPMAT_Y + y;
        coopMatStore(dst[y],
                     nrcKhrSharedBuffer,
                     NRC_KHR_ACC_COOPMAT_ELEMENT(0u, workgroupY),
                     NRC_KHR_MAT64_COOPMAT_STRIDE,
                     NRC_KHR_ACT_COOPMAT_MAJOR);
    }
    barrier();
}

vec3 NrcKhrReadInvocationOutput() {
    uvec2 packed = nrcKhrSharedBuffer[gl_LocalInvocationID.x * NRC_ENCODED_INPUT_UVEC4_COUNT].rg;
    return vec3(unpackHalf2x16(packed.x), unpackHalf2x16(packed.y).x);
}

#ifdef NRC_NN_BACKPROPAGATION
void NrcKhrGradientAtomicAdd(uint index, float value) {
    atomicAdd(nrcKhrDWeights[index], value);
}

void NrcKhrOutputDerivative(vec3 prediction, vec3 target, out float dOutput[NRC_NN_OUTPUT_WIDTH]) {
    float luminance = dot(vec3(0.299, 0.587, 0.114), max(prediction, vec3(0.0)));
    float denominator = luminance * luminance + 0.01;
    vec3 dLoss = 2.0 * NRC_LOSS_SCALE * (prediction - target) / denominator;
    dOutput[0] = dLoss.x;
    dOutput[1] = dLoss.y;
    dOutput[2] = dLoss.z;
}
#endif

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
