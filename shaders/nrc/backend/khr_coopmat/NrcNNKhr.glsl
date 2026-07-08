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
#define NRC_KHR_FP32_PER_UVEC4 4
#define NRC_KHR_DPRE_UVEC4_PER_SAMPLE (NRC_NN_HIDDEN_WIDTH / NRC_KHR_FP32_PER_UVEC4)
#define NRC_KHR_DPRE_UVEC4_COUNT (NRC_KHR_WORKGROUP_SIZE * NRC_KHR_DPRE_UVEC4_PER_SAMPLE)
#define NRC_KHR_BACKPROP_ACT_COPY_UVEC4_BASE NRC_KHR_ACT_UVEC4_COUNT
#define NRC_KHR_BACKPROP_SOURCE_ACT_COPY_UVEC4_BASE (NRC_KHR_BACKPROP_ACT_COPY_UVEC4_BASE + NRC_KHR_ACT_UVEC4_COUNT)
#define NRC_KHR_BACKPROP_PREVIOUS_SOURCE_ACT_COPY_UVEC4_BASE (NRC_KHR_BACKPROP_SOURCE_ACT_COPY_UVEC4_BASE + NRC_KHR_ACT_UVEC4_COUNT)
#define NRC_KHR_BACKPROP_EARLIER_SOURCE_ACT_COPY_UVEC4_BASE (NRC_KHR_BACKPROP_PREVIOUS_SOURCE_ACT_COPY_UVEC4_BASE + NRC_KHR_ACT_UVEC4_COUNT)
#define NRC_KHR_BACKPROP_OLDEST_SOURCE_ACT_COPY_UVEC4_BASE (NRC_KHR_BACKPROP_EARLIER_SOURCE_ACT_COPY_UVEC4_BASE + NRC_KHR_ACT_UVEC4_COUNT)
#define NRC_KHR_BACKPROP_ACT_COPY_SHARED_UVEC4_COUNT (NRC_KHR_ACT_UVEC4_COUNT * 6)
#define NRC_KHR_BACKPROP_REDUCTION_SHARED_UVEC4_COUNT (NRC_KHR_WEIGHT_64_UVEC4_COUNT * NRC_KHR_SUBGROUP_COUNT)
#if NRC_KHR_BACKPROP_REDUCTION_SHARED_UVEC4_COUNT > NRC_KHR_BACKPROP_ACT_COPY_SHARED_UVEC4_COUNT
#define NRC_KHR_BACKPROP_SHARED_UVEC4_COUNT NRC_KHR_BACKPROP_REDUCTION_SHARED_UVEC4_COUNT
#else
#define NRC_KHR_BACKPROP_SHARED_UVEC4_COUNT NRC_KHR_BACKPROP_ACT_COPY_SHARED_UVEC4_COUNT
#endif
#if NRC_KHR_BACKPROP_SHARED_UVEC4_COUNT > NRC_KHR_ACT_UVEC4_COUNT
#define NRC_KHR_SHARED_BUFFER_UVEC4_COUNT NRC_KHR_BACKPROP_SHARED_UVEC4_COUNT
#else
#define NRC_KHR_SHARED_BUFFER_UVEC4_COUNT NRC_KHR_ACT_UVEC4_COUNT
#endif
#if NRC_NN_HIDDEN_WIDTH % NRC_KHR_FP32_PER_UVEC4 != 0
#error NRC KHR backprop dPre scratch requires hidden width divisible by 4.
#endif
#if NRC_KHR_DPRE_UVEC4_COUNT > (NRC_KHR_ACT_UVEC4_COUNT * 2)
#error NRC KHR backprop dPre scratch must fit in the active tile plus retired final-hidden activation snapshot.
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

uint NrcKhrLayerWeightFloatOffset(uint layer) {
    return layer * NRC_NN_HIDDEN_WIDTH * NRC_NN_HIDDEN_WIDTH;
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

float NrcKhrLoadWeightFloat(uint index) {
    uvec4 packed = nrcKhrWeights[index / NRC_FP16_PER_UVEC4];
    uint lane = index & (NRC_FP16_PER_UVEC4 - 1u);
    vec2 pairValue = unpackHalf2x16(packed[lane >> 1u]);
    return (lane & 1u) == 0u ? pairValue.x : pairValue.y;
}

float NrcKhrUnpackFeature(uvec4 packed, uint lane) {
    vec2 pairValue = unpackHalf2x16(packed[lane >> 1u]);
    return (lane & 1u) == 0u ? pairValue.x : pairValue.y;
}

void NrcKhrStoreBackpropActivationsAt(uint dstBaseOffset) {
    uint srcBase = gl_LocalInvocationID.x * NRC_ENCODED_INPUT_UVEC4_COUNT;
    uint dstBase = dstBaseOffset + srcBase;
    for (uint i = 0u; i < NRC_ENCODED_INPUT_UVEC4_COUNT; ++i) {
        nrcKhrSharedBuffer[dstBase + i] = nrcKhrSharedBuffer[srcBase + i];
    }
    barrier();
}

void NrcKhrStoreBackpropActivations() {
    NrcKhrStoreBackpropActivationsAt(NRC_KHR_BACKPROP_ACT_COPY_UVEC4_BASE);
}

void NrcKhrStoreBackpropSourceActivations() {
    NrcKhrStoreBackpropActivationsAt(NRC_KHR_BACKPROP_SOURCE_ACT_COPY_UVEC4_BASE);
}

void NrcKhrStoreBackpropPreviousSourceActivations() {
    NrcKhrStoreBackpropActivationsAt(NRC_KHR_BACKPROP_PREVIOUS_SOURCE_ACT_COPY_UVEC4_BASE);
}

void NrcKhrStoreBackpropEarlierSourceActivations() {
    NrcKhrStoreBackpropActivationsAt(NRC_KHR_BACKPROP_EARLIER_SOURCE_ACT_COPY_UVEC4_BASE);
}

void NrcKhrStoreBackpropOldestSourceActivations() {
    NrcKhrStoreBackpropActivationsAt(NRC_KHR_BACKPROP_OLDEST_SOURCE_ACT_COPY_UVEC4_BASE);
}

float NrcKhrReadBackpropActivationFrom(uint baseOffset, uint sampleIndex, uint feature) {
    uint base = baseOffset +
                sampleIndex * NRC_ENCODED_INPUT_UVEC4_COUNT;
    return NrcKhrUnpackFeature(nrcKhrSharedBuffer[base + feature / NRC_FP16_PER_UVEC4],
                               feature & (NRC_FP16_PER_UVEC4 - 1u));
}

float NrcKhrReadBackpropActivation(uint sampleIndex, uint feature) {
    return NrcKhrReadBackpropActivationFrom(NRC_KHR_BACKPROP_ACT_COPY_UVEC4_BASE,
                                            sampleIndex,
                                            feature);
}

float NrcKhrReadBackpropSourceActivation(uint sampleIndex, uint feature) {
    return NrcKhrReadBackpropActivationFrom(NRC_KHR_BACKPROP_SOURCE_ACT_COPY_UVEC4_BASE,
                                            sampleIndex,
                                            feature);
}

float NrcKhrReadBackpropPreviousSourceActivation(uint sampleIndex, uint feature) {
    return NrcKhrReadBackpropActivationFrom(NRC_KHR_BACKPROP_PREVIOUS_SOURCE_ACT_COPY_UVEC4_BASE,
                                            sampleIndex,
                                            feature);
}

float NrcKhrReadBackpropEarlierSourceActivation(uint sampleIndex, uint feature) {
    return NrcKhrReadBackpropActivationFrom(NRC_KHR_BACKPROP_EARLIER_SOURCE_ACT_COPY_UVEC4_BASE,
                                            sampleIndex,
                                            feature);
}

float NrcKhrReadBackpropOldestSourceActivation(uint sampleIndex, uint feature) {
    return NrcKhrReadBackpropActivationFrom(NRC_KHR_BACKPROP_OLDEST_SOURCE_ACT_COPY_UVEC4_BASE,
                                            sampleIndex,
                                            feature);
}

float NrcKhrReadEncodedActivation(uvec4 encoded[NRC_ENCODED_INPUT_UVEC4_COUNT], uint feature) {
    return NrcKhrUnpackFeature(encoded[feature / NRC_FP16_PER_UVEC4],
                               feature & (NRC_FP16_PER_UVEC4 - 1u));
}

void NrcKhrOutputDerivative(vec3 prediction, vec3 target, out float dOutput[NRC_NN_OUTPUT_WIDTH]) {
    float luminance = dot(vec3(0.299, 0.587, 0.114), max(prediction, vec3(0.0)));
    float denominator = luminance * luminance + 0.01;
    vec3 dLoss = 2.0 * NRC_LOSS_SCALE * (prediction - target) / denominator;
    dOutput[0] = dLoss.x;
    dOutput[1] = dLoss.y;
    dOutput[2] = dLoss.z;
}

void NrcKhrAccumulateOutputDWeights(float dOutput[NRC_NN_OUTPUT_WIDTH]) {
    uint sampleIndex = gl_LocalInvocationID.x;
    uint outputOffset = NrcKhrLayerWeightFloatOffset(NRC_NN_HIDDEN_LAYER_COUNT);
    for (uint col = 0u; col < NRC_NN_HIDDEN_WIDTH; ++col) {
        float activation = NrcKhrReadBackpropActivation(sampleIndex, col);
        for (uint row = 0u; row < NRC_NN_OUTPUT_WIDTH; ++row) {
            NrcKhrGradientAtomicAdd(outputOffset + row * NRC_NN_HIDDEN_WIDTH + col,
                                    dOutput[row] * activation);
        }
    }
}

void NrcKhrAccumulateOutputDWeightsWorkgroup(float dOutput[NRC_NN_OUTPUT_WIDTH]) {
    uint sampleIndex = gl_LocalInvocationID.x;
    nrcKhrSharedBuffer[sampleIndex] = uvec4(floatBitsToUint(dOutput[0]),
                                            floatBitsToUint(dOutput[1]),
                                            floatBitsToUint(dOutput[2]),
                                            0u);
    barrier();

    uint outputOffset = NrcKhrLayerWeightFloatOffset(NRC_NN_HIDDEN_LAYER_COUNT);
    for (uint weight = gl_LocalInvocationID.x;
         weight < NRC_NN_OUTPUT_WIDTH * NRC_NN_HIDDEN_WIDTH;
         weight += NRC_KHR_WORKGROUP_SIZE) {
        uint row = weight / NRC_NN_HIDDEN_WIDTH;
        uint col = weight - row * NRC_NN_HIDDEN_WIDTH;
        float value = 0.0;
        for (uint sampleIndexInGroup = 0u;
             sampleIndexInGroup < NRC_KHR_WORKGROUP_SIZE;
             ++sampleIndexInGroup) {
            float sampleDOutput = uintBitsToFloat(nrcKhrSharedBuffer[sampleIndexInGroup][row]);
            float activation = NrcKhrReadBackpropActivation(sampleIndexInGroup, col);
            value += sampleDOutput * activation;
        }
        NrcKhrGradientAtomicAdd(outputOffset + row * NRC_NN_HIDDEN_WIDTH + col, value);
    }
    barrier();
}

void NrcKhrPropagateOutputDActivation(float dOutput[NRC_NN_OUTPUT_WIDTH],
                                      out float dActivation[NRC_NN_HIDDEN_WIDTH]) {
    uint outputOffset = NrcKhrLayerWeightFloatOffset(NRC_NN_HIDDEN_LAYER_COUNT);
    for (uint col = 0u; col < NRC_NN_HIDDEN_WIDTH; ++col) {
        float value = 0.0;
        for (uint row = 0u; row < NRC_NN_OUTPUT_WIDTH; ++row) {
            value += NrcKhrLoadWeightFloat(outputOffset + row * NRC_NN_HIDDEN_WIDTH + col) *
                     dOutput[row];
        }
        dActivation[col] = value;
    }
}

void NrcKhrPropagateHiddenDActivation(uint layer,
                                      uint activationBaseOffset,
                                      float dActivation[NRC_NN_HIDDEN_WIDTH],
                                      out float dSourceActivation[NRC_NN_HIDDEN_WIDTH]) {
    uint sampleIndex = gl_LocalInvocationID.x;
    uint layerOffset = NrcKhrLayerWeightFloatOffset(layer);
    for (uint col = 0u; col < NRC_NN_HIDDEN_WIDTH; ++col) {
        float value = 0.0;
        for (uint row = 0u; row < NRC_NN_HIDDEN_WIDTH; ++row) {
            float activation = NrcKhrReadBackpropActivationFrom(activationBaseOffset,
                                                                sampleIndex,
                                                                row);
            float dPreActivation = activation > 0.0 ? dActivation[row] : 0.0;
            value += NrcKhrLoadWeightFloat(layerOffset + row * NRC_NN_HIDDEN_WIDTH + col) *
                     dPreActivation;
        }
        dSourceActivation[col] = value;
    }
}

void NrcKhrAccumulateHiddenDWeights(uint layer,
                                    uint activationBaseOffset,
                                    uint sourceBaseOffset,
                                    float dActivation[NRC_NN_HIDDEN_WIDTH]) {
    uint sampleIndex = gl_LocalInvocationID.x;
    uint layerOffset = NrcKhrLayerWeightFloatOffset(layer);
    for (uint row = 0u; row < NRC_NN_HIDDEN_WIDTH; ++row) {
        float activation = NrcKhrReadBackpropActivationFrom(activationBaseOffset,
                                                            sampleIndex,
                                                            row);
        float dPreActivation = activation > 0.0 ? dActivation[row] : 0.0;
        for (uint col = 0u; col < NRC_NN_HIDDEN_WIDTH; ++col) {
            float srcActivation = NrcKhrReadBackpropActivationFrom(sourceBaseOffset,
                                                                   sampleIndex,
                                                                   col);
            NrcKhrGradientAtomicAdd(layerOffset + row * NRC_NN_HIDDEN_WIDTH + col,
                                    dPreActivation * srcActivation);
        }
    }
}

float NrcKhrReadSharedOutputDerivative(uint sampleIndex, uint row) {
    return uintBitsToFloat(nrcKhrSharedBuffer[sampleIndex][row]);
}

float NrcKhrPropagateSharedOutputDActivation(uint sampleIndex, uint col) {
    uint outputOffset = NrcKhrLayerWeightFloatOffset(NRC_NN_HIDDEN_LAYER_COUNT);
    float value = 0.0;
    for (uint row = 0u; row < NRC_NN_OUTPUT_WIDTH; ++row) {
        value += NrcKhrLoadWeightFloat(outputOffset + row * NRC_NN_HIDDEN_WIDTH + col) *
                 NrcKhrReadSharedOutputDerivative(sampleIndex, row);
    }
    return value;
}

void NrcKhrAccumulateFinalHiddenDWeightsWorkgroup(uint layer,
                                                 uint activationBaseOffset,
                                                 uint sourceBaseOffset) {
    uint layerOffset = NrcKhrLayerWeightFloatOffset(layer);
    for (uint weight = gl_LocalInvocationID.x;
         weight < NRC_NN_HIDDEN_WIDTH * NRC_NN_HIDDEN_WIDTH;
         weight += NRC_KHR_WORKGROUP_SIZE) {
        uint row = weight / NRC_NN_HIDDEN_WIDTH;
        uint col = weight - row * NRC_NN_HIDDEN_WIDTH;
        float value = 0.0;
        for (uint sampleIndexInGroup = 0u;
             sampleIndexInGroup < NRC_KHR_WORKGROUP_SIZE;
             ++sampleIndexInGroup) {
            float activation = NrcKhrReadBackpropActivationFrom(activationBaseOffset,
                                                                sampleIndexInGroup,
                                                                row);
            float dActivation = NrcKhrPropagateSharedOutputDActivation(sampleIndexInGroup, row);
            float dPreActivation = activation > 0.0 ? dActivation : 0.0;
            float srcActivation = NrcKhrReadBackpropActivationFrom(sourceBaseOffset,
                                                                   sampleIndexInGroup,
                                                                   col);
            value += dPreActivation * srcActivation;
        }
        NrcKhrGradientAtomicAdd(layerOffset + row * NRC_NN_HIDDEN_WIDTH + col, value);
    }
    barrier();
}

void NrcKhrStoreSharedDPreActivations(uint activationBaseOffset,
                                      float dActivation[NRC_NN_HIDDEN_WIDTH]) {
    uint sampleIndex = gl_LocalInvocationID.x;
    uint sampleBase = sampleIndex * NRC_KHR_DPRE_UVEC4_PER_SAMPLE;
    for (uint block = 0u; block < NRC_KHR_DPRE_UVEC4_PER_SAMPLE; ++block) {
        uint feature = block * NRC_KHR_FP32_PER_UVEC4;
        float activation0 = NrcKhrReadBackpropActivationFrom(activationBaseOffset,
                                                             sampleIndex,
                                                             feature + 0u);
        float activation1 = NrcKhrReadBackpropActivationFrom(activationBaseOffset,
                                                             sampleIndex,
                                                             feature + 1u);
        float activation2 = NrcKhrReadBackpropActivationFrom(activationBaseOffset,
                                                             sampleIndex,
                                                             feature + 2u);
        float activation3 = NrcKhrReadBackpropActivationFrom(activationBaseOffset,
                                                             sampleIndex,
                                                             feature + 3u);
        float dPre0 = activation0 > 0.0 ? dActivation[feature + 0u] : 0.0;
        float dPre1 = activation1 > 0.0 ? dActivation[feature + 1u] : 0.0;
        float dPre2 = activation2 > 0.0 ? dActivation[feature + 2u] : 0.0;
        float dPre3 = activation3 > 0.0 ? dActivation[feature + 3u] : 0.0;
        nrcKhrSharedBuffer[sampleBase + block] = uvec4(floatBitsToUint(dPre0),
                                                       floatBitsToUint(dPre1),
                                                       floatBitsToUint(dPre2),
                                                       floatBitsToUint(dPre3));
    }
    barrier();
}

float NrcKhrReadSharedDPreActivation(uint sampleIndex, uint feature) {
    uvec4 packed = nrcKhrSharedBuffer[sampleIndex * NRC_KHR_DPRE_UVEC4_PER_SAMPLE +
                                      feature / NRC_KHR_FP32_PER_UVEC4];
    return uintBitsToFloat(packed[feature & (NRC_KHR_FP32_PER_UVEC4 - 1u)]);
}

void NrcKhrAccumulateSharedHiddenDWeightsWorkgroup(uint layer,
                                                   uint sourceBaseOffset) {
    uint layerOffset = NrcKhrLayerWeightFloatOffset(layer);
    for (uint weight = gl_LocalInvocationID.x;
         weight < NRC_NN_HIDDEN_WIDTH * NRC_NN_HIDDEN_WIDTH;
         weight += NRC_KHR_WORKGROUP_SIZE) {
        uint row = weight / NRC_NN_HIDDEN_WIDTH;
        uint col = weight - row * NRC_NN_HIDDEN_WIDTH;
        float value = 0.0;
        for (uint sampleIndexInGroup = 0u;
             sampleIndexInGroup < NRC_KHR_WORKGROUP_SIZE;
             ++sampleIndexInGroup) {
            float dPreActivation = NrcKhrReadSharedDPreActivation(sampleIndexInGroup, row);
            float srcActivation = NrcKhrReadBackpropActivationFrom(sourceBaseOffset,
                                                                   sampleIndexInGroup,
                                                                   col);
            value += dPreActivation * srcActivation;
        }
        NrcKhrGradientAtomicAdd(layerOffset + row * NRC_NN_HIDDEN_WIDTH + col, value);
    }
    barrier();
}

void NrcKhrAccumulateInputHiddenDWeights(float dActivation[NRC_NN_HIDDEN_WIDTH],
                                         uvec4 encoded[NRC_ENCODED_INPUT_UVEC4_COUNT]) {
    uint sampleIndex = gl_LocalInvocationID.x;
    for (uint row = 0u; row < NRC_NN_HIDDEN_WIDTH; ++row) {
        float activation = NrcKhrReadBackpropOldestSourceActivation(sampleIndex, row);
        float dPreActivation = activation > 0.0 ? dActivation[row] : 0.0;
        for (uint col = 0u; col < NRC_NN_HIDDEN_WIDTH; ++col) {
            float srcActivation = NrcKhrReadEncodedActivation(encoded, col);
            NrcKhrGradientAtomicAdd(row * NRC_NN_HIDDEN_WIDTH + col,
                                    dPreActivation * srcActivation);
        }
    }
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
