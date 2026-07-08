#ifndef NRC_OPTIMIZE_GLSL
#define NRC_OPTIMIZE_GLSL

#include "../common/NrcTypes.glsl"

#ifndef NRC_OPTIMIZE_SET
#define NRC_OPTIMIZE_SET 0
#endif

#ifndef NRC_OPTIMIZE_WEIGHTS_BINDING
#define NRC_OPTIMIZE_WEIGHTS_BINDING 0
#endif

#ifndef NRC_OPTIMIZE_USE_WEIGHTS_BINDING
#define NRC_OPTIMIZE_USE_WEIGHTS_BINDING 1
#endif

#ifndef NRC_OPTIMIZE_GRADIENTS_BINDING
#define NRC_OPTIMIZE_GRADIENTS_BINDING 2
#endif

#ifndef NRC_OPTIMIZE_ENTRIES_BINDING
#define NRC_OPTIMIZE_ENTRIES_BINDING 3
#endif

#ifndef NRC_OPTIMIZE_QUEUE_STATE_BINDING
#define NRC_OPTIMIZE_QUEUE_STATE_BINDING 4
#endif

#ifndef NRC_OPTIMIZE_STATE_BINDING
#define NRC_OPTIMIZE_STATE_BINDING 5
#endif

#ifndef NRC_OPTIMIZE_WRITE_USE_WEIGHTS
#define NRC_OPTIMIZE_WRITE_USE_WEIGHTS 0
#endif

#ifndef NRC_OPTIMIZE_WORKGROUP_SIZE
#define NRC_OPTIMIZE_WORKGROUP_SIZE 64
#endif

layout(std430, set = NRC_OPTIMIZE_SET, binding = NRC_OPTIMIZE_WEIGHTS_BINDING) writeonly buffer NrcOptimizeWeightsBuffer {
    uvec4 nrcOptimizeWeights[];
};

#if NRC_OPTIMIZE_WRITE_USE_WEIGHTS
layout(std430, set = NRC_OPTIMIZE_SET, binding = NRC_OPTIMIZE_USE_WEIGHTS_BINDING) writeonly buffer NrcOptimizeUseWeightsBuffer {
    uvec4 nrcOptimizeUseWeights[];
};
#endif

layout(std430, set = NRC_OPTIMIZE_SET, binding = NRC_OPTIMIZE_GRADIENTS_BINDING) readonly buffer NrcOptimizeGradientsBuffer {
    float nrcOptimizeGradients[];
};

layout(std430, set = NRC_OPTIMIZE_SET, binding = NRC_OPTIMIZE_ENTRIES_BINDING) buffer NrcOptimizeEntriesBuffer {
    NrcOptimizerEntry nrcOptimizeEntries[];
};

layout(std430, set = NRC_OPTIMIZE_SET, binding = NRC_OPTIMIZE_QUEUE_STATE_BINDING) readonly buffer NrcOptimizeQueueStateBuffer {
    uint nrcOptimizeEvalCount;
    uint nrcOptimizeTrainCounts[NRC_TRAIN_BATCH_COUNT];
};

layout(std430, set = NRC_OPTIMIZE_SET, binding = NRC_OPTIMIZE_STATE_BINDING) readonly buffer NrcOptimizeStateBuffer {
    NrcOptimizerState nrcOptimizeState;
};

layout(push_constant) uniform NrcOptimizeConstants {
    uint batch;
    uint useEmaWeights;
} nrcOptimizeConstants;

uvec4 NrcOptimizePackFp16Block(float values[NRC_FP16_PER_UVEC4]) {
    return uvec4(packHalf2x16(vec2(values[0], values[1])),
                 packHalf2x16(vec2(values[2], values[3])),
                 packHalf2x16(vec2(values[4], values[5])),
                 packHalf2x16(vec2(values[6], values[7])));
}

void NrcOptimizeMain() {
    uint blockIndex = gl_GlobalInvocationID.x;
    if (blockIndex >= NRC_WEIGHT_UVEC4_COUNT) {
        return;
    }

    uint batch = min(nrcOptimizeConstants.batch, NRC_TRAIN_BATCH_COUNT - 1u);
    uint trainCount = min(nrcOptimizeTrainCounts[batch], NRC_TRAIN_BATCH_SIZE);
    if (trainCount == 0u) {
        return;
    }

    float invTrainScale = 1.0 / (float(trainCount) * NRC_LOSS_SCALE);
    float beta1Denom = 1.0 - nrcOptimizeState.beta1T;
    float beta2Denom = 1.0 - nrcOptimizeState.beta2T;
    float etaT = 1.0 - nrcOptimizeState.alphaT;
    float etaTMinus1 = 1.0 - nrcOptimizeState.alphaTMinus1;

    float packedWeights[NRC_FP16_PER_UVEC4];
#if NRC_OPTIMIZE_WRITE_USE_WEIGHTS
    float packedUseWeights[NRC_FP16_PER_UVEC4];
#endif

    for (uint lane = 0u; lane < NRC_FP16_PER_UVEC4; ++lane) {
        uint weightIndex = blockIndex * NRC_FP16_PER_UVEC4 + lane;
        float gradient = nrcOptimizeGradients[weightIndex] * invTrainScale;
        if (isnan(gradient) || isinf(gradient)) {
            gradient = 0.0;
        }

        NrcOptimizerEntry entry = nrcOptimizeEntries[weightIndex];
        entry.moment1 = NRC_ADAM_BETA1 * entry.moment1 + (1.0 - NRC_ADAM_BETA1) * gradient;
        entry.moment2 = NRC_ADAM_BETA2 * entry.moment2 + (1.0 - NRC_ADAM_BETA2) * gradient * gradient;

        float correctedMoment1 = entry.moment1 / beta1Denom;
        float correctedMoment2 = entry.moment2 / beta2Denom;
        entry.weight -= NRC_OPTIMIZER_LEARNING_RATE * correctedMoment1 /
                        (sqrt(correctedMoment2) + NRC_OPTIMIZER_EPSILON);
        entry.emaWeight = (1.0 - NRC_EMA_ALPHA) / etaT * entry.weight +
                          NRC_EMA_ALPHA * etaTMinus1 * entry.emaWeight;

        nrcOptimizeEntries[weightIndex] = entry;
        packedWeights[lane] = entry.weight;
#if NRC_OPTIMIZE_WRITE_USE_WEIGHTS
        packedUseWeights[lane] = nrcOptimizeConstants.useEmaWeights == 1u ? entry.emaWeight : entry.weight;
#endif
    }

    nrcOptimizeWeights[blockIndex] = NrcOptimizePackFp16Block(packedWeights);
#if NRC_OPTIMIZE_WRITE_USE_WEIGHTS
    nrcOptimizeUseWeights[blockIndex] = NrcOptimizePackFp16Block(packedUseWeights);
#endif
}

#endif
