#ifndef NRC_INFERENCE_SCALAR_GLSL
#define NRC_INFERENCE_SCALAR_GLSL

#include "../backend/scalar/NrcNNScalar.glsl"

#ifndef NRC_INFERENCE_SET
#define NRC_INFERENCE_SET 0
#endif

#ifndef NRC_INFERENCE_EVAL_RECORDS_BINDING
#define NRC_INFERENCE_EVAL_RECORDS_BINDING 1
#endif

#ifndef NRC_INFERENCE_EVAL_COUNT_BINDING
#define NRC_INFERENCE_EVAL_COUNT_BINDING 2
#endif

#ifndef NRC_INFERENCE_BIAS_FACTOR_R_BINDING
#define NRC_INFERENCE_BIAS_FACTOR_R_BINDING 3
#endif

#ifndef NRC_INFERENCE_FACTOR_GB_BINDING
#define NRC_INFERENCE_FACTOR_GB_BINDING 4
#endif

#ifndef NRC_INFERENCE_TRAIN_RECORDS_BINDING
#define NRC_INFERENCE_TRAIN_RECORDS_BINDING 5
#endif

#ifndef NRC_INFERENCE_WORKGROUP_SIZE
#define NRC_INFERENCE_WORKGROUP_SIZE 64
#endif

layout(std430, set = NRC_INFERENCE_SET, binding = NRC_INFERENCE_EVAL_RECORDS_BINDING) readonly buffer NrcInferenceEvalRecordsBuffer {
    NrcEvalRecord nrcInferenceEvalRecords[];
};

layout(std430, set = NRC_INFERENCE_SET, binding = NRC_INFERENCE_EVAL_COUNT_BINDING) readonly buffer NrcInferenceEvalCountBuffer {
    uint nrcInferenceEvalCount;
};

layout(set = NRC_INFERENCE_SET, binding = NRC_INFERENCE_BIAS_FACTOR_R_BINDING, rgba32f) uniform image2D nrcInferenceBiasFactorR;
layout(set = NRC_INFERENCE_SET, binding = NRC_INFERENCE_FACTOR_GB_BINDING, rg32f) readonly uniform image2D nrcInferenceFactorGB;

layout(std430, set = NRC_INFERENCE_SET, binding = NRC_INFERENCE_TRAIN_RECORDS_BINDING) buffer NrcInferenceTrainRecordsBuffer {
    NrcTrainRecord nrcInferenceTrainRecords[];
};

void NrcInferenceApplyScreen(uint encodedDst, vec3 prediction) {
    ivec2 coord = ivec2(NrcDecodeEvalDstScreen(encodedDst));
    vec4 biasFactorR = imageLoad(nrcInferenceBiasFactorR, coord);
    vec2 factorGb = imageLoad(nrcInferenceFactorGB, coord).rg;
    vec3 color = biasFactorR.rgb + vec3(biasFactorR.a, factorGb) * prediction;
    imageStore(nrcInferenceBiasFactorR, coord, vec4(color, 0.0));
}

void NrcInferenceApplyTrain(uint encodedDst, vec3 prediction) {
    uint batch;
    uint left;
    uint right;
    NrcDecodeEvalDstTrain(encodedDst, batch, left, right);

    if (batch >= NRC_TRAIN_BATCH_COUNT || left >= NRC_TRAIN_BATCH_SIZE) {
        return;
    }

    right = min(right, NRC_TRAIN_BATCH_SIZE - 1u);
    for (uint i = left; i <= right; ++i) {
        uint trainIndex = batch * NRC_TRAIN_BATCH_SIZE + i;
        NrcTrainRecord trainRecord = nrcInferenceTrainRecords[trainIndex];
        vec3 bias = vec3(trainRecord.biasR, trainRecord.biasG, trainRecord.biasB);
        vec3 factor = vec3(trainRecord.factorR, trainRecord.factorG, trainRecord.factorB);
        vec3 color = bias + factor * prediction;
        nrcInferenceTrainRecords[trainIndex].biasR = color.r;
        nrcInferenceTrainRecords[trainIndex].biasG = color.g;
        nrcInferenceTrainRecords[trainIndex].biasB = color.b;
    }
}

void NrcInferenceScalarMain() {
    uint evalIndex = gl_GlobalInvocationID.x;
    if (evalIndex >= nrcInferenceEvalCount) {
        return;
    }

    NrcEvalRecord evalRecord = nrcInferenceEvalRecords[evalIndex];
    if (evalRecord.dst == NRC_EVAL_INVALID_DST) {
        return;
    }

    NrcUnpackedInput unpackedInput = NrcUnpackPackedInput(evalRecord.packedInput);
    uvec4 encoded[NRC_ENCODED_INPUT_UVEC4_COUNT];
    NrcEncodeInput(unpackedInput, encoded);

    vec3 prediction;
    NrcScalarForward(encoded, prediction);
    prediction = max(prediction, vec3(0.0));

    if (NrcEvalDstType(evalRecord.dst) == NRC_EVAL_DST_SCREEN) {
        NrcInferenceApplyScreen(evalRecord.dst, prediction);
    } else {
        NrcInferenceApplyTrain(evalRecord.dst, prediction);
    }
}

#endif
