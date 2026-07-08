#ifndef NRC_GRADIENT_SCALAR_GLSL
#define NRC_GRADIENT_SCALAR_GLSL

#extension GL_EXT_shader_atomic_float : require

#ifndef NRC_GRADIENT_SET
#define NRC_GRADIENT_SET 0
#endif

#ifndef NRC_GRADIENT_TRAIN_RECORDS_BINDING
#define NRC_GRADIENT_TRAIN_RECORDS_BINDING 0
#endif

#ifndef NRC_GRADIENT_QUEUE_STATE_BINDING
#define NRC_GRADIENT_QUEUE_STATE_BINDING 1
#endif

#ifndef NRC_GRADIENT_WEIGHTS_BINDING
#define NRC_GRADIENT_WEIGHTS_BINDING 2
#endif

#ifndef NRC_GRADIENT_GRADIENTS_BINDING
#define NRC_GRADIENT_GRADIENTS_BINDING 3
#endif

#ifndef NRC_GRADIENT_WORKGROUP_SIZE
#define NRC_GRADIENT_WORKGROUP_SIZE 128
#endif

#ifndef NRC_NN_SET
#define NRC_NN_SET NRC_GRADIENT_SET
#endif

#ifndef NRC_NN_WEIGHTS_BINDING
#define NRC_NN_WEIGHTS_BINDING NRC_GRADIENT_WEIGHTS_BINDING
#endif

#include "../backend/scalar/NrcNNScalar.glsl"

layout(std430, set = NRC_GRADIENT_SET, binding = NRC_GRADIENT_TRAIN_RECORDS_BINDING) readonly buffer NrcGradientTrainRecordsBuffer {
    NrcTrainRecord nrcGradientTrainRecords[];
};

layout(std430, set = NRC_GRADIENT_SET, binding = NRC_GRADIENT_QUEUE_STATE_BINDING) readonly buffer NrcGradientQueueStateBuffer {
    uint nrcGradientEvalCount;
    uint nrcGradientTrainCounts[NRC_TRAIN_BATCH_COUNT];
};

layout(std430, set = NRC_GRADIENT_SET, binding = NRC_GRADIENT_GRADIENTS_BINDING) buffer NrcGradientGradientsBuffer {
    float nrcGradientGradients[];
};

layout(push_constant) uniform NrcGradientConstants {
    uint batch;
} nrcGradientConstants;

void NrcGradientAtomicAdd(uint index, float value) {
    atomicAdd(nrcGradientGradients[index], value);
}

void NrcGradientOutputDerivative(vec3 prediction, vec3 target, out float dOutput[NRC_NN_OUTPUT_WIDTH]) {
    float luminance = dot(vec3(0.299, 0.587, 0.114), max(prediction, vec3(0.0)));
    float denominator = luminance * luminance + 0.01;
    vec3 dLoss = 2.0 * NRC_LOSS_SCALE * (prediction - target) / denominator;
    dOutput[0] = dLoss.x;
    dOutput[1] = dLoss.y;
    dOutput[2] = dLoss.z;
}

void NrcGradientScalarMain() {
    uint batch = min(nrcGradientConstants.batch, NRC_TRAIN_BATCH_COUNT - 1u);
    uint trainCount = min(nrcGradientTrainCounts[batch], NRC_TRAIN_BATCH_SIZE);
    uint indexInBatch = gl_GlobalInvocationID.x;
    if (indexInBatch >= trainCount) {
        return;
    }

    uint trainIndex = batch * NRC_TRAIN_BATCH_SIZE + indexInBatch;
    NrcTrainRecord trainRecord = nrcGradientTrainRecords[trainIndex];

    uvec4 encoded[NRC_ENCODED_INPUT_UVEC4_COUNT];
    NrcEncodeInput(NrcUnpackPackedInput(trainRecord.packedInput), encoded);

    float activations0[NRC_NN_HIDDEN_WIDTH];
    float activations1[NRC_NN_HIDDEN_WIDTH];
    float activations2[NRC_NN_HIDDEN_WIDTH];
    float activations3[NRC_NN_HIDDEN_WIDTH];
    float activations4[NRC_NN_HIDDEN_WIDTH];
    float activations5[NRC_NN_HIDDEN_WIDTH];
    float preActivations0[NRC_NN_HIDDEN_WIDTH];
    float preActivations1[NRC_NN_HIDDEN_WIDTH];
    float preActivations2[NRC_NN_HIDDEN_WIDTH];
    float preActivations3[NRC_NN_HIDDEN_WIDTH];
    float preActivations4[NRC_NN_HIDDEN_WIDTH];

    NrcScalarUnpackInput(encoded, activations0);

#define NRC_GRADIENT_FORWARD_LAYER(layerIndex, srcValues, dstValues, preValues) \
    { \
        uint offset = NrcScalarLayerWeightOffset(layerIndex); \
        for (uint row = 0u; row < NRC_NN_HIDDEN_WIDTH; ++row) { \
            float value = 0.0; \
            for (uint col = 0u; col < NRC_NN_HIDDEN_WIDTH; ++col) { \
                value += NrcScalarLoadWeight(offset + row * NRC_NN_HIDDEN_WIDTH + col) * srcValues[col]; \
            } \
            preValues[row] = value; \
            dstValues[row] = max(value, 0.0); \
        } \
    }

    NRC_GRADIENT_FORWARD_LAYER(0u, activations0, activations1, preActivations0)
    NRC_GRADIENT_FORWARD_LAYER(1u, activations1, activations2, preActivations1)
    NRC_GRADIENT_FORWARD_LAYER(2u, activations2, activations3, preActivations2)
    NRC_GRADIENT_FORWARD_LAYER(3u, activations3, activations4, preActivations3)
    NRC_GRADIENT_FORWARD_LAYER(4u, activations4, activations5, preActivations4)

#undef NRC_GRADIENT_FORWARD_LAYER

    vec3 prediction = vec3(0.0);
    uint outputOffset = NrcScalarLayerWeightOffset(NRC_NN_HIDDEN_LAYER_COUNT);
    for (uint row = 0u; row < NRC_NN_OUTPUT_WIDTH; ++row) {
        float value = 0.0;
        for (uint col = 0u; col < NRC_NN_HIDDEN_WIDTH; ++col) {
            value += NrcScalarLoadWeight(outputOffset + row * NRC_NN_HIDDEN_WIDTH + col) * activations5[col];
        }
        prediction[row] = value;
    }

    float dOutput[NRC_NN_OUTPUT_WIDTH];
    NrcGradientOutputDerivative(prediction,
                                vec3(trainRecord.biasR, trainRecord.biasG, trainRecord.biasB),
                                dOutput);

    for (uint row = 0u; row < NRC_NN_OUTPUT_WIDTH; ++row) {
        for (uint col = 0u; col < NRC_NN_HIDDEN_WIDTH; ++col) {
            NrcGradientAtomicAdd(outputOffset + row * NRC_NN_HIDDEN_WIDTH + col,
                                 dOutput[row] * activations5[col]);
        }
    }

    float dActivation5[NRC_NN_HIDDEN_WIDTH];
    float dActivation4[NRC_NN_HIDDEN_WIDTH];
    float dActivation3[NRC_NN_HIDDEN_WIDTH];
    float dActivation2[NRC_NN_HIDDEN_WIDTH];
    float dActivation1[NRC_NN_HIDDEN_WIDTH];
    float dActivation0[NRC_NN_HIDDEN_WIDTH];

    for (uint col = 0u; col < NRC_NN_HIDDEN_WIDTH; ++col) {
        float value = 0.0;
        for (uint row = 0u; row < NRC_NN_OUTPUT_WIDTH; ++row) {
            value += NrcScalarLoadWeight(outputOffset + row * NRC_NN_HIDDEN_WIDTH + col) * dOutput[row];
        }
        dActivation5[col] = value;
    }

#define NRC_GRADIENT_BACKWARD_LAYER(layerIndex, srcDActivation, dstDActivation, preValues, srcActivations) \
    { \
        uint offset = NrcScalarLayerWeightOffset(layerIndex); \
        for (uint col = 0u; col < NRC_NN_HIDDEN_WIDTH; ++col) { \
            dstDActivation[col] = 0.0; \
        } \
        for (uint row = 0u; row < NRC_NN_HIDDEN_WIDTH; ++row) { \
            float dPreActivation = preValues[row] > 0.0 ? srcDActivation[row] : 0.0; \
            for (uint col = 0u; col < NRC_NN_HIDDEN_WIDTH; ++col) { \
                uint weightIndex = offset + row * NRC_NN_HIDDEN_WIDTH + col; \
                NrcGradientAtomicAdd(weightIndex, dPreActivation * srcActivations[col]); \
                dstDActivation[col] += NrcScalarLoadWeight(weightIndex) * dPreActivation; \
            } \
        } \
    }

    NRC_GRADIENT_BACKWARD_LAYER(4u, dActivation5, dActivation4, preActivations4, activations4)
    NRC_GRADIENT_BACKWARD_LAYER(3u, dActivation4, dActivation3, preActivations3, activations3)
    NRC_GRADIENT_BACKWARD_LAYER(2u, dActivation3, dActivation2, preActivations2, activations2)
    NRC_GRADIENT_BACKWARD_LAYER(1u, dActivation2, dActivation1, preActivations1, activations1)
    NRC_GRADIENT_BACKWARD_LAYER(0u, dActivation1, dActivation0, preActivations0, activations0)

#undef NRC_GRADIENT_BACKWARD_LAYER
}

#endif
