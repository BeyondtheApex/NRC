#ifndef NRC_GRADIENT_KHR_GLSL
#define NRC_GRADIENT_KHR_GLSL

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

#ifndef NRC_NN_DWEIGHTS_BINDING
#define NRC_NN_DWEIGHTS_BINDING NRC_GRADIENT_GRADIENTS_BINDING
#endif

#ifndef NRC_KHR_WORKGROUP_SIZE
#define NRC_KHR_WORKGROUP_SIZE NRC_GRADIENT_WORKGROUP_SIZE
#endif

#if NRC_GRADIENT_WORKGROUP_SIZE != NRC_KHR_WORKGROUP_SIZE
#error NRC KHR gradient pass requires NRC_GRADIENT_WORKGROUP_SIZE and NRC_KHR_WORKGROUP_SIZE to match.
#endif

#ifndef NRC_NN_BACKPROPAGATION
#define NRC_NN_BACKPROPAGATION
#endif
#include "../backend/khr_coopmat/NrcNNKhr.glsl"

layout(std430, set = NRC_GRADIENT_SET, binding = NRC_GRADIENT_TRAIN_RECORDS_BINDING) readonly buffer NrcGradientKhrTrainRecordsBuffer {
    NrcTrainRecord nrcGradientKhrTrainRecords[];
};

layout(std430, set = NRC_GRADIENT_SET, binding = NRC_GRADIENT_QUEUE_STATE_BINDING) readonly buffer NrcGradientKhrQueueStateBuffer {
    uint nrcGradientKhrEvalCount;
    uint nrcGradientKhrTrainCounts[NRC_TRAIN_BATCH_COUNT];
};

layout(push_constant) uniform NrcGradientKhrConstants {
    uint batch;
} nrcGradientKhrConstants;

void NrcGradientKhrMain() {
    uint batch = min(nrcGradientKhrConstants.batch, NRC_TRAIN_BATCH_COUNT - 1u);
    uint trainCount = min(nrcGradientKhrTrainCounts[batch], NRC_TRAIN_BATCH_SIZE);
    uint indexInBatch = gl_GlobalInvocationID.x;
    bool validTrainRecord = indexInBatch < trainCount;

    uvec4 encoded[NRC_ENCODED_INPUT_UVEC4_COUNT];
    vec3 target = vec3(0.0);
    if (validTrainRecord) {
        uint trainIndex = batch * NRC_TRAIN_BATCH_SIZE + indexInBatch;
        NrcTrainRecord trainRecord = nrcGradientKhrTrainRecords[trainIndex];
        NrcEncodeInput(NrcUnpackPackedInput(trainRecord.packedInput), encoded);
        target = vec3(trainRecord.biasR, trainRecord.biasG, trainRecord.biasB);
    } else {
        for (uint i = 0u; i < NRC_ENCODED_INPUT_UVEC4_COUNT; ++i) {
            encoded[i] = uvec4(0u);
        }
    }

    NrcKhrLoadInput(encoded);
    for (uint layer = 0u; layer < NRC_NN_HIDDEN_LAYER_COUNT - 4u; ++layer) {
        NrcKhrLoadHiddenWeights(layer);
        NrcKhrForwardHiddenRelu();
    }
    NrcKhrStoreBackpropOldestSourceActivations();

    NrcKhrLoadHiddenWeights(NRC_NN_HIDDEN_LAYER_COUNT - 4u);
    NrcKhrForwardHiddenRelu();
    NrcKhrStoreBackpropEarlierSourceActivations();

    NrcKhrLoadHiddenWeights(NRC_NN_HIDDEN_LAYER_COUNT - 3u);
    NrcKhrForwardHiddenRelu();
    NrcKhrStoreBackpropPreviousSourceActivations();

    NrcKhrLoadHiddenWeights(NRC_NN_HIDDEN_LAYER_COUNT - 2u);
    NrcKhrForwardHiddenRelu();
    NrcKhrStoreBackpropSourceActivations();

    NrcKhrLoadHiddenWeights(NRC_NN_HIDDEN_LAYER_COUNT - 1u);
    NrcKhrForwardHiddenRelu();
    NrcKhrStoreBackpropActivations();

    NrcKhrLoadOutputWeights(NRC_NN_HIDDEN_LAYER_COUNT);
    NrcKhrForwardOutputStore();
    vec3 prediction = NrcKhrReadInvocationOutput();

    float dOutput[NRC_NN_OUTPUT_WIDTH];
    for (uint i = 0u; i < NRC_NN_OUTPUT_WIDTH; ++i) {
        dOutput[i] = 0.0;
    }
    if (validTrainRecord) {
        NrcKhrOutputDerivative(prediction, target, dOutput);
    }
    NrcKhrAccumulateOutputDWeightsWorkgroup(dOutput);

    float dHidden4[NRC_NN_HIDDEN_WIDTH];
    NrcKhrPropagateOutputDActivation(dOutput, dHidden4);
    NrcKhrAccumulateFinalHiddenDWeightsWorkgroup(NRC_NN_HIDDEN_LAYER_COUNT - 1u,
                                                 NRC_KHR_BACKPROP_ACT_COPY_UVEC4_BASE,
                                                 NRC_KHR_BACKPROP_SOURCE_ACT_COPY_UVEC4_BASE);

    float dHidden3[NRC_NN_HIDDEN_WIDTH];
    NrcKhrPropagateHiddenDActivation(NRC_NN_HIDDEN_LAYER_COUNT - 1u,
                                     NRC_KHR_BACKPROP_ACT_COPY_UVEC4_BASE,
                                     dHidden4,
                                     dHidden3);
    NrcKhrStoreSharedDPreActivations(NRC_KHR_BACKPROP_SOURCE_ACT_COPY_UVEC4_BASE,
                                     dHidden3);
    NrcKhrAccumulateSharedHiddenDWeightsWorkgroup(NRC_NN_HIDDEN_LAYER_COUNT - 2u,
                                                  NRC_KHR_BACKPROP_PREVIOUS_SOURCE_ACT_COPY_UVEC4_BASE);

    float dHidden2[NRC_NN_HIDDEN_WIDTH];
    NrcKhrPropagateHiddenDActivation(NRC_NN_HIDDEN_LAYER_COUNT - 2u,
                                     NRC_KHR_BACKPROP_SOURCE_ACT_COPY_UVEC4_BASE,
                                     dHidden3,
                                     dHidden2);
    NrcKhrStoreSharedDPreActivations(NRC_KHR_BACKPROP_PREVIOUS_SOURCE_ACT_COPY_UVEC4_BASE,
                                     dHidden2);
    NrcKhrAccumulateSharedHiddenDWeightsWorkgroup(NRC_NN_HIDDEN_LAYER_COUNT - 3u,
                                                  NRC_KHR_BACKPROP_EARLIER_SOURCE_ACT_COPY_UVEC4_BASE);

    float dHidden1[NRC_NN_HIDDEN_WIDTH];
    NrcKhrPropagateHiddenDActivation(NRC_NN_HIDDEN_LAYER_COUNT - 3u,
                                     NRC_KHR_BACKPROP_PREVIOUS_SOURCE_ACT_COPY_UVEC4_BASE,
                                     dHidden2,
                                     dHidden1);
    NrcKhrStoreSharedDPreActivations(NRC_KHR_BACKPROP_EARLIER_SOURCE_ACT_COPY_UVEC4_BASE,
                                     dHidden1);
    NrcKhrAccumulateSharedHiddenDWeightsWorkgroup(NRC_NN_HIDDEN_LAYER_COUNT - 4u,
                                                  NRC_KHR_BACKPROP_OLDEST_SOURCE_ACT_COPY_UVEC4_BASE);

    float dHidden0[NRC_NN_HIDDEN_WIDTH];
    NrcKhrPropagateHiddenDActivation(NRC_NN_HIDDEN_LAYER_COUNT - 4u,
                                     NRC_KHR_BACKPROP_EARLIER_SOURCE_ACT_COPY_UVEC4_BASE,
                                     dHidden1,
                                     dHidden0);
    NrcKhrAccumulateInputHiddenDWeights(dHidden0, encoded);
}

#endif
