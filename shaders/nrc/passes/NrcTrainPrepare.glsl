#ifndef NRC_TRAIN_PREPARE_GLSL
#define NRC_TRAIN_PREPARE_GLSL

#include "../common/NrcTypes.glsl"

#ifndef NRC_TRAIN_PREPARE_SET
#define NRC_TRAIN_PREPARE_SET 0
#endif

#ifndef NRC_TRAIN_PREPARE_QUEUE_STATE_BINDING
#define NRC_TRAIN_PREPARE_QUEUE_STATE_BINDING 0
#endif

#ifndef NRC_TRAIN_PREPARE_INDIRECT_BINDING
#define NRC_TRAIN_PREPARE_INDIRECT_BINDING 1
#endif

#ifndef NRC_TRAIN_PREPARE_OPTIMIZER_STATE_BINDING
#define NRC_TRAIN_PREPARE_OPTIMIZER_STATE_BINDING 2
#endif

#ifndef NRC_GRADIENT_WORKGROUP_SIZE
#define NRC_GRADIENT_WORKGROUP_SIZE 128
#endif

struct NrcDispatchIndirectCommand {
    uint x;
    uint y;
    uint z;
};

layout(std430, set = NRC_TRAIN_PREPARE_SET, binding = NRC_TRAIN_PREPARE_QUEUE_STATE_BINDING) buffer NrcTrainPrepareQueueStateBuffer {
    uint nrcTrainPrepareEvalCount;
    uint nrcTrainPrepareTrainCounts[NRC_TRAIN_BATCH_COUNT];
};

layout(std430, set = NRC_TRAIN_PREPARE_SET, binding = NRC_TRAIN_PREPARE_INDIRECT_BINDING) writeonly buffer NrcTrainPrepareIndirectBuffer {
    NrcDispatchIndirectCommand nrcTrainPrepareIndirectCommand;
};

layout(std430, set = NRC_TRAIN_PREPARE_SET, binding = NRC_TRAIN_PREPARE_OPTIMIZER_STATE_BINDING) buffer NrcTrainPrepareOptimizerStateBuffer {
    NrcOptimizerState nrcTrainPrepareOptimizerState;
};

layout(push_constant) uniform NrcTrainPrepareConstants {
    uint batch;
} nrcTrainPrepareConstants;

void NrcTrainPrepareMain() {
    uint batch = min(nrcTrainPrepareConstants.batch, NRC_TRAIN_BATCH_COUNT - 1u);
    uint count = min(nrcTrainPrepareTrainCounts[batch], NRC_TRAIN_BATCH_SIZE);
    nrcTrainPrepareTrainCounts[batch] = count;
    nrcTrainPrepareIndirectCommand = NrcDispatchIndirectCommand(
        (count + NRC_GRADIENT_WORKGROUP_SIZE - 1u) / NRC_GRADIENT_WORKGROUP_SIZE,
        1u,
        1u);

    if (count != 0u) {
        nrcTrainPrepareOptimizerState.step += 1u;
        nrcTrainPrepareOptimizerState.beta1T *= NRC_ADAM_BETA1;
        nrcTrainPrepareOptimizerState.beta2T *= NRC_ADAM_BETA2;
        nrcTrainPrepareOptimizerState.alphaTMinus1 = nrcTrainPrepareOptimizerState.alphaT;
        nrcTrainPrepareOptimizerState.alphaT *= NRC_EMA_ALPHA;
    }
}

#endif
