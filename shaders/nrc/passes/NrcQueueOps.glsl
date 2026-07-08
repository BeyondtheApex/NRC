#ifndef NRC_QUEUE_OPS_GLSL
#define NRC_QUEUE_OPS_GLSL

#include "../common/NrcQueue.glsl"

#ifndef NRC_QUEUE_SET
#define NRC_QUEUE_SET 0
#endif

#ifndef NRC_QUEUE_STATE_BINDING
#define NRC_QUEUE_STATE_BINDING 0
#endif

#ifndef NRC_QUEUE_EVAL_RECORDS_BINDING
#define NRC_QUEUE_EVAL_RECORDS_BINDING 1
#endif

#ifndef NRC_QUEUE_TRAIN_RECORDS_BINDING
#define NRC_QUEUE_TRAIN_RECORDS_BINDING 2
#endif

#ifndef NRC_QUEUE_RESET_WORKGROUP_SIZE
#define NRC_QUEUE_RESET_WORKGROUP_SIZE 64
#endif

layout(std430, set = NRC_QUEUE_SET, binding = NRC_QUEUE_STATE_BINDING) buffer NrcQueueStateBuffer {
    uint nrcQueueEvalCount;
    uint nrcQueueTrainCounts[NRC_TRAIN_BATCH_COUNT];
};

layout(std430, set = NRC_QUEUE_SET, binding = NRC_QUEUE_EVAL_RECORDS_BINDING) buffer NrcQueueEvalRecordsBuffer {
    NrcEvalRecord nrcQueueEvalRecords[];
};

layout(std430, set = NRC_QUEUE_SET, binding = NRC_QUEUE_TRAIN_RECORDS_BINDING) buffer NrcQueueTrainRecordsBuffer {
    NrcTrainRecord nrcQueueTrainRecords[];
};

uint NrcTrainFlatIndex(uint batch, uint indexInBatch) {
    return batch * NRC_TRAIN_BATCH_SIZE + indexInBatch;
}

void NrcQueueResetMain() {
    uint index = gl_GlobalInvocationID.x;
    if (index == 0u) {
        nrcQueueEvalCount = 0u;
    }
    if (index < NRC_TRAIN_BATCH_COUNT) {
        nrcQueueTrainCounts[index] = 0u;
    }
}

bool NrcReserveEvalRecord(uint capacity, out uint index) {
    for (;;) {
        uint observed = nrcQueueEvalCount;
        if (observed >= capacity) {
            index = capacity;
            return false;
        }

        uint previous = atomicCompSwap(nrcQueueEvalCount, observed, observed + 1u);
        if (previous == observed) {
            index = observed;
            return true;
        }
    }
}

bool NrcPushEvalRecord(NrcEvalRecord record, uint capacity, out uint index) {
    if (!NrcReserveEvalRecord(capacity, index)) {
        return false;
    }
    nrcQueueEvalRecords[index] = record;
    return true;
}

bool NrcReserveTrainRecords(uint batch, uint requestedCount, out uint firstIndex, out uint reservedCount) {
    firstIndex = NRC_TRAIN_BATCH_SIZE;
    reservedCount = 0u;

    if (batch >= NRC_TRAIN_BATCH_COUNT || requestedCount == 0u) {
        return false;
    }

    for (;;) {
        uint observed = nrcQueueTrainCounts[batch];
        if (observed >= NRC_TRAIN_BATCH_SIZE) {
            return false;
        }

        uint desired = min(observed + requestedCount, NRC_TRAIN_BATCH_SIZE);
        uint previous = atomicCompSwap(nrcQueueTrainCounts[batch], observed, desired);
        if (previous == observed) {
            firstIndex = observed;
            reservedCount = desired - observed;
            return reservedCount != 0u;
        }
    }
}

void NrcStoreTrainRecord(uint batch, uint indexInBatch, NrcTrainRecord record) {
    nrcQueueTrainRecords[NrcTrainFlatIndex(batch, indexInBatch)] = record;
}

void NrcStoreTrainRecords(uint batch, uint firstIndex, uint recordCount, NrcTrainRecord record) {
    for (uint i = 0u; i < recordCount; ++i) {
        NrcStoreTrainRecord(batch, firstIndex + i, record);
    }
}

#endif
