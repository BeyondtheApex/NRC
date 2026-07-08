#ifndef NRC_QUEUE_GLSL
#define NRC_QUEUE_GLSL

#include "NrcEncoding.glsl"

NrcEvalRecord NrcMakeEvalScreen(PackedNrcInput nrcInput, uvec2 pixel) {
    NrcEvalRecord record;
    record.dst = NrcEncodeEvalDstScreen(pixel);
    record.packedInput = nrcInput;
    return record;
}

NrcEvalRecord NrcMakeEvalTrain(PackedNrcInput nrcInput, uint batch2, uint left14, uint right14) {
    NrcEvalRecord record;
    record.dst = NrcEncodeEvalDstTrain(batch2, left14, right14);
    record.packedInput = nrcInput;
    return record;
}

NrcTrainRecord NrcMakeTrainRecord(vec3 bias, vec3 factor, PackedNrcInput nrcInput) {
    NrcTrainRecord record;
    record.biasR = bias.r;
    record.biasG = bias.g;
    record.biasB = bias.b;
    record.factorR = factor.r;
    record.factorG = factor.g;
    record.factorB = factor.b;
    record.packedInput = nrcInput;
    return record;
}

bool NrcIsValidEvalIndex(uint index, uint capacity) {
    return index < capacity;
}

bool NrcIsValidTrainIndex(uint index) {
    return index < NRC_TRAIN_BATCH_SIZE;
}

#endif
