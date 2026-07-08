#pragma once

#include "nrc/nrc_records.hpp"

#include <cstdint>

namespace nrc {

struct Extent2D {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

struct ResourceLayout {
    std::uint64_t weightCount = 0;
    std::uint64_t weightsBytes = 0;
    std::uint64_t useWeightsBytes = 0;
    std::uint64_t optimizerStateBytes = 0;
    std::uint64_t optimizerEntriesBytes = 0;
    std::uint64_t gradientsBytes = 0;

    std::uint64_t evalRecordCount = 0;
    std::uint64_t evalRecordsBytes = 0;
    std::uint64_t evalCountBytes = 0;
    std::uint64_t queueStateBytes = 0;

    std::uint64_t trainBatchCount = 0;
    std::uint64_t trainBatchSize = 0;
    std::uint64_t trainBatchCountBytes = 0;
    std::uint64_t trainBatchRecordBytes = 0;
    std::uint64_t trainBatchRecordsTotalBytes = 0;
};

struct PackedInputLayout {
    static constexpr std::uint64_t size = sizeof(PackedNrcInput);
};

struct EvalRecordLayout {
    static constexpr std::uint64_t size = sizeof(NrcEvalRecord);
};

struct TrainRecordLayout {
    static constexpr std::uint64_t size = sizeof(NrcTrainRecord);
};

struct OptimizerStateLayout {
    static constexpr std::uint64_t size = sizeof(NrcOptimizerState);
};

struct OptimizerEntryLayout {
    static constexpr std::uint64_t size = sizeof(NrcOptimizerEntry);
};

struct QueueStateLayout {
    static constexpr std::uint64_t size = sizeof(NrcQueueState);
};

ResourceLayout compute_resource_layout(Extent2D extent);

} // namespace nrc
