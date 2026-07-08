#pragma once

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

    std::uint64_t trainBatchCount = 0;
    std::uint64_t trainBatchSize = 0;
    std::uint64_t trainBatchCountBytes = 0;
    std::uint64_t trainBatchRecordBytes = 0;
    std::uint64_t trainBatchRecordsTotalBytes = 0;
};

struct PackedInputLayout {
    static constexpr std::uint64_t size = 16;
};

struct EvalRecordLayout {
    static constexpr std::uint64_t size = 20;
};

struct TrainRecordLayout {
    static constexpr std::uint64_t size = 40;
};

struct OptimizerStateLayout {
    static constexpr std::uint64_t size = 20;
};

struct OptimizerEntryLayout {
    static constexpr std::uint64_t size = 16;
};

ResourceLayout compute_resource_layout(Extent2D extent);

} // namespace nrc
