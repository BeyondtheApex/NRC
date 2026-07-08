#include "nrc/nrc_resource_layout.hpp"

#include "nrc/nrc_constants.hpp"

namespace nrc {

ResourceLayout compute_resource_layout(Extent2D extent) {
    const std::uint64_t pixelCount =
        static_cast<std::uint64_t>(extent.width) * static_cast<std::uint64_t>(extent.height);
    const std::uint64_t batchCount = kTrainBatchCount;
    const std::uint64_t batchSize = kTrainBatchSize;
    const std::uint64_t trainCapacity = batchCount * batchSize;
    const std::uint64_t weights = weight_count();

    ResourceLayout layout{};
    layout.weightCount = weights;
    layout.weightsBytes = weights * sizeof(std::uint16_t);
    layout.useWeightsBytes = layout.weightsBytes;
    layout.optimizerStateBytes = OptimizerStateLayout::size;
    layout.optimizerEntriesBytes = weights * OptimizerEntryLayout::size;
    layout.gradientsBytes = weights * sizeof(float);

    layout.evalRecordCount = pixelCount + trainCapacity;
    layout.evalRecordsBytes = layout.evalRecordCount * EvalRecordLayout::size;
    layout.evalCountBytes = sizeof(std::uint32_t);
    layout.queueStateBytes = QueueStateLayout::size;

    layout.trainBatchCount = batchCount;
    layout.trainBatchSize = batchSize;
    layout.trainBatchCountBytes = batchCount * sizeof(std::uint32_t);
    layout.trainBatchRecordBytes = batchSize * TrainRecordLayout::size;
    layout.trainBatchRecordsTotalBytes = batchCount * layout.trainBatchRecordBytes;
    return layout;
}

} // namespace nrc
