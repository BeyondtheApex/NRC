#include "nrc/nrc_khr_shader_layout.hpp"

#include "nrc/nrc_constants.hpp"

#include <algorithm>

namespace nrc {
namespace {

constexpr std::uint32_t kUvec4Bytes = 16;
constexpr std::uint32_t kFp32PerUvec4 = 4;
constexpr std::uint32_t kBackpropActivationSnapshotCount = 6;

bool is_valid_khr_config(const ProfileShaderConfig &config) {
    if (config.backend != BackendKind::KhrCooperativeMatrix) {
        return false;
    }
    if (config.workgroupSize == 0 || config.subgroupSize == 0) {
        return false;
    }
    if ((config.workgroupSize % config.subgroupSize) != 0) {
        return false;
    }
    if (config.cooperativeMatrixM == 0 ||
        config.cooperativeMatrixN == 0 ||
        config.cooperativeMatrixK == 0) {
        return false;
    }
    if ((kNnHiddenWidth % config.cooperativeMatrixM) != 0 ||
        (kNnHiddenWidth % config.cooperativeMatrixK) != 0 ||
        (config.cooperativeMatrixM % kFp16PerUvec4) != 0 ||
        (config.cooperativeMatrixN % kFp16PerUvec4) != 0) {
        return false;
    }
    return true;
}

} // namespace

KhrShaderSharedMemoryLayout compute_khr_shader_shared_memory_layout(
    const ProfileShaderConfig &config) {
    KhrShaderSharedMemoryLayout layout{};
    if (!is_valid_khr_config(config)) {
        return layout;
    }

    layout.valid = true;
    layout.subgroupCount = config.workgroupSize / config.subgroupSize;
    layout.activeTileUvec4Count =
        (kNnHiddenWidth * config.workgroupSize) / kFp16PerUvec4;
    layout.activationSnapshotCount = kBackpropActivationSnapshotCount;

    layout.activeTile = {0, layout.activeTileUvec4Count};
    layout.finalHiddenSnapshot = {layout.activeTileUvec4Count, layout.activeTileUvec4Count};
    layout.outputSourceSnapshot = {layout.activeTileUvec4Count * 2u, layout.activeTileUvec4Count};
    layout.previousSourceSnapshot = {layout.activeTileUvec4Count * 3u, layout.activeTileUvec4Count};
    layout.earlierSourceSnapshot = {layout.activeTileUvec4Count * 4u, layout.activeTileUvec4Count};
    layout.oldestSourceSnapshot = {layout.activeTileUvec4Count * 5u, layout.activeTileUvec4Count};

    layout.activationCopySharedUvec4Count =
        layout.activeTileUvec4Count * layout.activationSnapshotCount;
    layout.reductionSharedUvec4Count =
        (hidden_layer_weight_count() / kFp16PerUvec4) * layout.subgroupCount;
    layout.sharedUvec4Count = std::max(layout.activeTileUvec4Count,
                                       std::max(layout.activationCopySharedUvec4Count,
                                                layout.reductionSharedUvec4Count));
    layout.sharedBytes = layout.sharedUvec4Count * kUvec4Bytes;

    const std::uint32_t dPreUvec4PerSample = kNnHiddenWidth / kFp32PerUvec4;
    layout.dPreScratch = {0, config.workgroupSize * dPreUvec4PerSample};

    const std::uint32_t outputDerivativeUvec4PerSample =
        config.cooperativeMatrixN / kFp16PerUvec4;
    layout.outputDerivativeFp16Scratch = {
        0,
        config.workgroupSize * outputDerivativeUvec4PerSample,
    };

    const std::uint32_t outputDWeightPartialSubgroupUvec4Count =
        (kNnHiddenWidth * config.cooperativeMatrixN) / kFp16PerUvec4;
    layout.outputDWeightPartialScratch = {
        0,
        outputDWeightPartialSubgroupUvec4Count * layout.subgroupCount,
    };

    layout.hiddenDWeightPartialScratch = {0, layout.reductionSharedUvec4Count};

    const KhrSharedMemoryRegion activePlusFinalHidden{
        layout.activeTile.baseUvec4,
        layout.activeTileUvec4Count * 2u,
    };
    const KhrSharedMemoryRegion wholeShared{
        0,
        layout.sharedUvec4Count,
    };

    layout.dPreScratchFitsActivePlusFinalSnapshot =
        activePlusFinalHidden.contains(layout.dPreScratch);
    layout.outputDerivativeFp16FitsActiveTile =
        layout.activeTile.contains(layout.outputDerivativeFp16Scratch);
    layout.outputDWeightPartialFitsActiveTile =
        layout.activeTile.contains(layout.outputDWeightPartialScratch);
    layout.hiddenDWeightPartialFitsSharedAllocation =
        wholeShared.contains(layout.hiddenDWeightPartialScratch);
    layout.hiddenDWeightPartialPreservesSharedHiddenActivationSnapshots =
        layout.hiddenDWeightPartialScratch.end_uvec4() <= layout.outputSourceSnapshot.baseUvec4;

    return layout;
}

} // namespace nrc
