#pragma once

#include "nrc/nrc_profile.hpp"

#include <cstdint>

namespace nrc {

struct KhrSharedMemoryRegion {
    std::uint32_t baseUvec4 = 0;
    std::uint32_t countUvec4 = 0;

    constexpr std::uint32_t end_uvec4() const {
        return baseUvec4 + countUvec4;
    }

    constexpr bool contains(const KhrSharedMemoryRegion &other) const {
        return other.baseUvec4 >= baseUvec4 && other.end_uvec4() <= end_uvec4();
    }

    constexpr bool overlaps(const KhrSharedMemoryRegion &other) const {
        return baseUvec4 < other.end_uvec4() && other.baseUvec4 < end_uvec4();
    }
};

struct KhrHiddenDWeightTileStreamingLayout {
    bool valid = false;
    std::uint32_t rowBlockCount = 0;
    std::uint32_t columnBlockCount = 0;
    std::uint32_t tileCount = 0;
    std::uint32_t dPreRowScratchUvec4Count = 0;
    std::uint32_t partialTileUvec4Count = 0;
    std::uint32_t partialScratchUvec4Count = 0;
    std::uint32_t totalScratchUvec4Count = 0;

    KhrSharedMemoryRegion dPreRowScratch;
    KhrSharedMemoryRegion partialScratch;
    KhrSharedMemoryRegion combinedScratch;

    bool fitsActiveTile = false;
    bool preservesSharedActivationSnapshots = false;
};

struct KhrShaderSharedMemoryLayout {
    bool valid = false;
    std::uint32_t subgroupCount = 0;
    std::uint32_t activeTileUvec4Count = 0;
    std::uint32_t activationSnapshotCount = 0;
    std::uint32_t activationCopySharedUvec4Count = 0;
    std::uint32_t reductionSharedUvec4Count = 0;
    std::uint32_t sharedUvec4Count = 0;
    std::uint32_t sharedBytes = 0;

    KhrSharedMemoryRegion activeTile;
    KhrSharedMemoryRegion finalHiddenSnapshot;
    KhrSharedMemoryRegion outputSourceSnapshot;
    KhrSharedMemoryRegion previousSourceSnapshot;
    KhrSharedMemoryRegion earlierSourceSnapshot;
    KhrSharedMemoryRegion oldestSourceSnapshot;

    KhrSharedMemoryRegion dPreScratch;
    KhrSharedMemoryRegion outputDerivativeFp16Scratch;
    KhrSharedMemoryRegion outputDWeightPartialScratch;
    KhrSharedMemoryRegion hiddenDWeightPartialScratch;
    KhrHiddenDWeightTileStreamingLayout hiddenDWeightTileStreaming;

    bool dPreScratchFitsActivePlusFinalSnapshot = false;
    bool outputDerivativeFp16FitsActiveTile = false;
    bool outputDWeightPartialFitsActiveTile = false;
    bool hiddenDWeightPartialFitsSharedAllocation = false;
    bool hiddenDWeightPartialPreservesSharedHiddenActivationSnapshots = false;
};

KhrShaderSharedMemoryLayout compute_khr_shader_shared_memory_layout(
    const ProfileShaderConfig &config);

} // namespace nrc
