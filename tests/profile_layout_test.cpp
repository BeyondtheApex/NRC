#include "nrc/nrc_constants.hpp"
#include "nrc/nrc_profile.hpp"
#include "nrc/nrc_records.hpp"
#include "nrc/nrc_resource_layout.hpp"

#include <array>
#include <cstdlib>
#include <iostream>

namespace {

bool require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }
    return true;
}

} // namespace

int main() {
    bool ok = true;

    ok &= require(nrc::weight_count() == 20672u, "VkNRC-compatible weight count");
    ok &= require(nrc::PackedInputLayout::size == 16u, "packed input layout size");
    ok &= require(nrc::EvalRecordLayout::size == 20u, "eval record layout size");
    ok &= require(nrc::TrainRecordLayout::size == 40u, "train record layout size");
    ok &= require(nrc::OptimizerStateLayout::size == 20u, "optimizer state layout size");
    ok &= require(nrc::OptimizerEntryLayout::size == 16u, "optimizer entry layout size");
    ok &= require(nrc::QueueStateLayout::size == 20u, "queue state layout size");

    const nrc::ResourceLayout layout = nrc::compute_resource_layout({1280, 720});
    ok &= require(layout.weightCount == 20672u, "resource layout weight count");
    ok &= require(layout.weightsBytes == 41344u, "weights are fp16");
    ok &= require(layout.useWeightsBytes == 41344u, "use weights are fp16");
    ok &= require(layout.optimizerEntriesBytes == 330752u, "optimizer entries are fp32 state");
    ok &= require(layout.gradientsBytes == 82688u, "gradients are fp32");
    ok &= require(layout.evalRecordCount == 987136u, "eval records cover pixels plus train records");
    ok &= require(layout.evalRecordsBytes == 19742720u, "eval record bytes");
    ok &= require(layout.evalCountBytes == 4u, "eval count bytes");
    ok &= require(layout.queueStateBytes == 20u, "queue state bytes");
    ok &= require(layout.trainBatchCountBytes == 16u, "all train batch count bytes");
    ok &= require(layout.trainBatchRecordBytes == 655360u, "single train batch bytes");
    ok &= require(layout.trainBatchRecordsTotalBytes == 2621440u, "all train batch bytes");

    const std::array<std::uint32_t, 1> sg32 = {32};
    const std::array<nrc::CooperativeMatrixCapability, 1> mat16 = {{
        {
            16,
            16,
            16,
            nrc::ComponentType::Float16,
            nrc::ComponentType::Float16,
            nrc::ComponentType::Float16,
            nrc::ComponentType::Float16,
            nrc::MatrixScope::Subgroup,
            true,
        },
    }};
    const nrc::DeviceCapabilitiesView caps16{
        true,
        true,
        true,
        true,
        true,
        true,
        sg32,
        mat16,
    };
    ok &= require(nrc::choose_profile(caps16) ==
                      nrc::ProfileId::KhrFp16M16N16K16Sg32Wg128,
                  "prefer SG32 16x16x16 KHR profile");

    const std::array<std::uint32_t, 1> sg16 = {16};
    const nrc::DeviceCapabilitiesView caps16Sg16{
        true,
        true,
        true,
        true,
        true,
        true,
        sg16,
        mat16,
    };
    ok &= require(nrc::choose_profile(caps16Sg16) ==
                      nrc::ProfileId::KhrFp16M16N16K16Sg16Wg128,
                  "choose SG16 profile when SG32 is unavailable");

    const nrc::DeviceCapabilitiesView noAtomic{
        true,
        true,
        false,
        true,
        true,
        true,
        sg32,
        mat16,
    };
    ok &= require(nrc::choose_profile(noAtomic) == nrc::ProfileId::None,
                  "full training profile requires float atomic add");
    ok &= require(nrc::choose_profile(noAtomic, {.requireTraining = false}) ==
                      nrc::ProfileId::KhrFp16M16N16K16Sg32Wg128,
                  "inference-only profile can skip float atomic add");

    const std::array<nrc::CooperativeMatrixCapability, 1> mat8 = {{
        {
            8,
            16,
            16,
            nrc::ComponentType::Float16,
            nrc::ComponentType::Float16,
            nrc::ComponentType::Float16,
            nrc::ComponentType::Float16,
            nrc::MatrixScope::Subgroup,
            true,
        },
    }};
    const nrc::DeviceCapabilitiesView capsM8Only{
        true,
        true,
        true,
        true,
        true,
        true,
        sg32,
        mat8,
    };
    ok &= require(nrc::choose_profile(capsM8Only) ==
                      nrc::ProfileId::KhrFp16M8N16K16Sg32Wg128,
                  "choose implemented M8 profile when M16 is unavailable");

    const nrc::DeviceCapabilitiesView emptyCaps{};
    ok &= require(nrc::choose_profile(emptyCaps) == nrc::ProfileId::None,
                  "no silent fallback by default");
    ok &= require(nrc::choose_profile(emptyCaps, {.allowReferenceBackend = true}) ==
                      nrc::ProfileId::ScalarReferenceWg64,
                  "reference fallback is explicit");

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
