#include "nrc/nrc_constants.hpp"
#include "nrc/nrc_profile.hpp"
#include "nrc/nrc_records.hpp"
#include "nrc/nrc_resource_layout.hpp"

#include <array>
#include <cstring>
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

    ok &= require(nrc::find_profile_shader_config(nrc::ProfileId::None) == nullptr,
                  "no shader config for none profile");

    for (const nrc::Profile &profile : nrc::profiles()) {
        if (!profile.implemented) {
            continue;
        }

        const nrc::ProfileShaderConfig *config = nrc::find_profile_shader_config(profile.id);
        ok &= require(config != nullptr, "implemented profile has shader config");
        if (config == nullptr) {
            continue;
        }

        ok &= require(config->id == profile.id, "shader config profile id matches");
        ok &= require(config->backend == profile.backend, "shader config backend matches");
        ok &= require(config->workgroupSize == profile.workgroupSize, "shader config workgroup matches profile");
        ok &= require(config->shaderHeader != nullptr && config->shaderHeader[0] != '\0',
                      "shader config has a header");

        if (profile.backend == nrc::BackendKind::KhrCooperativeMatrix) {
            ok &= require(config->subgroupSize == profile.requiredSubgroupSize,
                          "KHR shader config subgroup matches profile");
            ok &= require(config->cooperativeMatrixM == profile.matrixM,
                          "KHR shader config M matches profile");
            ok &= require(config->cooperativeMatrixN == profile.matrixN,
                          "KHR shader config N matches profile");
            ok &= require(config->cooperativeMatrixK == profile.matrixK,
                          "KHR shader config K matches profile");
            ok &= require(config->requireFullSubgroups, "KHR shader config requires full subgroups");
            ok &= require(config->requireSubgroupSizeControl,
                          "KHR shader config requires subgroup size control");
            ok &= require(std::strcmp(config->shaderHeader,
                                      "nrc/backend/khr_coopmat/NrcNNKhr.glsl") == 0,
                          "KHR shader config header path");
        } else if (profile.backend == nrc::BackendKind::ScalarReference) {
            ok &= require(config->subgroupSize == 0u, "scalar shader config has no subgroup");
            ok &= require(config->cooperativeMatrixM == 0u &&
                              config->cooperativeMatrixN == 0u &&
                              config->cooperativeMatrixK == 0u,
                          "scalar shader config has no cooperative matrix dimensions");
            ok &= require(!config->requireFullSubgroups,
                          "scalar shader config does not require full subgroups");
            ok &= require(!config->requireSubgroupSizeControl,
                          "scalar shader config does not require subgroup size control");
        }
    }

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
