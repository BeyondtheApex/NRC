#include "nrc/nrc_constants.hpp"
#include "nrc/nrc_khr_shader_layout.hpp"
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
        true,
        sg32,
        mat16,
    };
    ok &= require(nrc::choose_profile(caps16) ==
                      nrc::ProfileId::KhrFp16M16N16K16Sg32Wg128,
                  "prefer SG32 16x16x16 KHR profile");

    const nrc::Profile *sg32Profile =
        nrc::find_profile(nrc::ProfileId::KhrFp16M16N16K16Sg32Wg128);
    ok &= require(sg32Profile != nullptr, "find SG32 KHR profile");
    if (sg32Profile != nullptr) {
        const nrc::ProfileSupportReport report =
            nrc::evaluate_profile_support(caps16, {}, *sg32Profile);
        ok &= require(report.supported, "support report marks valid KHR profile supported");
        ok &= require(report.status == nrc::ProfileSupportStatus::Supported,
                      "support report records supported status");
        ok &= require(report.profileId == sg32Profile->id,
                      "support report records profile id");
        ok &= require(std::strcmp(report.profileName, sg32Profile->name) == 0,
                      "support report records profile name");
    }

    const std::array<std::uint32_t, 1> sg16 = {16};
    const nrc::DeviceCapabilitiesView caps16Sg16{
        true,
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
        true,
        sg32,
        mat16,
    };
    ok &= require(nrc::choose_profile(noAtomic) == nrc::ProfileId::None,
                  "full training profile requires float atomic add");
    ok &= require(nrc::choose_profile(noAtomic, {.requireTraining = false}) ==
                      nrc::ProfileId::KhrFp16M16N16K16Sg32Wg128,
                  "inference-only profile can skip float atomic add");
    if (sg32Profile != nullptr) {
        const nrc::ProfileSupportReport trainingAtomicReport =
            nrc::evaluate_profile_support(noAtomic, {}, *sg32Profile);
        ok &= require(!trainingAtomicReport.supported,
                      "support report rejects missing training atomic add");
        ok &= require(trainingAtomicReport.status ==
                          nrc::ProfileSupportStatus::MissingTrainingAtomicFloat,
                      "support report records missing training atomic add");

        const nrc::ProfileSupportReport inferenceOnlyReport =
            nrc::evaluate_profile_support(noAtomic, {.requireTraining = false}, *sg32Profile);
        ok &= require(inferenceOnlyReport.supported,
                      "support report accepts inference-only profile without atomic add");
        ok &= require(inferenceOnlyReport.status == nrc::ProfileSupportStatus::Supported,
                      "support report records inference-only support");

        const nrc::DeviceCapabilitiesView noSubgroupArithmetic{
            true,
            true,
            true,
            true,
            true,
            false,
            true,
            sg32,
            mat16,
        };
        ok &= require(nrc::choose_profile(noSubgroupArithmetic) == nrc::ProfileId::None,
                      "full training profile requires subgroup arithmetic");
        ok &= require(nrc::choose_profile(noSubgroupArithmetic, {.requireTraining = false}) ==
                          nrc::ProfileId::KhrFp16M16N16K16Sg32Wg128,
                      "inference-only profile can skip subgroup arithmetic");
        const nrc::ProfileSupportReport subgroupArithmeticReport =
            nrc::evaluate_profile_support(noSubgroupArithmetic, {}, *sg32Profile);
        ok &= require(!subgroupArithmeticReport.supported,
                      "support report rejects missing subgroup arithmetic");
        ok &= require(subgroupArithmeticReport.status ==
                          nrc::ProfileSupportStatus::MissingSubgroupArithmetic,
                      "support report records missing subgroup arithmetic");

        const nrc::DeviceCapabilitiesView missingFeature{
            false,
            true,
            true,
            true,
            true,
            true,
            true,
            sg32,
            mat16,
        };
        const nrc::ProfileSupportReport missingFeatureReport =
            nrc::evaluate_profile_support(missingFeature, {}, *sg32Profile);
        ok &= require(missingFeatureReport.status ==
                          nrc::ProfileSupportStatus::MissingRequiredFeature,
                      "support report records missing required feature");

        const std::array<std::uint32_t, 1> sg64 = {64};
        const nrc::DeviceCapabilitiesView missingSubgroup{
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            sg64,
            mat16,
        };
        const nrc::ProfileSupportReport missingSubgroupReport =
            nrc::evaluate_profile_support(missingSubgroup, {}, *sg32Profile);
        ok &= require(missingSubgroupReport.status ==
                          nrc::ProfileSupportStatus::MissingSubgroupSize,
                      "support report records missing subgroup size");

        const std::array<nrc::CooperativeMatrixCapability, 0> noMatrices{};
        const nrc::DeviceCapabilitiesView missingMatrix{
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            sg32,
            noMatrices,
        };
        const nrc::ProfileSupportReport missingMatrixReport =
            nrc::evaluate_profile_support(missingMatrix, {}, *sg32Profile);
        ok &= require(missingMatrixReport.status ==
                          nrc::ProfileSupportStatus::MissingCooperativeMatrixShape,
                      "support report records missing cooperative matrix shape");
    }

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
    const nrc::Profile *scalarProfile =
        nrc::find_profile(nrc::ProfileId::ScalarReferenceWg64);
    ok &= require(scalarProfile != nullptr, "find scalar reference profile");
    if (scalarProfile != nullptr) {
        const nrc::ProfileSupportReport disallowedReferenceReport =
            nrc::evaluate_profile_support(emptyCaps, {}, *scalarProfile);
        ok &= require(!disallowedReferenceReport.supported,
                      "support report rejects scalar reference by default");
        ok &= require(disallowedReferenceReport.status ==
                          nrc::ProfileSupportStatus::ReferenceBackendNotAllowed,
                      "support report records reference backend not allowed");

        const nrc::ProfileSupportReport allowedReferenceReport =
            nrc::evaluate_profile_support(emptyCaps, {.allowReferenceBackend = true}, *scalarProfile);
        ok &= require(allowedReferenceReport.supported,
                      "support report accepts explicitly allowed scalar reference");
        ok &= require(allowedReferenceReport.status == nrc::ProfileSupportStatus::Supported,
                      "support report records scalar reference support");
    }

    const nrc::Profile noneProfile{};
    const nrc::ProfileSupportReport noneReport =
        nrc::evaluate_profile_support(caps16, {}, noneProfile);
    ok &= require(!noneReport.supported, "support report rejects none backend");
    ok &= require(noneReport.status == nrc::ProfileSupportStatus::UnsupportedBackend,
                  "support report records unsupported backend");

    const nrc::Profile unimplementedProfile{
        nrc::ProfileId::KhrFp16M16N16K16Sg32Wg128,
        nrc::BackendKind::KhrCooperativeMatrix,
        "UNIMPLEMENTED_KHR_PROFILE",
        16,
        16,
        16,
        32,
        128,
        true,
        false,
    };
    const nrc::ProfileSupportReport unimplementedReport =
        nrc::evaluate_profile_support(caps16, {}, unimplementedProfile);
    ok &= require(!unimplementedReport.supported,
                  "support report rejects unimplemented profile");
    ok &= require(unimplementedReport.status ==
                      nrc::ProfileSupportStatus::ProfileNotImplemented,
                  "support report records profile not implemented");

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
            ok &= require(config->requireSubgroupArithmeticForTraining,
                          "KHR shader config requires subgroup arithmetic for training");
            const nrc::KhrShaderSharedMemoryLayout sharedLayout =
                nrc::compute_khr_shader_shared_memory_layout(*config);
            ok &= require(sharedLayout.valid, "KHR shader shared layout is valid");
            ok &= require(sharedLayout.subgroupCount ==
                              config->workgroupSize / config->subgroupSize,
                          "KHR shader shared layout subgroup count");
            ok &= require(sharedLayout.activeTileUvec4Count == 1024u,
                          "KHR shader shared layout active tile size");
            ok &= require(sharedLayout.finalHiddenSnapshot.baseUvec4 == 1024u,
                          "KHR shader shared layout final hidden snapshot base");
            ok &= require(sharedLayout.outputSourceSnapshot.baseUvec4 == 2048u,
                          "KHR shader shared layout output source snapshot base");
            ok &= require(sharedLayout.activationCopySharedUvec4Count == 6144u,
                          "KHR shader shared layout activation copy size");
            ok &= require(sharedLayout.sharedUvec4Count == 6144u,
                          "KHR shader shared layout total uvec4 count");
            ok &= require(sharedLayout.sharedBytes == 98304u,
                          "KHR shader shared layout byte count");
            ok &= require(sharedLayout.dPreScratch.countUvec4 == 2048u,
                          "KHR shader shared layout dPre scratch size");
            ok &= require(sharedLayout.dPreScratchFitsActivePlusFinalSnapshot,
                          "KHR shader shared layout dPre lifetime fit");
            ok &= require(sharedLayout.outputDerivativeFp16FitsActiveTile,
                          "KHR shader shared layout output derivative fit");
            ok &= require(sharedLayout.outputDWeightPartialFitsActiveTile,
                          "KHR shader shared layout output dW partial fit");

            const std::uint32_t expectedHiddenPartialUvec4 =
                nrc::hidden_layer_weight_count() / nrc::kFp16PerUvec4 *
                sharedLayout.subgroupCount;
            ok &= require(sharedLayout.hiddenDWeightPartialScratch.countUvec4 ==
                              expectedHiddenPartialUvec4,
                          "KHR shader shared layout hidden dW partial size");
            ok &= require(sharedLayout.hiddenDWeightPartialFitsSharedAllocation,
                          "KHR shader shared layout hidden dW partial fits allocation");
            ok &= require(sharedLayout.hiddenDWeightPartialPreservesSharedHiddenActivationSnapshots ==
                              (expectedHiddenPartialUvec4 <=
                               sharedLayout.outputSourceSnapshot.baseUvec4),
                          "KHR shader shared layout hidden dW snapshot preservation");
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
            ok &= require(!config->requireSubgroupArithmeticForTraining,
                          "scalar shader config does not require subgroup arithmetic");
            ok &= require(!nrc::compute_khr_shader_shared_memory_layout(*config).valid,
                          "scalar shader config has no KHR shared layout");
        }
    }

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
