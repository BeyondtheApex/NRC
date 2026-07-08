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
    constexpr std::uint32_t kRequiredKhrTrainingSharedBytes = 98304u;

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
        kRequiredKhrTrainingSharedBytes,
    };
    ok &= require(nrc::choose_profile(caps16) ==
                      nrc::ProfileId::KhrFp16M16N16K16Sg32Wg128,
                  "prefer SG32 16x16x16 KHR profile");
    const nrc::TrainingProfileSelection defaultTraining =
        nrc::choose_training_profile(caps16);
    ok &= require(defaultTraining.selected,
                  "default training profile selection succeeds for valid KHR capabilities");
    ok &= require(defaultTraining.status ==
                      nrc::TrainingProfileSelectionStatus::SelectedKhrGroundwork,
                  "default training profile uses KHR groundwork mode");
    ok &= require(defaultTraining.implementation ==
                      nrc::TrainingImplementationKind::KhrGroundwork,
                  "default training implementation records KHR groundwork");
    ok &= require(defaultTraining.profileId ==
                      nrc::ProfileId::KhrFp16M16N16K16Sg32Wg128,
                  "default training profile records selected KHR profile id");
    ok &= require(defaultTraining.shaderConfig != nullptr &&
                      defaultTraining.shaderConfig->id == defaultTraining.profileId,
                  "default training profile carries shader config");
    ok &= require(defaultTraining.requiredSharedMemoryBytes ==
                      kRequiredKhrTrainingSharedBytes,
                  "default training profile reports required shared memory");
    ok &= require(!defaultTraining.fallback,
                  "default KHR training selection is not a fallback");
    ok &= require(!defaultTraining.useKhrCoopmatAllDWeights &&
                      !defaultTraining.useKhrCoopmatAllDActivation,
                  "default KHR training selection does not enable experimental all-coopmat flags");
    ok &= require(defaultTraining.khrProfileSupport.supported,
                  "default training selection carries supported KHR report");
    const nrc::TrainingShaderCompileConfig defaultTrainingCompile =
        nrc::make_training_shader_compile_config(defaultTraining);
    ok &= require(defaultTrainingCompile.valid,
                  "default training compile config is valid");
    ok &= require(defaultTrainingCompile.backend ==
                      nrc::BackendKind::KhrCooperativeMatrix,
                  "default training compile config records KHR backend");
    ok &= require(std::strcmp(defaultTrainingCompile.shaderHeader,
                              "nrc/backend/khr_coopmat/NrcNNKhr.glsl") == 0,
                  "default training compile config records KHR shader header");
    ok &= require(defaultTrainingCompile.workgroupSize == 128u &&
                      defaultTrainingCompile.subgroupSize == 32u,
                  "default training compile config records workgroup and subgroup sizes");
    ok &= require(defaultTrainingCompile.cooperativeMatrixM == 16u &&
                      defaultTrainingCompile.cooperativeMatrixN == 16u &&
                      defaultTrainingCompile.cooperativeMatrixK == 16u,
                  "default training compile config records cooperative matrix shape");
    ok &= require(defaultTrainingCompile.requireFullSubgroups &&
                      defaultTrainingCompile.requireSubgroupSizeControl &&
                      defaultTrainingCompile.requireSubgroupArithmeticForTraining,
                  "default training compile config records KHR pipeline requirements");
    ok &= require(!defaultTrainingCompile.useKhrCoopmatAllDWeights &&
                      !defaultTrainingCompile.useKhrCoopmatAllDActivation,
                  "default training compile config keeps all-coopmat defines disabled");

    const nrc::TrainingProfileSelection allCoopmatTraining =
        nrc::choose_training_profile(
            caps16,
            {.enableExperimentalAllCoopmat = true});
    ok &= require(allCoopmatTraining.selected,
                  "experimental all-coopmat training selection succeeds");
    ok &= require(allCoopmatTraining.status ==
                      nrc::TrainingProfileSelectionStatus::SelectedKhrAllCoopmat,
                  "experimental all-coopmat training records selected status");
    ok &= require(allCoopmatTraining.implementation ==
                      nrc::TrainingImplementationKind::KhrAllCoopmat,
                  "experimental all-coopmat training records implementation");
    ok &= require(allCoopmatTraining.profileId ==
                      nrc::ProfileId::KhrFp16M16N16K16Sg32Wg128,
                  "experimental all-coopmat training keeps selected KHR profile");
    ok &= require(allCoopmatTraining.useKhrCoopmatAllDWeights &&
                      allCoopmatTraining.useKhrCoopmatAllDActivation,
                  "experimental all-coopmat training enables both all-dW and all-dA flags");
    const nrc::TrainingShaderCompileConfig allCoopmatCompile =
        nrc::make_training_shader_compile_config(allCoopmatTraining);
    ok &= require(allCoopmatCompile.valid,
                  "experimental all-coopmat compile config is valid");
    ok &= require(allCoopmatCompile.useKhrCoopmatAllDWeights &&
                      allCoopmatCompile.useKhrCoopmatAllDActivation,
                  "experimental all-coopmat compile config enables both all-coopmat defines");

    const nrc::TrainingProfileSelection noAllowedKhrTraining =
        nrc::choose_training_profile(
            caps16,
            {false, false, false});
    ok &= require(!noAllowedKhrTraining.selected,
                  "training selection rejects valid KHR profile when all KHR modes are disabled");
    ok &= require(noAllowedKhrTraining.status ==
                      nrc::TrainingProfileSelectionStatus::NoSupportedTrainingProfile,
                  "disabled KHR training modes record no supported training profile");
    ok &= require(noAllowedKhrTraining.khrProfileSupport.supported,
                  "disabled KHR training modes still carry supported KHR capability report");
    ok &= require(!nrc::make_training_shader_compile_config(noAllowedKhrTraining).valid,
                  "unselected training profile has invalid compile config");

    const nrc::TrainingProfileSelection scalarBecauseKhrDisabled =
        nrc::choose_training_profile(
            caps16,
            {false, false, true});
    ok &= require(scalarBecauseKhrDisabled.selected,
                  "training selection can fall back to scalar when KHR modes are disabled");
    ok &= require(scalarBecauseKhrDisabled.status ==
                      nrc::TrainingProfileSelectionStatus::SelectedScalarReference,
                  "scalar fallback records selected scalar status");
    ok &= require(scalarBecauseKhrDisabled.implementation ==
                      nrc::TrainingImplementationKind::ScalarReference,
                  "scalar fallback records scalar implementation");
    ok &= require(scalarBecauseKhrDisabled.profileId ==
                      nrc::ProfileId::ScalarReferenceWg64,
                  "scalar fallback records scalar profile id");
    ok &= require(scalarBecauseKhrDisabled.fallback,
                  "scalar training selection records fallback");
    const nrc::TrainingShaderCompileConfig scalarFallbackCompile =
        nrc::make_training_shader_compile_config(scalarBecauseKhrDisabled);
    ok &= require(scalarFallbackCompile.valid,
                  "scalar fallback compile config is valid");
    ok &= require(scalarFallbackCompile.backend == nrc::BackendKind::ScalarReference,
                  "scalar fallback compile config records scalar backend");
    ok &= require(std::strcmp(scalarFallbackCompile.shaderHeader,
                              "nrc/backend/scalar/NrcNNScalar.glsl") == 0,
                  "scalar fallback compile config records scalar shader header");
    ok &= require(scalarFallbackCompile.workgroupSize == 64u &&
                      scalarFallbackCompile.subgroupSize == 0u,
                  "scalar fallback compile config records scalar workgroup shape");

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

        nrc::DeviceCapabilitiesView insufficientSharedMemory = caps16;
        insufficientSharedMemory.maxComputeSharedMemorySize =
            kRequiredKhrTrainingSharedBytes - 1u;
        ok &= require(nrc::choose_profile(insufficientSharedMemory) == nrc::ProfileId::None,
                      "full training profile requires enough shared memory");
        ok &= require(nrc::choose_profile(insufficientSharedMemory,
                                          {.requireTraining = false}) ==
                          nrc::ProfileId::KhrFp16M16N16K16Sg32Wg128,
                      "inference-only profile skips training shared memory requirement");
        const nrc::ProfileSupportReport sharedMemoryReport =
            nrc::evaluate_profile_support(insufficientSharedMemory, {}, *sg32Profile);
        ok &= require(!sharedMemoryReport.supported,
                      "support report rejects insufficient shared memory");
        ok &= require(sharedMemoryReport.status ==
                          nrc::ProfileSupportStatus::MissingSharedMemory,
                      "support report records missing shared memory");
        const nrc::TrainingProfileSelection missingSharedTraining =
            nrc::choose_training_profile(insufficientSharedMemory);
        ok &= require(!missingSharedTraining.selected,
                      "training selection rejects insufficient KHR shared memory by default");
        ok &= require(missingSharedTraining.khrProfileSupport.status ==
                          nrc::ProfileSupportStatus::MissingSharedMemory,
                      "training selection carries missing shared memory KHR reason");
        const nrc::TrainingProfileSelection sharedMemoryScalarFallback =
            nrc::choose_training_profile(
                insufficientSharedMemory,
                {false, true, true});
        ok &= require(sharedMemoryScalarFallback.selected,
                      "training selection can fall back to scalar on insufficient shared memory");
        ok &= require(sharedMemoryScalarFallback.status ==
                          nrc::TrainingProfileSelectionStatus::SelectedScalarReference,
                      "insufficient shared memory fallback selects scalar status");
        ok &= require(sharedMemoryScalarFallback.fallback,
                      "insufficient shared memory scalar selection records fallback");
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
        kRequiredKhrTrainingSharedBytes,
    };
    ok &= require(nrc::choose_profile(caps16Sg16) ==
                      nrc::ProfileId::KhrFp16M16N16K16Sg16Wg128,
                  "choose SG16 profile when SG32 is unavailable");
    ok &= require(nrc::choose_training_profile(caps16Sg16).profileId ==
                      nrc::ProfileId::KhrFp16M16N16K16Sg16Wg128,
                  "training selection chooses SG16 profile when SG32 is unavailable");
    ok &= require(nrc::make_training_shader_compile_config(
                      nrc::choose_training_profile(caps16Sg16)).subgroupSize == 16u,
                  "training compile config records SG16 profile subgroup size");

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
        kRequiredKhrTrainingSharedBytes,
    };
    ok &= require(nrc::choose_profile(capsM8Only) ==
                      nrc::ProfileId::KhrFp16M8N16K16Sg32Wg128,
                  "choose implemented M8 profile when M16 is unavailable");
    ok &= require(nrc::choose_training_profile(capsM8Only).profileId ==
                      nrc::ProfileId::KhrFp16M8N16K16Sg32Wg128,
                  "training selection chooses implemented M8 profile when M16 is unavailable");
    ok &= require(nrc::make_training_shader_compile_config(
                      nrc::choose_training_profile(capsM8Only)).cooperativeMatrixM == 8u,
                  "training compile config records implemented M8 matrix shape");

    const nrc::DeviceCapabilitiesView emptyCaps{};
    ok &= require(nrc::choose_profile(emptyCaps) == nrc::ProfileId::None,
                  "no silent fallback by default");
    ok &= require(nrc::choose_profile(emptyCaps, {.allowReferenceBackend = true}) ==
                      nrc::ProfileId::ScalarReferenceWg64,
                  "reference fallback is explicit");
    const nrc::TrainingProfileSelection emptyTraining =
        nrc::choose_training_profile(emptyCaps);
    ok &= require(!emptyTraining.selected,
                  "training selection rejects empty capabilities by default");
    ok &= require(emptyTraining.khrProfileSupport.status ==
                      nrc::ProfileSupportStatus::MissingRequiredFeature,
                  "training selection records missing KHR features for empty capabilities");
    const nrc::TrainingProfileSelection emptyScalarFallback =
        nrc::choose_training_profile(emptyCaps, {false, true, true});
    ok &= require(emptyScalarFallback.selected,
                  "training selection can fall back to scalar for empty capabilities");
    ok &= require(emptyScalarFallback.status ==
                      nrc::TrainingProfileSelectionStatus::SelectedScalarReference,
                  "empty capability fallback records scalar status");
    ok &= require(emptyScalarFallback.khrProfileSupport.status ==
                      nrc::ProfileSupportStatus::MissingRequiredFeature,
                  "empty capability fallback preserves KHR rejection reason");
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
            const nrc::KhrHiddenDWeightTileStreamingLayout &tileStreaming =
                sharedLayout.hiddenDWeightTileStreaming;
            const std::uint32_t expectedHiddenRowBlocks =
                nrc::kNnHiddenWidth / config->cooperativeMatrixM;
            const std::uint32_t expectedHiddenColumnBlocks =
                nrc::kNnHiddenWidth / config->cooperativeMatrixN;
            const std::uint32_t expectedDPreRowScratch =
                (config->cooperativeMatrixM * config->workgroupSize) /
                nrc::kFp16PerUvec4;
            const std::uint32_t expectedTilePartial =
                (config->cooperativeMatrixM * config->cooperativeMatrixN) /
                nrc::kFp16PerUvec4;
            ok &= require(tileStreaming.valid,
                          "KHR hidden dW tile-streaming plan is valid");
            ok &= require(tileStreaming.rowBlockCount == expectedHiddenRowBlocks,
                          "KHR hidden dW tile-streaming row block count");
            ok &= require(tileStreaming.columnBlockCount == expectedHiddenColumnBlocks,
                          "KHR hidden dW tile-streaming column block count");
            ok &= require(tileStreaming.tileCount ==
                              expectedHiddenRowBlocks * expectedHiddenColumnBlocks,
                          "KHR hidden dW tile-streaming tile count");
            ok &= require(tileStreaming.dPreRowScratch.baseUvec4 == 0u,
                          "KHR hidden dW tile-streaming dPre base");
            ok &= require(tileStreaming.dPreRowScratch.countUvec4 ==
                              expectedDPreRowScratch,
                          "KHR hidden dW tile-streaming dPre row scratch size");
            ok &= require(tileStreaming.partialTileUvec4Count == expectedTilePartial,
                          "KHR hidden dW tile-streaming partial tile size");
            ok &= require(tileStreaming.partialScratch.baseUvec4 ==
                              tileStreaming.dPreRowScratch.end_uvec4(),
                          "KHR hidden dW tile-streaming partial base");
            ok &= require(tileStreaming.partialScratch.countUvec4 ==
                              expectedTilePartial * sharedLayout.subgroupCount,
                          "KHR hidden dW tile-streaming partial scratch size");
            ok &= require(tileStreaming.totalScratchUvec4Count ==
                              tileStreaming.dPreRowScratch.countUvec4 +
                                  tileStreaming.partialScratch.countUvec4,
                          "KHR hidden dW tile-streaming total scratch size");
            ok &= require(tileStreaming.combinedScratch.end_uvec4() <=
                              sharedLayout.activeTile.end_uvec4(),
                          "KHR hidden dW tile-streaming stays inside active tile");
            ok &= require(tileStreaming.fitsActiveTile,
                          "KHR hidden dW tile-streaming fits active tile");
            ok &= require(tileStreaming.preservesSharedActivationSnapshots,
                          "KHR hidden dW tile-streaming preserves snapshots");
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
            const nrc::KhrShaderSharedMemoryLayout scalarSharedLayout =
                nrc::compute_khr_shader_shared_memory_layout(*config);
            ok &= require(!scalarSharedLayout.valid,
                          "scalar shader config has no KHR shared layout");
            ok &= require(!scalarSharedLayout.hiddenDWeightTileStreaming.valid,
                          "scalar shader config has no hidden dW tile-streaming plan");
        }
    }

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
