#include "nrc/nrc_profile.hpp"

#include "nrc/nrc_khr_shader_layout.hpp"

#include <array>

namespace nrc {

namespace {

constexpr std::array<Profile, 4> kProfiles = {{
    {
        ProfileId::KhrFp16M16N16K16Sg32Wg128,
        BackendKind::KhrCooperativeMatrix,
        "KHR_FP16_M16N16K16_SG32_WG128",
        16,
        16,
        16,
        32,
        128,
        true,
        true,
    },
    {
        ProfileId::KhrFp16M16N16K16Sg16Wg128,
        BackendKind::KhrCooperativeMatrix,
        "KHR_FP16_M16N16K16_SG16_WG128",
        16,
        16,
        16,
        16,
        128,
        true,
        true,
    },
    {
        ProfileId::KhrFp16M8N16K16Sg32Wg128,
        BackendKind::KhrCooperativeMatrix,
        "KHR_FP16_M8N16K16_SG32_WG128",
        8,
        16,
        16,
        32,
        128,
        true,
        true,
    },
    {
        ProfileId::ScalarReferenceWg64,
        BackendKind::ScalarReference,
        "SCALAR_REFERENCE_WG64",
        0,
        0,
        0,
        0,
        64,
        false,
        true,
    },
}};

constexpr std::array<ProfileShaderConfig, 4> kProfileShaderConfigs = {{
    {
        ProfileId::KhrFp16M16N16K16Sg32Wg128,
        BackendKind::KhrCooperativeMatrix,
        "nrc/backend/khr_coopmat/NrcNNKhr.glsl",
        128,
        32,
        16,
        16,
        16,
        true,
        true,
        true,
    },
    {
        ProfileId::KhrFp16M16N16K16Sg16Wg128,
        BackendKind::KhrCooperativeMatrix,
        "nrc/backend/khr_coopmat/NrcNNKhr.glsl",
        128,
        16,
        16,
        16,
        16,
        true,
        true,
        true,
    },
    {
        ProfileId::KhrFp16M8N16K16Sg32Wg128,
        BackendKind::KhrCooperativeMatrix,
        "nrc/backend/khr_coopmat/NrcNNKhr.glsl",
        128,
        32,
        8,
        16,
        16,
        true,
        true,
        true,
    },
    {
        ProfileId::ScalarReferenceWg64,
        BackendKind::ScalarReference,
        "nrc/backend/scalar/NrcNNScalar.glsl",
        64,
        0,
        0,
        0,
        0,
        false,
        false,
        false,
    },
}};

bool has_subgroup_size(std::span<const std::uint32_t> sizes, std::uint32_t required) {
    for (std::uint32_t size : sizes) {
        if (size == required) {
            return true;
        }
    }
    return false;
}

bool has_matrix(const DeviceCapabilitiesView &capabilities, const Profile &profile) {
    for (const CooperativeMatrixCapability &matrix : capabilities.cooperativeMatrices) {
        if (matrix.m == profile.matrixM &&
            matrix.n == profile.matrixN &&
            matrix.k == profile.matrixK &&
            matrix.aType == ComponentType::Float16 &&
            matrix.bType == ComponentType::Float16 &&
            matrix.cType == ComponentType::Float16 &&
            matrix.resultType == ComponentType::Float16 &&
            matrix.scope == MatrixScope::Subgroup &&
            matrix.computeStage) {
            return true;
        }
    }
    return false;
}

bool has_required_feature_set(const DeviceCapabilitiesView &capabilities) {
    return capabilities.khrCooperativeMatrix &&
           capabilities.shaderFloat16 &&
           capabilities.storageBuffer16BitAccess &&
           capabilities.subgroupSizeControl &&
           capabilities.computeFullSubgroups;
}

bool is_reference_allowed(const ProfileRequirements &requirements, const Profile &profile) {
    return profile.backend == BackendKind::ScalarReference &&
           profile.implemented &&
           requirements.allowReferenceBackend;
}

ProfileSupportReport make_report(const Profile &profile, ProfileSupportStatus status) {
    return {
        profile.id,
        profile.name,
        status,
        status == ProfileSupportStatus::Supported,
    };
}

ProfileSupportReport make_empty_report() {
    return {
        ProfileId::None,
        "NONE",
        ProfileSupportStatus::UnsupportedBackend,
        false,
    };
}

} // namespace

std::span<const Profile> profiles() {
    return kProfiles;
}

const Profile *find_profile(ProfileId id) {
    for (const Profile &profile : kProfiles) {
        if (profile.id == id) {
            return &profile;
        }
    }
    return nullptr;
}

std::span<const ProfileShaderConfig> profile_shader_configs() {
    return kProfileShaderConfigs;
}

const ProfileShaderConfig *find_profile_shader_config(ProfileId id) {
    for (const ProfileShaderConfig &config : kProfileShaderConfigs) {
        if (config.id == id) {
            return &config;
        }
    }
    return nullptr;
}

std::uint32_t required_training_shared_memory_bytes(const Profile &profile) {
    const ProfileShaderConfig *config = find_profile_shader_config(profile.id);
    if (config == nullptr || config->backend != BackendKind::KhrCooperativeMatrix) {
        return 0;
    }

    const KhrShaderSharedMemoryLayout layout =
        compute_khr_shader_shared_memory_layout(*config);
    return layout.valid ? layout.sharedBytes : 0;
}

ProfileSupportReport evaluate_profile_support(const DeviceCapabilitiesView &capabilities,
                                               const ProfileRequirements &requirements,
                                               const Profile &profile) {
    if (profile.backend == BackendKind::ScalarReference) {
        if (!profile.implemented) {
            return make_report(profile, ProfileSupportStatus::ProfileNotImplemented);
        }
        if (!is_reference_allowed(requirements, profile)) {
            return make_report(profile, ProfileSupportStatus::ReferenceBackendNotAllowed);
        }
        return make_report(profile, ProfileSupportStatus::Supported);
    }

    if (profile.backend != BackendKind::KhrCooperativeMatrix) {
        return make_report(profile, ProfileSupportStatus::UnsupportedBackend);
    }

    if (!profile.implemented) {
        return make_report(profile, ProfileSupportStatus::ProfileNotImplemented);
    }

    if (!has_required_feature_set(capabilities)) {
        return make_report(profile, ProfileSupportStatus::MissingRequiredFeature);
    }

    if (requirements.requireTraining && !capabilities.shaderBufferFloat32AtomicAdd) {
        return make_report(profile, ProfileSupportStatus::MissingTrainingAtomicFloat);
    }

    if (requirements.requireTraining && !capabilities.subgroupArithmetic) {
        return make_report(profile, ProfileSupportStatus::MissingSubgroupArithmetic);
    }

    if (!has_subgroup_size(capabilities.supportedSubgroupSizes, profile.requiredSubgroupSize)) {
        return make_report(profile, ProfileSupportStatus::MissingSubgroupSize);
    }

    if (!has_matrix(capabilities, profile)) {
        return make_report(profile, ProfileSupportStatus::MissingCooperativeMatrixShape);
    }

    if (requirements.requireTraining) {
        const std::uint32_t requiredSharedMemoryBytes =
            required_training_shared_memory_bytes(profile);
        if (requiredSharedMemoryBytes > 0 &&
            capabilities.maxComputeSharedMemorySize < requiredSharedMemoryBytes) {
            return make_report(profile, ProfileSupportStatus::MissingSharedMemory);
        }
    }

    return make_report(profile, ProfileSupportStatus::Supported);
}

bool supports_profile(const DeviceCapabilitiesView &capabilities,
                      const ProfileRequirements &requirements,
                      const Profile &profile) {
    return evaluate_profile_support(capabilities, requirements, profile).supported;
}

ProfileId choose_profile(const DeviceCapabilitiesView &capabilities,
                         const ProfileRequirements &requirements) {
    for (const Profile &profile : kProfiles) {
        if (supports_profile(capabilities, requirements, profile)) {
            return profile.id;
        }
    }
    return ProfileId::None;
}

TrainingProfileSelection choose_training_profile(
    const DeviceCapabilitiesView &capabilities,
    const TrainingProfileSelectionOptions &options) {
    TrainingProfileSelection selection{};
    selection.khrProfileSupport = make_empty_report();
    selection.scalarProfileSupport = make_empty_report();

    const ProfileRequirements khrTrainingRequirements{
        true,
        false,
    };
    const Profile *selectedKhrProfile = nullptr;
    for (const Profile &profile : kProfiles) {
        if (profile.backend != BackendKind::KhrCooperativeMatrix) {
            continue;
        }

        const ProfileSupportReport report =
            evaluate_profile_support(capabilities, khrTrainingRequirements, profile);
        if (selection.khrProfileSupport.profileId == ProfileId::None) {
            selection.khrProfileSupport = report;
        }
        if (report.supported) {
            selection.khrProfileSupport = report;
            selectedKhrProfile = &profile;
            break;
        }
    }

    if (selectedKhrProfile != nullptr) {
        if (options.enableExperimentalAllCoopmat) {
            selection.status = TrainingProfileSelectionStatus::SelectedKhrAllCoopmat;
            selection.implementation = TrainingImplementationKind::KhrAllCoopmat;
            selection.profileId = selectedKhrProfile->id;
            selection.profileName = selectedKhrProfile->name;
            selection.shaderConfig = find_profile_shader_config(selectedKhrProfile->id);
            selection.requiredSharedMemoryBytes =
                required_training_shared_memory_bytes(*selectedKhrProfile);
            selection.selected = true;
            selection.useKhrCoopmatAllDWeights = true;
            selection.useKhrCoopmatAllDActivation = true;
            return selection;
        }

        if (options.allowKhrGroundwork) {
            selection.status = TrainingProfileSelectionStatus::SelectedKhrGroundwork;
            selection.implementation = TrainingImplementationKind::KhrGroundwork;
            selection.profileId = selectedKhrProfile->id;
            selection.profileName = selectedKhrProfile->name;
            selection.shaderConfig = find_profile_shader_config(selectedKhrProfile->id);
            selection.requiredSharedMemoryBytes =
                required_training_shared_memory_bytes(*selectedKhrProfile);
            selection.selected = true;
            return selection;
        }
    }

    const Profile *scalarProfile = find_profile(ProfileId::ScalarReferenceWg64);
    if (scalarProfile != nullptr) {
        const ProfileRequirements scalarRequirements{
            true,
            options.allowScalarReferenceFallback,
        };
        selection.scalarProfileSupport =
            evaluate_profile_support(capabilities, scalarRequirements, *scalarProfile);
        if (selection.scalarProfileSupport.supported) {
            selection.status = TrainingProfileSelectionStatus::SelectedScalarReference;
            selection.implementation = TrainingImplementationKind::ScalarReference;
            selection.profileId = scalarProfile->id;
            selection.profileName = scalarProfile->name;
            selection.shaderConfig = find_profile_shader_config(scalarProfile->id);
            selection.selected = true;
            selection.fallback = true;
            return selection;
        }
    }

    return selection;
}

TrainingShaderCompileConfig make_training_shader_compile_config(
    const TrainingProfileSelection &selection) {
    TrainingShaderCompileConfig config{};
    if (!selection.selected || selection.shaderConfig == nullptr) {
        return config;
    }

    config.valid = true;
    config.backend = selection.shaderConfig->backend;
    config.shaderHeader = selection.shaderConfig->shaderHeader;
    config.workgroupSize = selection.shaderConfig->workgroupSize;
    config.subgroupSize = selection.shaderConfig->subgroupSize;
    config.cooperativeMatrixM = selection.shaderConfig->cooperativeMatrixM;
    config.cooperativeMatrixN = selection.shaderConfig->cooperativeMatrixN;
    config.cooperativeMatrixK = selection.shaderConfig->cooperativeMatrixK;
    config.requireFullSubgroups = selection.shaderConfig->requireFullSubgroups;
    config.requireSubgroupSizeControl = selection.shaderConfig->requireSubgroupSizeControl;
    config.requireSubgroupArithmeticForTraining =
        selection.shaderConfig->requireSubgroupArithmeticForTraining;
    config.useKhrCoopmatAllDWeights = selection.useKhrCoopmatAllDWeights;
    config.useKhrCoopmatAllDActivation = selection.useKhrCoopmatAllDActivation;
    return config;
}

} // namespace nrc
