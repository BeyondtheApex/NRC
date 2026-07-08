#include "nrc/nrc_profile.hpp"

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

} // namespace nrc
