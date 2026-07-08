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

bool supports_profile(const DeviceCapabilitiesView &capabilities,
                      const ProfileRequirements &requirements,
                      const Profile &profile) {
    if (profile.backend == BackendKind::ScalarReference) {
        return requirements.allowReferenceBackend;
    }

    if (profile.backend != BackendKind::KhrCooperativeMatrix) {
        return false;
    }

    if (!capabilities.khrCooperativeMatrix ||
        !capabilities.shaderFloat16 ||
        !capabilities.storageBuffer16BitAccess ||
        !capabilities.subgroupSizeControl ||
        !capabilities.computeFullSubgroups) {
        return false;
    }

    if (requirements.requireTraining && !capabilities.shaderBufferFloat32AtomicAdd) {
        return false;
    }

    if (!has_subgroup_size(capabilities.supportedSubgroupSizes, profile.requiredSubgroupSize)) {
        return false;
    }

    return has_matrix(capabilities, profile);
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
