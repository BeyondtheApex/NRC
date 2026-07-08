#pragma once

#include <cstdint>
#include <span>

namespace nrc {

enum class ComponentType : std::uint8_t {
    Float16,
    Float32,
};

enum class MatrixScope : std::uint8_t {
    Subgroup,
};

enum class BackendKind : std::uint8_t {
    None,
    KhrCooperativeMatrix,
    ScalarReference,
};

enum class ProfileId : std::uint8_t {
    None,
    KhrFp16M16N16K16Sg32Wg128,
    KhrFp16M16N16K16Sg16Wg128,
    KhrFp16M8N16K16Sg32Wg128,
    ScalarReferenceWg64,
};

struct CooperativeMatrixCapability {
    std::uint32_t m = 0;
    std::uint32_t n = 0;
    std::uint32_t k = 0;
    ComponentType aType = ComponentType::Float16;
    ComponentType bType = ComponentType::Float16;
    ComponentType cType = ComponentType::Float16;
    ComponentType resultType = ComponentType::Float16;
    MatrixScope scope = MatrixScope::Subgroup;
    bool computeStage = false;
};

struct DeviceCapabilitiesView {
    bool shaderFloat16 = false;
    bool storageBuffer16BitAccess = false;
    bool shaderBufferFloat32AtomicAdd = false;
    bool subgroupSizeControl = false;
    bool computeFullSubgroups = false;
    bool khrCooperativeMatrix = false;
    std::span<const std::uint32_t> supportedSubgroupSizes;
    std::span<const CooperativeMatrixCapability> cooperativeMatrices;
};

struct ProfileRequirements {
    bool requireTraining = true;
    bool allowReferenceBackend = false;
};

struct Profile {
    ProfileId id = ProfileId::None;
    BackendKind backend = BackendKind::None;
    const char *name = "NONE";
    std::uint32_t matrixM = 0;
    std::uint32_t matrixN = 0;
    std::uint32_t matrixK = 0;
    std::uint32_t requiredSubgroupSize = 0;
    std::uint32_t workgroupSize = 0;
    bool production = false;
};

std::span<const Profile> profiles();
const Profile *find_profile(ProfileId id);
bool supports_profile(const DeviceCapabilitiesView &capabilities,
                      const ProfileRequirements &requirements,
                      const Profile &profile);
ProfileId choose_profile(const DeviceCapabilitiesView &capabilities,
                         const ProfileRequirements &requirements = {});

} // namespace nrc
