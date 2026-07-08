#pragma once

#include <cstdint>
#include <span>

namespace nrc {

enum class ComponentType : std::uint8_t {
    Unknown,
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
    bool subgroupArithmetic = false;
    bool khrCooperativeMatrix = false;
    std::span<const std::uint32_t> supportedSubgroupSizes;
    std::span<const CooperativeMatrixCapability> cooperativeMatrices;
    std::uint32_t maxComputeSharedMemorySize = 0;
};

struct ProfileRequirements {
    bool requireTraining = true;
    bool allowReferenceBackend = false;
};

enum class ProfileSupportStatus : std::uint8_t {
    Supported,
    ProfileNotImplemented,
    ReferenceBackendNotAllowed,
    UnsupportedBackend,
    MissingRequiredFeature,
    MissingTrainingAtomicFloat,
    MissingSubgroupArithmetic,
    MissingSubgroupSize,
    MissingCooperativeMatrixShape,
    MissingSharedMemory,
};

enum class TrainingImplementationKind : std::uint8_t {
    None,
    KhrGroundwork,
    KhrAllCoopmat,
    ScalarReference,
};

enum class TrainingProfileSelectionStatus : std::uint8_t {
    SelectedKhrGroundwork,
    SelectedKhrAllCoopmat,
    SelectedScalarReference,
    NoSupportedTrainingProfile,
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
    bool implemented = false;
};

struct ProfileShaderConfig {
    ProfileId id = ProfileId::None;
    BackendKind backend = BackendKind::None;
    const char *shaderHeader = "";
    std::uint32_t workgroupSize = 0;
    std::uint32_t subgroupSize = 0;
    std::uint32_t cooperativeMatrixM = 0;
    std::uint32_t cooperativeMatrixN = 0;
    std::uint32_t cooperativeMatrixK = 0;
    bool requireFullSubgroups = false;
    bool requireSubgroupSizeControl = false;
    bool requireSubgroupArithmeticForTraining = false;
};

struct ProfileSupportReport {
    ProfileId profileId = ProfileId::None;
    const char *profileName = "NONE";
    ProfileSupportStatus status = ProfileSupportStatus::UnsupportedBackend;
    bool supported = false;
};

struct TrainingProfileSelectionOptions {
    bool enableExperimentalAllCoopmat = false;
    bool allowKhrGroundwork = true;
    bool allowScalarReferenceFallback = false;
};

struct TrainingProfileSelection {
    TrainingProfileSelectionStatus status =
        TrainingProfileSelectionStatus::NoSupportedTrainingProfile;
    TrainingImplementationKind implementation = TrainingImplementationKind::None;
    ProfileId profileId = ProfileId::None;
    const char *profileName = "NONE";
    const ProfileShaderConfig *shaderConfig = nullptr;
    ProfileSupportReport khrProfileSupport;
    ProfileSupportReport scalarProfileSupport;
    std::uint32_t requiredSharedMemoryBytes = 0;
    bool selected = false;
    bool fallback = false;
    bool useKhrCoopmatAllDWeights = false;
    bool useKhrCoopmatAllDActivation = false;
};

std::span<const Profile> profiles();
const Profile *find_profile(ProfileId id);
std::span<const ProfileShaderConfig> profile_shader_configs();
const ProfileShaderConfig *find_profile_shader_config(ProfileId id);
std::uint32_t required_training_shared_memory_bytes(const Profile &profile);
ProfileSupportReport evaluate_profile_support(const DeviceCapabilitiesView &capabilities,
                                               const ProfileRequirements &requirements,
                                               const Profile &profile);
bool supports_profile(const DeviceCapabilitiesView &capabilities,
                      const ProfileRequirements &requirements,
                      const Profile &profile);
ProfileId choose_profile(const DeviceCapabilitiesView &capabilities,
                         const ProfileRequirements &requirements = {});
TrainingProfileSelection choose_training_profile(
    const DeviceCapabilitiesView &capabilities,
    const TrainingProfileSelectionOptions &options = {});

} // namespace nrc
