#pragma once

#include "nrc/nrc_profile.hpp"

#include <cstdint>
#include <optional>
#include <vector>
#include <vulkan/vulkan.h>

namespace nrc::vulkan {

struct VulkanDeviceCapabilities {
    bool shaderFloat16 = false;
    bool storageBuffer16BitAccess = false;
    bool shaderBufferFloat32AtomicAdd = false;
    bool subgroupSizeControl = false;
    bool computeFullSubgroups = false;
    bool khrCooperativeMatrix = false;
    std::vector<std::uint32_t> supportedSubgroupSizes;
    std::vector<CooperativeMatrixCapability> cooperativeMatrices;

    DeviceCapabilitiesView view() const;
};

enum class VulkanValidationStepStatus : std::uint8_t {
    NotRun,
    Passed,
    Failed,
    Skipped,
};

enum class VulkanProfileValidationStatus : std::uint8_t {
    Supported,
    ProfileNotSupported,
    MissingShaderConfig,
    ShaderConfigMismatch,
    PipelineCreationNotRun,
    PipelineCreationSkipped,
    PipelineCreationFailed,
    NumericalValidationNotRun,
    NumericalValidationSkipped,
    NumericalValidationFailed,
};

struct VulkanValidationStep {
    VulkanValidationStepStatus status = VulkanValidationStepStatus::NotRun;
    VkResult vkResult = VK_SUCCESS;
    bool structuredException = false;
};

struct VulkanProfileValidationRequirements {
    bool requirePipelineCreation = false;
    bool requireNumericalValidation = false;
    VulkanValidationStep pipelineCreation;
    VulkanValidationStep numericalValidation;
};

struct VulkanProfileValidationReport {
    ProfileSupportReport profileSupport;
    const ProfileShaderConfig *shaderConfig = nullptr;
    VulkanValidationStep pipelineCreation;
    VulkanValidationStep numericalValidation;
    VulkanProfileValidationStatus status = VulkanProfileValidationStatus::ProfileNotSupported;
    bool usable = false;
};

ComponentType map_component_type(VkComponentTypeKHR type);
std::vector<std::uint32_t> enumerate_subgroup_sizes(std::uint32_t minSize,
                                                    std::uint32_t maxSize,
                                                    std::uint32_t fallbackSize);
std::optional<CooperativeMatrixCapability>
translate_cooperative_matrix_property(const VkCooperativeMatrixPropertiesKHR &property,
                                      VkShaderStageFlags supportedStages);

VulkanDeviceCapabilities query_device_capabilities(
    VkPhysicalDevice physicalDevice,
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR getCooperativeMatrixProperties = nullptr);

VulkanProfileValidationReport evaluate_vulkan_profile_validation(
    const VulkanDeviceCapabilities &capabilities,
    const ProfileRequirements &profileRequirements,
    const Profile &profile,
    const VulkanProfileValidationRequirements &validationRequirements = {});

} // namespace nrc::vulkan
