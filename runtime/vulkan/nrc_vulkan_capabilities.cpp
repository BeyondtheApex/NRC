#include "nrc/vulkan/nrc_vulkan_capabilities.hpp"

#include <algorithm>
#include <limits>
#include <string_view>
#include <vector>

namespace nrc::vulkan {

namespace {

bool has_extension(VkPhysicalDevice physicalDevice, const char *name) {
    std::uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr) != VK_SUCCESS) {
        return false;
    }

    std::vector<VkExtensionProperties> extensions(count);
    if (vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, extensions.data()) != VK_SUCCESS) {
        return false;
    }

    return std::any_of(extensions.begin(), extensions.end(), [&](const VkExtensionProperties &extension) {
        return std::string_view(extension.extensionName) == name;
    });
}

bool is_failed(VulkanValidationStepStatus status) {
    return status == VulkanValidationStepStatus::Failed;
}

bool is_skipped(VulkanValidationStepStatus status) {
    return status == VulkanValidationStepStatus::Skipped;
}

bool is_not_run(VulkanValidationStepStatus status) {
    return status == VulkanValidationStepStatus::NotRun;
}

} // namespace

DeviceCapabilitiesView VulkanDeviceCapabilities::view() const {
    return {
        shaderFloat16,
        storageBuffer16BitAccess,
        shaderBufferFloat32AtomicAdd,
        subgroupSizeControl,
        computeFullSubgroups,
        subgroupArithmetic,
        khrCooperativeMatrix,
        supportedSubgroupSizes,
        cooperativeMatrices,
        maxComputeSharedMemorySize,
    };
}

ComponentType map_component_type(VkComponentTypeKHR type) {
    switch (type) {
        case VK_COMPONENT_TYPE_FLOAT16_KHR: return ComponentType::Float16;
        case VK_COMPONENT_TYPE_FLOAT32_KHR: return ComponentType::Float32;
        default: return ComponentType::Unknown;
    }
}

std::vector<std::uint32_t> enumerate_subgroup_sizes(std::uint32_t minSize,
                                                    std::uint32_t maxSize,
                                                    std::uint32_t fallbackSize) {
    std::vector<std::uint32_t> sizes;
    if (minSize != 0 && maxSize >= minSize) {
        for (std::uint32_t size = minSize; size <= maxSize && size != 0; size <<= 1u) {
            sizes.push_back(size);
            if (size > (std::numeric_limits<std::uint32_t>::max() >> 1u)) {
                break;
            }
        }
    }

    if (sizes.empty() && fallbackSize != 0) {
        sizes.push_back(fallbackSize);
    }

    return sizes;
}

std::optional<CooperativeMatrixCapability>
translate_cooperative_matrix_property(const VkCooperativeMatrixPropertiesKHR &property,
                                      VkShaderStageFlags supportedStages) {
    if ((supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0) {
        return std::nullopt;
    }
    if (property.scope != VK_SCOPE_SUBGROUP_KHR) {
        return std::nullopt;
    }

    CooperativeMatrixCapability capability{};
    capability.m = property.MSize;
    capability.n = property.NSize;
    capability.k = property.KSize;
    capability.aType = map_component_type(property.AType);
    capability.bType = map_component_type(property.BType);
    capability.cType = map_component_type(property.CType);
    capability.resultType = map_component_type(property.ResultType);
    capability.scope = MatrixScope::Subgroup;
    capability.computeStage = true;
    return capability;
}

VulkanDeviceCapabilities query_device_capabilities(
    VkPhysicalDevice physicalDevice,
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR getCooperativeMatrixProperties) {
    VulkanDeviceCapabilities out{};
    if (physicalDevice == VK_NULL_HANDLE) {
        return out;
    }

    out.khrCooperativeMatrix = has_extension(physicalDevice, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    const bool hasAtomicFloat = has_extension(physicalDevice, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);

    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{};
    atomicFloatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperativeMatrixFeatures{};
    cooperativeMatrixFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    cooperativeMatrixFeatures.pNext = &atomicFloatFeatures;

    VkPhysicalDeviceVulkan13Features vulkan13Features{};
    vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan13Features.pNext = &cooperativeMatrixFeatures;

    VkPhysicalDeviceVulkan12Features vulkan12Features{};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Features.pNext = &vulkan13Features;

    VkPhysicalDeviceVulkan11Features vulkan11Features{};
    vulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    vulkan11Features.pNext = &vulkan12Features;

    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &vulkan11Features;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &features);

    out.shaderFloat16 = vulkan12Features.shaderFloat16 == VK_TRUE;
    out.storageBuffer16BitAccess = vulkan11Features.storageBuffer16BitAccess == VK_TRUE;
    out.shaderBufferFloat32AtomicAdd =
        hasAtomicFloat && atomicFloatFeatures.shaderBufferFloat32AtomicAdd == VK_TRUE;
    out.subgroupSizeControl = vulkan13Features.subgroupSizeControl == VK_TRUE;
    out.computeFullSubgroups = vulkan13Features.computeFullSubgroups == VK_TRUE;
    out.khrCooperativeMatrix =
        out.khrCooperativeMatrix && cooperativeMatrixFeatures.cooperativeMatrix == VK_TRUE;

    VkPhysicalDeviceSubgroupSizeControlProperties subgroupSizeControlProperties{};
    subgroupSizeControlProperties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;

    VkPhysicalDeviceSubgroupProperties subgroupProperties{};
    subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    subgroupProperties.pNext = &subgroupSizeControlProperties;

    VkPhysicalDeviceCooperativeMatrixPropertiesKHR cooperativeMatrixDeviceProperties{};
    cooperativeMatrixDeviceProperties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    cooperativeMatrixDeviceProperties.pNext = &subgroupProperties;

    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &cooperativeMatrixDeviceProperties;
    vkGetPhysicalDeviceProperties2(physicalDevice, &properties);
    out.maxComputeSharedMemorySize =
        properties.properties.limits.maxComputeSharedMemorySize;

    out.supportedSubgroupSizes = enumerate_subgroup_sizes(
        subgroupSizeControlProperties.minSubgroupSize,
        subgroupSizeControlProperties.maxSubgroupSize,
        subgroupProperties.subgroupSize);
    out.subgroupArithmetic =
        (subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;

    if (out.khrCooperativeMatrix && getCooperativeMatrixProperties == nullptr) {
        getCooperativeMatrixProperties =
            reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
                vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
    }

    if (out.khrCooperativeMatrix && getCooperativeMatrixProperties != nullptr) {
        std::uint32_t propertyCount = 0;
        if (getCooperativeMatrixProperties(physicalDevice, &propertyCount, nullptr) == VK_SUCCESS &&
            propertyCount > 0) {
            std::vector<VkCooperativeMatrixPropertiesKHR> propertiesList(propertyCount);
            for (auto &property : propertiesList) {
                property.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            }
            if (getCooperativeMatrixProperties(physicalDevice, &propertyCount, propertiesList.data()) == VK_SUCCESS) {
                for (const auto &property : propertiesList) {
                    auto translated = translate_cooperative_matrix_property(
                        property,
                        cooperativeMatrixDeviceProperties.cooperativeMatrixSupportedStages);
                    if (translated.has_value()) {
                        out.cooperativeMatrices.push_back(*translated);
                    }
                }
            }
        }
    }

    return out;
}

VulkanProfileValidationReport evaluate_vulkan_profile_validation(
    const VulkanDeviceCapabilities &capabilities,
    const ProfileRequirements &profileRequirements,
    const Profile &profile,
    const VulkanProfileValidationRequirements &validationRequirements) {
    VulkanProfileValidationReport report{};
    report.profileSupport =
        evaluate_profile_support(capabilities.view(), profileRequirements, profile);
    report.shaderModuleCreation = validationRequirements.shaderModuleCreation;
    report.pipelineCreation = validationRequirements.pipelineCreation;
    report.numericalValidation = validationRequirements.numericalValidation;

    if (!report.profileSupport.supported) {
        report.status = VulkanProfileValidationStatus::ProfileNotSupported;
        return report;
    }

    report.shaderConfig = find_profile_shader_config(profile.id);
    if (report.shaderConfig == nullptr) {
        report.status = VulkanProfileValidationStatus::MissingShaderConfig;
        return report;
    }

    if (report.shaderConfig->backend != profile.backend) {
        report.status = VulkanProfileValidationStatus::ShaderConfigMismatch;
        return report;
    }

    if (is_failed(report.shaderModuleCreation.status)) {
        report.status = VulkanProfileValidationStatus::ShaderModuleCreationFailed;
        return report;
    }
    if (validationRequirements.requireShaderModuleCreation) {
        if (is_not_run(report.shaderModuleCreation.status)) {
            report.status = VulkanProfileValidationStatus::ShaderModuleCreationNotRun;
            return report;
        }
        if (is_skipped(report.shaderModuleCreation.status)) {
            report.status = VulkanProfileValidationStatus::ShaderModuleCreationSkipped;
            return report;
        }
    }

    if (is_failed(report.pipelineCreation.status)) {
        report.status = VulkanProfileValidationStatus::PipelineCreationFailed;
        return report;
    }
    if (validationRequirements.requirePipelineCreation) {
        if (is_not_run(report.pipelineCreation.status)) {
            report.status = VulkanProfileValidationStatus::PipelineCreationNotRun;
            return report;
        }
        if (is_skipped(report.pipelineCreation.status)) {
            report.status = VulkanProfileValidationStatus::PipelineCreationSkipped;
            return report;
        }
    }

    if (is_failed(report.numericalValidation.status)) {
        report.status = VulkanProfileValidationStatus::NumericalValidationFailed;
        return report;
    }
    if (validationRequirements.requireNumericalValidation) {
        if (is_not_run(report.numericalValidation.status)) {
            report.status = VulkanProfileValidationStatus::NumericalValidationNotRun;
            return report;
        }
        if (is_skipped(report.numericalValidation.status)) {
            report.status = VulkanProfileValidationStatus::NumericalValidationSkipped;
            return report;
        }
    }

    report.status = VulkanProfileValidationStatus::Supported;
    report.usable = true;
    return report;
}

} // namespace nrc::vulkan
