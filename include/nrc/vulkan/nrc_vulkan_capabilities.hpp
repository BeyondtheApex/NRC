#pragma once

#include "nrc/nrc_profile.hpp"

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

} // namespace nrc::vulkan
