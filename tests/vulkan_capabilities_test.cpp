#include "nrc/vulkan/nrc_vulkan_capabilities.hpp"

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

    ok &= require(nrc::vulkan::map_component_type(VK_COMPONENT_TYPE_FLOAT16_KHR) ==
                      nrc::ComponentType::Float16,
                  "map fp16 component type");
    ok &= require(nrc::vulkan::map_component_type(VK_COMPONENT_TYPE_FLOAT32_KHR) ==
                      nrc::ComponentType::Float32,
                  "map fp32 component type");

    const auto subgroupSizes = nrc::vulkan::enumerate_subgroup_sizes(16, 64, 32);
    ok &= require(subgroupSizes.size() == 3, "subgroup size count");
    ok &= require(subgroupSizes[0] == 16 && subgroupSizes[1] == 32 && subgroupSizes[2] == 64,
                  "subgroup size powers of two");

    VkCooperativeMatrixPropertiesKHR property{};
    property.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    property.MSize = 16;
    property.NSize = 16;
    property.KSize = 16;
    property.AType = VK_COMPONENT_TYPE_FLOAT16_KHR;
    property.BType = VK_COMPONENT_TYPE_FLOAT16_KHR;
    property.CType = VK_COMPONENT_TYPE_FLOAT16_KHR;
    property.ResultType = VK_COMPONENT_TYPE_FLOAT16_KHR;
    property.scope = VK_SCOPE_SUBGROUP_KHR;

    auto translated = nrc::vulkan::translate_cooperative_matrix_property(property, VK_SHADER_STAGE_COMPUTE_BIT);
    ok &= require(translated.has_value(), "translate compute subgroup cooperative matrix");
    if (translated.has_value()) {
        ok &= require(translated->m == 16 && translated->n == 16 && translated->k == 16,
                      "translated matrix dimensions");
        ok &= require(translated->aType == nrc::ComponentType::Float16,
                      "translated matrix component type");
        ok &= require(translated->computeStage, "translated compute stage");
    }

    auto noCompute = nrc::vulkan::translate_cooperative_matrix_property(property, VK_SHADER_STAGE_VERTEX_BIT);
    ok &= require(!noCompute.has_value(), "reject non-compute cooperative matrix property");

    property.scope = VK_SCOPE_WORKGROUP_KHR;
    auto noSubgroup = nrc::vulkan::translate_cooperative_matrix_property(property, VK_SHADER_STAGE_COMPUTE_BIT);
    ok &= require(!noSubgroup.has_value(), "reject non-subgroup cooperative matrix property");

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
