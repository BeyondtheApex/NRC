# Vulkan Capability Query

The optional `nrc_vulkan` target translates Vulkan device capabilities into the renderer-neutral `nrc::DeviceCapabilitiesView`.

Header:

```cpp
#include "nrc/vulkan/nrc_vulkan_capabilities.hpp"
```

Basic use:

```cpp
nrc::vulkan::VulkanDeviceCapabilities caps =
    nrc::vulkan::query_device_capabilities(physicalDevice, getCoopMatrixPropertiesKHR);

nrc::ProfileId profile = nrc::choose_profile(caps.view());
```

`getCoopMatrixPropertiesKHR` should usually come from the application's Vulkan loader, for example volk or `vkGetInstanceProcAddr`.

The function pointer is optional, but passing it explicitly is recommended because extension commands are not guaranteed to link from `vulkan-1.lib`.

## Features Queried

- `shaderFloat16`
- `storageBuffer16BitAccess`
- `shaderBufferFloat32AtomicAdd`
- `subgroupSizeControl`
- `computeFullSubgroups`
- `VK_KHR_cooperative_matrix`

## Cooperative Matrix Translation

Only compute-stage, subgroup-scope matrix properties are translated into NRC capabilities.

The first production profiles expect FP16 matrices such as:

```text
M=16 N=16 K=16 scope=subgroup
M=8  N=16 K=16 scope=subgroup
```

Unsupported component types are mapped to `ComponentType::Unknown` and therefore will not match production profiles.

## Profile Validation Report

`nrc::vulkan::evaluate_vulkan_profile_validation` combines the renderer-neutral profile support report with the Vulkan shader configuration and optional validation stages:

- capability/profile support from `nrc::evaluate_profile_support`;
- shader config lookup through `nrc::find_profile_shader_config`;
- pipeline creation status supplied by the caller;
- numerical validation status supplied by the caller.

Basic capability/config check:

```cpp
const nrc::Profile *profile = nrc::find_profile(nrc::choose_profile(caps.view()));
if (profile != nullptr) {
    nrc::vulkan::VulkanProfileValidationReport report =
        nrc::vulkan::evaluate_vulkan_profile_validation(caps, {}, *profile);
}
```

Strict integration check after a renderer has tried to create and validate its pipeline:

```cpp
nrc::vulkan::VulkanProfileValidationRequirements validation{};
validation.requirePipelineCreation = true;
validation.requireNumericalValidation = true;
validation.pipelineCreation.status = nrc::vulkan::VulkanValidationStepStatus::Passed;
validation.numericalValidation.status = nrc::vulkan::VulkanValidationStepStatus::Passed;

nrc::vulkan::VulkanProfileValidationReport report =
    nrc::vulkan::evaluate_vulkan_profile_validation(caps, {}, *profile, validation);
```

`report.usable` is true only when the profile capabilities and shader config are valid and all required validation stages have passed. Failed stages always make the report unusable. `NotRun` and `Skipped` stages only block usability when the corresponding `require...` flag is set.

## Profile Compute Pipeline Helper

`nrc::vulkan::create_profile_compute_pipeline` creates a compute pipeline from a caller-owned device, shader module, pipeline layout, and `ProfileShaderConfig`.

Header:

```cpp
#include "nrc/vulkan/nrc_vulkan_pipeline.hpp"
```

For KHR cooperative matrix profiles, the helper applies the shader-stage requirements carried by the profile config:

- `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` when subgroup size control is required;
- `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT` when full subgroups are required.

The result contains:

- created `VkPipeline`;
- `VulkanValidationStepStatus`;
- `VkResult`;
- `structuredException`, set when MSVC/Windows catches a structured exception from `vkCreateComputePipelines`.

The helper does not own the returned pipeline. Callers must destroy it with `vkDestroyPipeline`.

## Current Limit

This module reports capabilities, aggregates profile validation state, and provides a small profile-aware compute pipeline creation helper. It does not create devices, enable features, compile shaders, create shader modules, create pipeline layouts, or dispatch NRC workloads. Applications and tests should run those Vulkan operations through their own renderer code, then feed pipeline/numerical outcomes back into `VulkanProfileValidationRequirements`.
