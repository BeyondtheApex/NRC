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
- shader module creation status supplied by the caller;
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
validation.requireShaderModuleCreation = true;
validation.requirePipelineCreation = true;
validation.requireNumericalValidation = true;
validation.shaderModuleCreation.status = nrc::vulkan::VulkanValidationStepStatus::Passed;
validation.pipelineCreation.status = nrc::vulkan::VulkanValidationStepStatus::Passed;
validation.numericalValidation.status = nrc::vulkan::VulkanValidationStepStatus::Passed;

nrc::vulkan::VulkanProfileValidationReport report =
    nrc::vulkan::evaluate_vulkan_profile_validation(caps, {}, *profile, validation);
```

`report.usable` is true only when the profile capabilities and shader config are valid and all required validation stages have passed. Failed stages always make the report unusable. `NotRun` and `Skipped` stages only block usability when the corresponding `require...` flag is set.

## Shader Module Helper

`nrc::vulkan::create_shader_module` creates a `VkShaderModule` from caller-supplied SPIR-V words and returns a `VulkanValidationStep`.

Header:

```cpp
#include "nrc/vulkan/nrc_vulkan_shader.hpp"
```

The helper does not read files or own the returned shader module. Callers keep control of their asset system and must destroy the result with `vkDestroyShaderModule`.

## Storage Buffer Descriptor Helper

`nrc::vulkan::create_storage_buffer_descriptor_set_layout` creates a descriptor set layout from explicit storage-buffer bindings. `nrc::vulkan::update_storage_buffer_descriptors` writes storage-buffer descriptors into a caller-allocated descriptor set.

Header:

```cpp
#include "nrc/vulkan/nrc_vulkan_descriptors.hpp"
```

The helper is intentionally narrow:

- storage buffers only;
- one descriptor per binding;
- caller-owned descriptor pools and descriptor sets;
- caller-owned layout destruction through `vkDestroyDescriptorSetLayout`.

This covers the scalar and KHR forward dispatch validation path while leaving image descriptors, samplers, bindless arrays, and engine-specific descriptor allocation policies to the host renderer.

## Memory Barrier Helper

`nrc::vulkan::cmd_memory_barrier` records a single `VkMemoryBarrier` and returns a `VulkanValidationStep`.

Header:

```cpp
#include "nrc/vulkan/nrc_vulkan_barriers.hpp"
```

Common presets are available for the dispatch validation path:

- `host_write_to_compute_shader_barrier`
- `compute_shader_write_to_host_read_barrier`
- `compute_shader_write_to_compute_shader_barrier`

The helper is deliberately limited to memory barriers. It does not perform image layout transitions or buffer/image memory barriers yet.

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

This module reports capabilities, aggregates profile validation state, creates shader modules from already-loaded SPIR-V words, writes simple storage-buffer descriptors, records simple memory barriers, and provides a small profile-aware compute pipeline creation helper. It does not create devices, enable features, compile shaders, read shader files, create pipeline layouts, allocate descriptor pools/sets, create resources, perform image layout transitions, or dispatch NRC workloads. Applications and tests should run those Vulkan operations through their own renderer code, then feed shader/pipeline/numerical outcomes back into `VulkanProfileValidationRequirements`.
