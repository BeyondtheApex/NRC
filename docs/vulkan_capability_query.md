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
nrc::TrainingProfileSelection training =
    nrc::choose_training_profile(caps.view());
nrc::vulkan::VulkanTrainingProfilePlan trainingPlan =
    nrc::vulkan::make_vulkan_training_profile_plan(caps);
```

`getCoopMatrixPropertiesKHR` should usually come from the application's Vulkan loader, for example volk or `vkGetInstanceProcAddr`.

The function pointer is optional, but passing it explicitly is recommended because extension commands are not guaranteed to link from `vulkan-1.lib`.

## Features Queried

- `shaderFloat16`
- `storageBuffer16BitAccess`
- `shaderBufferFloat32AtomicAdd`
- `subgroupSizeControl`
- `computeFullSubgroups`
- `subgroupArithmetic`
- `VK_KHR_cooperative_matrix`
- `maxComputeSharedMemorySize`

`subgroupArithmetic` is translated from
`VkPhysicalDeviceSubgroupProperties::supportedOperations &
VK_SUBGROUP_FEATURE_ARITHMETIC_BIT`. It is required by NRC training profiles
because the current KHR gradient path uses subgroup reductions. KHR inference
selection can still be requested with `ProfileRequirements::requireTraining =
false`.

`maxComputeSharedMemorySize` is read from
`VkPhysicalDeviceProperties::limits.maxComputeSharedMemorySize`. Current KHR
training profiles compare it against the host-side KHR shader shared-memory
layout. The active backprop layout requires 98304 bytes, while inference-only
selection does not require that training allocation.

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

## Training Selection Policy

Use `nrc::choose_training_profile` when integrating the training path rather than directly treating `choose_profile` as a production policy decision:

```cpp
nrc::TrainingProfileSelectionOptions options{};
options.enableExperimentalAllCoopmat = false;
options.allowKhrGroundwork = true;
options.allowScalarReferenceFallback = false;

nrc::TrainingProfileSelection selection =
    nrc::choose_training_profile(caps.view(), options);
```

The default selection chooses the supported KHR training shader profile but keeps the current KHR groundwork gradient behavior. Enabling `enableExperimentalAllCoopmat` selects the same KHR profile with both all-coopmat compile switches requested:

```text
NRC_KHR_USE_COOPMAT_ALL_DW=1
NRC_KHR_USE_COOPMAT_ALL_DA=1
```

The selection carries the decisive KHR `ProfileSupportReport`, so integrations can log why a device rejected KHR training, such as missing subgroup arithmetic, missing matrix shape, or insufficient `maxComputeSharedMemorySize`. Scalar fallback is available only when `allowScalarReferenceFallback` is set.

`nrc::vulkan::make_vulkan_training_profile_plan` combines the renderer-neutral selection with a shader compile config and a Vulkan validation report:

```cpp
nrc::vulkan::VulkanProfileValidationRequirements validation{};
validation.requirePipelineCreation = true;
validation.pipelineCreation.status = pipelineCreated
    ? nrc::vulkan::VulkanValidationStepStatus::Passed
    : nrc::vulkan::VulkanValidationStepStatus::Failed;

nrc::vulkan::VulkanTrainingProfilePlan plan =
    nrc::vulkan::make_vulkan_training_profile_plan(caps, options, validation);
```

`plan.shaderCompile` is the integration-facing shader plan. For KHR profiles it carries the selected workgroup size, subgroup size, cooperative matrix shape, full-subgroup requirement, and all-coopmat switches. `plan.usable` is true only when a training profile was selected, the shader compile config is valid, and the requested Vulkan validation stages are usable.

If an integration wants explicit scalar fallback after a KHR shader module, pipeline, or numerical validation failure, use `make_vulkan_training_profile_plan_with_runtime_fallback`:

```cpp
nrc::TrainingProfileSelectionOptions options{};
options.allowScalarReferenceFallback = true;

nrc::vulkan::VulkanProfileValidationRequirements khrValidation{};
khrValidation.requirePipelineCreation = true;
khrValidation.pipelineCreation = khrPipelineResult.validation;

nrc::vulkan::VulkanTrainingProfilePlan plan =
    nrc::vulkan::make_vulkan_training_profile_plan_with_runtime_fallback(
        caps,
        options,
        khrValidation);
```

This helper preserves the plain `make_vulkan_training_profile_plan` behavior unless all of the following are true:

- a KHR training implementation was selected;
- required KHR validation made that plan unusable;
- `allowScalarReferenceFallback` is true;
- the scalar reference profile can be selected.

When runtime fallback is used, `plan.usedRuntimeFallback` is true, `plan.selection` describes the selected scalar plan, and `plan.fallbackSourceSelection` plus `plan.fallbackSourceValidation` preserve the rejected KHR plan. This keeps fallback explicit in logs instead of silently erasing the failed KHR shader or pipeline result.

For integration logs, use:

- `nrc::training_implementation_kind_name(plan.selection.implementation)`
- `nrc::training_profile_selection_status_name(plan.selection.status)`
- `nrc::profile_support_status_name(plan.selection.khrProfileSupport.status)`
- `nrc::vulkan::vulkan_profile_validation_status_name(plan.validation.status)`
- `nrc::vulkan::vulkan_validation_step_status_name(plan.validation.pipelineCreation.status)`
- `nrc::vulkan::vulkan_profile_validation_status_name(plan.fallbackSourceValidation.status)` when `plan.usedRuntimeFallback` is true.

The helpers return stable ASCII enum tokens such as `KhrGroundwork`, `SelectedScalarReference`, `MissingSharedMemory`, and `PipelineCreationFailed`.

## Shader Module Helper

`nrc::vulkan::create_shader_module` creates a `VkShaderModule` from caller-supplied SPIR-V words and returns a `VulkanValidationStep`.

Header:

```cpp
#include "nrc/vulkan/nrc_vulkan_shader.hpp"
```

The helper does not read files or own the returned shader module. Callers keep control of their asset system and must destroy the result with `vkDestroyShaderModule`.

## Descriptor Helpers

`nrc::vulkan::create_descriptor_set_layout` creates a descriptor set layout from explicit descriptor bindings. `nrc::vulkan::create_storage_buffer_descriptor_set_layout` remains as a narrow storage-buffer convenience wrapper.

`nrc::vulkan::update_storage_buffer_descriptors` writes storage-buffer descriptors into a caller-allocated descriptor set. `nrc::vulkan::update_storage_image_descriptors` does the same for storage images.

Header:

```cpp
#include "nrc/vulkan/nrc_vulkan_descriptors.hpp"
```

The helpers are intentionally renderer-neutral:

- caller-owned descriptor pools and descriptor sets;
- caller-owned layout destruction through `vkDestroyDescriptorSetLayout`;
- one descriptor per storage-buffer or storage-image update write;
- generic layout creation for mixed descriptor sets;
- storage-buffer and storage-image update helpers for the current NRC validation paths.

They do not allocate descriptor pools/sets, bind descriptor sets, or impose a descriptor indexing/bindless policy.

Current descriptor update coverage:

- storage buffers;
- storage images.

Current descriptor update exclusions:

- sampled images;
- samplers;
- acceleration structures;
- descriptor arrays;
- bindless descriptors.

This covers the scalar/KHR forward dispatch validation path and the scalar inference storage-image path while leaving engine-specific descriptor allocation policies to the host renderer.

## Memory/Pipeline Barrier Helpers

`nrc::vulkan::cmd_memory_barrier` records a single `VkMemoryBarrier` and returns a `VulkanValidationStep`.

`nrc::vulkan::cmd_pipeline_barrier` records caller-specified memory access barriers and image barriers in one `vkCmdPipelineBarrier` call. `nrc::vulkan::color_image_barrier` builds a one-mip, one-layer color image barrier for common storage-image upload/compute/readback transitions.

Header:

```cpp
#include "nrc/vulkan/nrc_vulkan_barriers.hpp"
```

Common presets are available for the dispatch validation path:

- `host_write_to_compute_shader_barrier`
- `compute_shader_write_to_host_read_barrier`
- `compute_shader_write_to_compute_shader_barrier`

The helpers do not track resource states, allocate resources, or infer queue ownership. Callers pass explicit stages, access masks, layouts, and subresource ranges.

## Compute Pipeline Helpers

`nrc::vulkan::create_compute_pipeline` creates a plain compute pipeline from a caller-owned device, shader module, and pipeline layout. `nrc::vulkan::create_profile_compute_pipeline` adds NRC profile-specific shader-stage requirements on top of the same guarded creation path.

Header:

```cpp
#include "nrc/vulkan/nrc_vulkan_pipeline.hpp"
```

Use the plain helper for non-profile utility passes such as queue, prepare, gradient, and optimize dispatches.

For KHR cooperative matrix profiles, the profile helper applies the shader-stage requirements carried by the profile config:

- `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` when subgroup size control is required;
- `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT` when full subgroups are required.

The result contains:

- created `VkPipeline`;
- `VulkanValidationStepStatus`;
- `VkResult`;
- `structuredException`, set when MSVC/Windows catches a structured exception from `vkCreateComputePipelines`.

The helpers do not own returned pipelines. Callers must destroy them with `vkDestroyPipeline`.

## Current Limit

This module reports capabilities, aggregates profile validation state, creates shader modules from already-loaded SPIR-V words, creates descriptor set layouts, writes simple storage-buffer/storage-image descriptors, records simple memory/image pipeline barriers, and provides small compute pipeline creation helpers. It does not create devices, enable features, compile shaders, read shader files, create pipeline layouts, allocate descriptor pools/sets, create resources, track resource states, or dispatch NRC workloads. Applications and tests should run those Vulkan operations through their own renderer code, then feed shader/pipeline/numerical outcomes back into `VulkanProfileValidationRequirements`.
