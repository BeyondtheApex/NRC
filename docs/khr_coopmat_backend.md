# KHR Cooperative Matrix Backend

The first KHR backend is `shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl`.

## Current Scope

Implemented:

- Forward inference only.
- 64 input values.
- 5 hidden ReLU layers of width 64.
- 3 output values.
- FP16 cooperative matrices.
- 16x16x16 subgroup matrices.
- Workgroup size 128.
- Subgroup size 16 or 32, selected by `NRC_KHR_SUBGROUP_SIZE`.

Not implemented yet:

- 8x16x16 profile.
- Backpropagation.
- Gradient accumulation.
- Optimizer.
- Runtime numerical validation on a driver that can create the KHR cooperative matrix pipeline successfully.

## Weight Layout

The backend consumes the same packed fp16 `uvec4` layout documented in `weight_layout.md`.

Hidden layers are loaded as full 64x64 matrices.

The output layer uses a 16x64 cooperative matrix tile. Only the first 3 rows are populated from the packed 3x64 output weights; remaining rows are zero-filled.

## Compile Tests

CTest currently compiles the backend twice:

```text
NRC_KHR_SUBGROUP_SIZE=32
NRC_KHR_SUBGROUP_SIZE=16
```

These tests prove that the GLSL syntax, cooperative matrix types, and include structure are valid for the local Vulkan SDK. They do not prove runtime support or numerical correctness.

## Runtime Dispatch Test

Milestone 8 adds `tests/vulkan_khr_dispatch_test.cpp` and `tests/shader_compile/nrc_khr_dispatch.comp`.

The test:

- Builds SG32 and SG16 SPIR-V variants.
- Queries real Vulkan device capabilities through `nrc::vulkan::query_device_capabilities`.
- Selects an implemented KHR inference profile with `requireTraining = false`.
- Creates the device with KHR cooperative matrix, shader float16, storage buffer 16-bit access, subgroup size control, and full compute subgroup features.
- Creates the compute pipeline with `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` and `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT`.
- Dispatches 149 samples and compares against an fp16-rounded CPU reference if pipeline creation and dispatch succeed.

Current local result:

- Device: AMD Radeon RX 9070 XT.
- Driver: AMD proprietary driver 26.6.4 (LLPC).
- Vulkan API: 1.4.349.
- The device reports an implemented `KHR_FP16_M16N16K16_SG32_WG128` profile.
- `vkCreateComputePipelines` raises a structured exception for the VkNRC-style KHR cooperative matrix shader. The test catches this on MSVC/Windows, prints `SKIP:`, and returns success so the rest of the suite remains usable.

This means capability/profile selection is necessary but not sufficient for production enablement. A downstream integration should compile or otherwise validate the selected KHR pipeline before enabling it for rendering.

## Profile Selection

Profiles now carry an `implemented` flag. The 16x16x16 KHR SG16/SG32 profiles are implemented. The M8 profile remains in the profile list as a planned target but is not selected by `choose_profile` yet.
