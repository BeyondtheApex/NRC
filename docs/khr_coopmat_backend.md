# KHR Cooperative Matrix Backend

The first KHR backend is `shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl`.

## Current Scope

Implemented:

- Forward inference only.
- 64 input values.
- 5 hidden ReLU layers of width 64.
- 3 output values.
- FP16 cooperative matrices.
- `M16N16K16` and `M8N16K16` subgroup matrices.
- Workgroup size 128.
- Subgroup size 16 or 32, selected by `NRC_KHR_SUBGROUP_SIZE`.
- Matrix M size selected by `NRC_KHR_COOPMAT_M`.

Not implemented yet:

- Backpropagation.
- Gradient accumulation.
- Optimizer.
- Runtime numerical validation on a driver that can create the KHR cooperative matrix pipeline successfully.

## Weight Layout

The backend consumes the same packed fp16 `uvec4` layout documented in `weight_layout.md`.

Hidden layers are loaded as full 64x64 matrices. The backend no longer assumes `M == K`: hidden-layer outputs are stored back into the shared activation tile and reloaded through the K dimension before the next layer. This keeps the layout stable for both `M16N16K16` and `M8N16K16`.

The output layer uses the first M-row cooperative matrix tile. Only the first 3 rows are populated from the packed 3x64 output weights; remaining rows are zero-filled.

## Compile Tests

CTest currently compiles the backend variants:

```text
NRC_KHR_SUBGROUP_SIZE=32
NRC_KHR_SUBGROUP_SIZE=16
NRC_KHR_SUBGROUP_SIZE=32, NRC_KHR_COOPMAT_M=8
```

These tests prove that the GLSL syntax, cooperative matrix types, and include structure are valid for the local Vulkan SDK. They do not prove runtime support or numerical correctness.

## Runtime Dispatch Test

Milestone 8 adds `tests/vulkan_khr_dispatch_test.cpp` and `tests/shader_compile/nrc_khr_dispatch.comp`.

The test:

- Builds SG32, SG16, and M8/SG32 SPIR-V variants.
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

Profiles carry an `implemented` flag. The implemented KHR inference profiles are:

- `KHR_FP16_M16N16K16_SG32_WG128`
- `KHR_FP16_M16N16K16_SG16_WG128`
- `KHR_FP16_M8N16K16_SG32_WG128`

`choose_profile` still prefers the M16 profiles when the device reports both M16 and M8 support. M8 is selected when it is the first implemented matching profile for the reported matrix properties and subgroup size.
