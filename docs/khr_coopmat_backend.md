# KHR Cooperative Matrix Backend

The first KHR backend is `shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl`.

## Current Scope

Implemented:

- Forward inference.
- 64 input values.
- 5 hidden ReLU layers of width 64.
- 3 output values.
- FP16 cooperative matrices.
- `M16N16K16` and `M8N16K16` subgroup matrices.
- Workgroup size 128.
- Subgroup size 16 or 32, selected by `NRC_KHR_SUBGROUP_SIZE`.
- Matrix M size selected by `NRC_KHR_COOPMAT_M`.
- Backpropagation ABI groundwork behind `NRC_NN_BACKPROPAGATION`:
  - writable fp32 gradient buffer binding;
  - scalar relative-L2 luminance output derivative;
  - fp32 storage-buffer atomic add helper;
  - activation snapshots for hidden layers 1, 2, 3, and 4 plus the output source activations;
  - encoded-input source staging for hidden-layer 0 `dW`;
  - subgroup-arithmetic output-layer `dW` accumulation in the shared row-major weight layout;
  - scalar output-derivative propagation to final hidden activations;
  - subgroup-arithmetic final hidden-layer `dW` accumulation in the shared row-major weight layout;
  - fp32 shared `dZ` scratch that reuses the active tile and retired final-hidden activation snapshot;
  - subgroup-arithmetic hidden-layer 3/2/1/0 `dW` accumulation in the shared row-major weight layout;
  - compile-validated experimental output-layer cooperative-matrix `dW` accumulation behind `NRC_KHR_USE_COOPMAT_OUTPUT_DW`;
  - compile-validated experimental final-hidden cooperative-matrix `dW` tile-streaming helper behind `NRC_KHR_USE_COOPMAT_FINAL_HIDDEN_DW`;
  - generic scalar hidden-layer derivative propagation helper;
  - enlarged shared-buffer allocation for future cooperative-matrix reductions.
- `shaders/nrc/passes/NrcGradientKhr.glsl`, a train-record pass skeleton that dispatches KHR forward and accumulates every hidden/output `dW` region.
- `nrc::reference_backward_khr_fp16`, a host-side fp16-rounded CPU oracle for the current KHR gradient path.

Not implemented yet:

- Default-path cooperative-matrix `dW` reduction beyond the current subgroup reductions.
- Cooperative-matrix gradient accumulation/reduction for hidden regions before the final hidden layer.
- Cooperative-matrix `dA` propagation matching VkNRC's fully fused NV path.
- Production-default KHR training cooperative-matrix reduction path.
- Runtime numerical validation on a driver that can create the KHR cooperative matrix pipelines successfully.

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

CTest also compiles the same three variants with `NRC_NN_BACKPROPAGATION` enabled through `tests/shader_compile/nrc_khr_backprop_smoke.comp`. The backprop smoke test exercises the gradient buffer declaration, output derivative, output-layer subgroup `dW` reduction helper, output-to-hidden derivative propagation, final-hidden subgroup `dW` reduction helper, shared `dZ` scratch, hidden-layer 3/2/1 subgroup `dW` reduction helpers, hidden-layer 0 encoded-source staging and subgroup `dW` reduction helper, atomic add helper, and forward path in one translation unit.

`tests/shader_compile/nrc_gradient_khr_dispatch.comp` compile-validates the KHR train-record pass for the same three profile shader shapes. It proves the pass-level binding ABI, push constants, train-count handling, integration-provided unpack hook, current hidden/output subgroup `dW` reductions, and encoded-source staging compile together.

The same dispatch smoke file is also compile-tested with `NRC_KHR_USE_COOPMAT_OUTPUT_DW=1` for SG32, SG16, and M8/SG32. These tests prove that the optional output-layer cooperative-matrix `dW` path type-checks for the currently implemented KHR profile shapes. They do not make it the default runtime path.

The dispatch smoke file is also compile-tested with `NRC_KHR_USE_COOPMAT_FINAL_HIDDEN_DW=1` for SG32, SG16, and M8/SG32. These tests prove that the final-hidden cooperative-matrix tile-streaming helper type-checks in the full train-record pass. The default runtime path still uses the subgroup final-hidden helper.

`tests/khr_gradient_reference_test.cpp` validates the reusable host-side KHR fp16 oracle. It proves the CPU oracle covers every hidden/output gradient region before that oracle is used by the Vulkan KHR runtime test.

`tests/vulkan_gradient_khr_test.cpp` also has a `--coopmat-output` mode. CTest registers it as `nrc_vulkan_gradient_khr_coopmat_output_test` with SPIR-V variants compiled using `NRC_KHR_USE_COOPMAT_OUTPUT_DW=1`. That mode compares output-layer gradients against a test-side oracle that mirrors fp16 output derivative staging and fp16 output partial storage, while hidden-layer gradients still use `nrc::reference_backward_khr_fp16`.

The optional output-coopmat helper restores the fp32 shared output derivative cache after reducing output dW. This keeps the downstream final-hidden dW helper on the same shared derivative contract as the default subgroup output path.

`nrc::compute_khr_shader_shared_memory_layout` mirrors the backend's shared-memory constants on the host side. The current KHR backprop layout allocates 6144 `uvec4` values, or 98304 bytes, for the active tile, six activation-sized regions, and reduction scratch. `docs/khr_shared_memory_layout.md` records the per-profile scratch sizes and the hidden dW partial-storage constraint.

Training profile selection now requires device `maxComputeSharedMemorySize` to meet that 98304-byte allocation. Inference-only selection can still be requested without the training shared-memory requirement.

The same layout model now records a hidden-layer cooperative-matrix `dW` tile-streaming plan. It avoids the SG16 full-layer partial overwrite hazard by using only the active tile for row-block dPre staging and one tile of partials per subgroup.

`NrcKhrAccumulateHiddenDWeightsCoopmatTileStreaming` implements the first shader helper using that plan. It currently has compile validation for final-hidden `dW`; runtime numerical validation and earlier hidden-layer integration are still pending.

These tests prove that the GLSL syntax, cooperative matrix types, and include structure are valid for the local Vulkan SDK. They do not prove runtime support, gradient correctness, or training performance.

## Runtime Dispatch Test

Milestone 8 adds `tests/vulkan_khr_dispatch_test.cpp` and `tests/shader_compile/nrc_khr_dispatch.comp`.

The test:

- Builds SG32, SG16, and M8/SG32 SPIR-V variants.
- Queries real Vulkan device capabilities through `nrc::vulkan::query_device_capabilities`.
- Selects an implemented KHR inference profile with `requireTraining = false`.
- Creates the device with KHR cooperative matrix, shader float16, storage buffer 16-bit access, subgroup size control, and full compute subgroup features.
- Creates and updates the storage-buffer descriptor set layout through the reusable Vulkan descriptor helpers.
- Creates the shader module through `nrc::vulkan::create_shader_module` and records the result in `VulkanProfileValidationReport`.
- Creates the compute pipeline through `nrc::vulkan::create_profile_compute_pipeline`, which applies `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` and `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT` from the selected `ProfileShaderConfig`.
- Records host-to-shader and shader-to-host memory barriers through `nrc::vulkan::cmd_memory_barrier`.
- Dispatches 149 samples and compares against an fp16-rounded CPU reference if pipeline creation and dispatch succeed.

Current local result:

- Device: AMD Radeon RX 9070 XT.
- Driver: AMD proprietary driver 26.6.4 (LLPC).
- Vulkan API: 1.4.349.
- The device reports an implemented `KHR_FP16_M16N16K16_SG32_WG128` profile.
- `vkCreateComputePipelines` raises a structured exception for the VkNRC-style KHR cooperative matrix shader. The test catches this on MSVC/Windows, prints `SKIP:`, and returns success so the rest of the suite remains usable.

This means capability/profile selection is necessary but not sufficient for production enablement. A downstream integration should compile or otherwise validate the selected KHR pipeline before enabling it for rendering.

## Runtime Gradient Test

`tests/vulkan_gradient_khr_test.cpp` builds SG32, SG16, and M8/SG32 SPIR-V variants for `tests/shader_compile/nrc_gradient_khr_dispatch.comp`.

The test:

- queries real Vulkan device capabilities;
- selects an implemented KHR training profile with `requireTraining = true`;
- creates a device with KHR cooperative matrix, fp32 storage-buffer atomic add, shader float16, storage buffer 16-bit access, subgroup size control, and full compute subgroup features, and requires subgroup arithmetic support reported by subgroup properties;
- creates the KHR gradient pipeline through `nrc::vulkan::create_profile_compute_pipeline`;
- dispatches a small train-record batch through `NrcGradientKhr.glsl`;
- compares every hidden/output gradient region against `nrc::reference_backward_khr_fp16` if pipeline creation and dispatch succeed.

Current local result:

- `vkCreateComputePipelines` raises a structured exception for the KHR gradient shader.
- The default KHR gradient test and the optional output-coopmat KHR gradient test catch this on MSVC/Windows, print `SKIP:`, and return success so the rest of the suite remains usable.

## Profile Selection

Profiles carry an `implemented` flag. The implemented KHR inference profiles are:

- `KHR_FP16_M16N16K16_SG32_WG128`
- `KHR_FP16_M16N16K16_SG16_WG128`
- `KHR_FP16_M8N16K16_SG32_WG128`

`choose_profile` still prefers the M16 profiles when the device reports both M16 and M8 support. M8 is selected when it is the first implemented matching profile for the reported matrix properties and subgroup size.

After selecting a profile, integrations should call `nrc::find_profile_shader_config` and use the returned workgroup, subgroup, and cooperative-matrix dimensions to select or compile the matching shader variant. The Vulkan KHR dispatch test uses this path to choose between SG32, SG16, and M8/SG32 SPIR-V files, then uses `nrc::vulkan::create_profile_compute_pipeline` for profile-specific pipeline creation.

Training profile selection additionally requires subgroup arithmetic support because the current KHR gradient helpers use `subgroupAdd` for hidden/output `dW` reductions. Inference-only selection can pass `ProfileRequirements{.requireTraining = false}` and does not require subgroup arithmetic or fp32 buffer atomic add.
