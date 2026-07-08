# NRC Implementation Log

All entries should be updated as work happens. The log is intentionally detailed so later audits can recover context after interruptions or context compaction.

## 2026-07-08 - Milestone 0 Bootstrap

### Request

The project goal is to build a complete reusable Neural Radiance Cache implementation with:

- Core NRC implementation.
- A test suite.
- A small example/reference path tracing pipeline used to validate integration quality and behavior.
- A structure suitable for later integration into projects similar to `G:/cpp/radiance/MCVR-custom`.
- Milestone-based git commits.
- Documentation that records every implementation step, current limits, and known follow-up work.

The user prefers starting from scratch if the reusable NRC code size is manageable, while strictly following the NRC paper and the existing `G:/cpp/radiance/VkNRC` behavior before doing later quality/performance iterations.

### Local Environment

- Visual Studio: `D:/Program Files/Microsoft Visual Studio/18/Community`
- Vulkan SDK: `D:/VulkanSDK/1.4.335.0`
- Workspace root: `G:/cpp/radiance`
- New repository path: `G:/cpp/radiance/NRC`
- Reference implementation: `G:/cpp/radiance/VkNRC`
- Target downstream-style integration reference: `G:/cpp/radiance/MCVR-custom`
- SHARC packaging reference: `G:/cpp/radiance/SHARC` and `G:/cpp/radiance/MCVR-custom/extern/sharc`

### Initial Repository State

- `G:/cpp/radiance` is not a git repository.
- `G:/cpp/radiance/VkNRC` is a git repository and was clean at bootstrap.
- `G:/cpp/radiance/MCVR-custom` has existing unrelated local modifications. This NRC repository must not modify or revert those changes unless explicitly requested.

### Initial Investigation Summary

- SHARC is distributed as shader-only source headers, but MCVR-custom still needs host-side runtime work: cache buffers, config buffers, descriptor binding, update pass dispatch, and resolve pass dispatch.
- MCVR-custom copies SHARC headers into built-in shader-pack staging at install/package time. Relevant reference: `G:/cpp/radiance/MCVR-custom/src/shader/CMakeLists.txt`.
- MCVR-custom's SHARC runtime currently lives inside `RayTracingModule` and creates hash, lock, accumulation, resolved, and per-frame config buffers.
- VkNRC is a compact reference for the actual NRC algorithm. The core NRC-related files measured during investigation total about 1747 lines, excluding Eigen, vuda, general scene loading, and framework code.
- VkNRC's current high-performance backend depends on `VK_NV_cooperative_matrix` / `GL_NV_cooperative_matrix`.
- VkNRC contains an experimental KHR cooperative matrix evaluation shader under `G:/cpp/radiance/VkNRC/test/evaluate.comp`, which can guide the first KHR backend.
- Local MCVR-custom Vulkan headers already expose `VK_KHR_cooperative_matrix`, including `VkPhysicalDeviceCooperativeMatrixFeaturesKHR` and `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`.

### Design Direction

Start a new repository from scratch, but treat VkNRC and the paper as the behavioral reference. Do not fork VkNRC directly because its code is tied to its sample renderer, `myvk_rg`, and NV cooperative matrix assumptions.

The reusable project should be layered as:

- `include/nrc`: public C++ ABI, constants, profiles, capability descriptions, and resource layouts.
- `shaders/nrc/common`: shader-side record formats, encoding, queues, loss, and optimizer helpers.
- `shaders/nrc/backend`: NN backends such as KHR cooperative matrix and scalar/reference.
- `shaders/nrc/passes`: inference, train-prepare, gradient, optimizer, and indirect dispatch compute shaders.
- `runtime/vulkan`: optional helper runtime for Vulkan resource/descriptors/barriers, without forcing downstream renderers to adopt it.
- `sample/path_tracer`: reference integration and visual validation pipeline.
- `tests`: shader compile tests, ABI/layout tests, profile-selection tests, queue tests, and NN reference tests.

### Profile Policy

Different vendors may expose different cooperative matrix sizes and subgroup behavior. The project should not allow arbitrary unchecked workgroup/matrix combinations.

The implementation should query capabilities and select from a whitelist of validated profiles. A slow scalar/reference backend may be used for correctness and CI, but product integrations should not silently fall back to it when the goal is no quality/performance loss.

Initial candidate profiles:

- `KHR_FP16_M16N16K16_SG32_WG128`
- `KHR_FP16_M16N16K16_SG16_WG128`
- `KHR_FP16_M8N16K16_SG32_WG128`
- `SCALAR_REFERENCE_WG64`

### Milestone Plan

1. Bootstrap repository and documentation.
2. Define public ABI, resource layouts, and capability/profile selection model.
3. Port record formats, input encoding, and queue helpers from VkNRC behavior.
4. Implement scalar/reference NN path for correctness tests.
5. Implement KHR cooperative matrix inference backend.
6. Implement gradient and Adam/EMA optimizer.
7. Build reference path tracing sample.
8. Add integration documentation for MCVR-style shader pack/runtime use.

### Current Limitations

- No NRC algorithm code has been implemented yet.
- No sample renderer exists yet.
- No tests exist yet.
- KHR cooperative matrix profile selection is only planned, not implemented.
- The exact NRC paper-to-code mapping still needs a dedicated design note before algorithm implementation begins.

### Validation

- `cmake` is not currently available on `PATH` in the default PowerShell environment.
- Used Visual Studio bundled CMake at `D:/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe`.
- Configure command succeeded:
  - `cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DNRC_BUILD_SAMPLE_PATH_TRACER=ON -DNRC_BUILD_TESTS=ON`
- CMake selected Windows SDK `10.0.26100.0`.
- CMake identified MSVC `19.51.36243.0`.
- Build command succeeded:
  - `cmake --build build --config Debug --target nrc_reference_path_tracer`
- Built artifact:
  - `G:/cpp/radiance/NRC/build/sample/path_tracer/Debug/nrc_reference_path_tracer.exe`

### Next Step

Create and commit the bootstrap repository state, then start Milestone 1 by defining `NrcProfile`, `NrcCapabilities`, and resource layout contracts.

## 2026-07-08 - Milestone 1 Profile and Layout Contract

### Implemented

- Added `nrc_core` as a real static library target instead of a pure interface target.
- Added public constants in `include/nrc/nrc_constants.hpp`.
- Added public profile/capability selection ABI in `include/nrc/nrc_profile.hpp`.
- Added resource layout ABI in `include/nrc/nrc_resource_layout.hpp`.
- Implemented profile selection in `src/nrc_profile.cpp`.
- Implemented resource-size calculation in `src/nrc_resource_layout.cpp`.
- Added `tests/profile_layout_test.cpp` and registered it with CTest.

### VkNRC Compatibility Decisions

- Network structure follows VkNRC's first target:
  - Encoded input width: 64 fp16 values.
  - Hidden layer count: 5.
  - Hidden layer width: 64.
  - Output width: 3.
  - Train batch size: 16384.
  - Train batch count: 4.
  - Weight count: 20672.
- Record/resource sizes follow VkNRC's C++ and shader layout assumptions:
  - `PackedNRCInput`: 16 bytes.
  - `NRCEvalRecord`: 20 bytes.
  - `NRCTrainRecord`: 40 bytes.
  - `OptimizerState`: 20 bytes.
  - `OptimizerEntry`: 16 bytes.
- Eval record capacity follows VkNRC:
  - `width * height + trainBatchSize * trainBatchCount`.

### Profile Selection Policy

- Production profiles currently require KHR cooperative matrix support, float16 shader support, 16-bit storage buffer access, subgroup size control, full compute subgroups, a matching cooperative matrix property, and the required subgroup size.
- Full training profiles also require shader buffer float32 atomic add.
- Inference-only selection can skip the atomic-add requirement by setting `ProfileRequirements::requireTraining = false`.
- Scalar reference fallback is not selected unless `ProfileRequirements::allowReferenceBackend = true`.

### Validation

- Configure command succeeded:
  - `cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DNRC_BUILD_SAMPLE_PATH_TRACER=ON -DNRC_BUILD_TESTS=ON`
- Build command succeeded:
  - `cmake --build build --config Debug --target nrc_profile_layout_test`
- Test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `1/1 Test #1: nrc_profile_layout_test Passed`.

### Current Limitations

- Capability data is not queried from Vulkan yet. Milestone 1 only defines the renderer-independent capability view and selection policy.
- KHR cooperative matrix property mapping from Vulkan enums to `nrc::CooperativeMatrixCapability` is still missing.
- No shader ABI files have been ported yet.
- No reference renderer logic exists yet beyond the placeholder executable.
- Profile whitelist is preliminary and will need verification against actual device-reported matrix properties and shader compile/run behavior.

### Next Step

Start Milestone 2 by adding Vulkan capability query helpers or, if keeping Vulkan out of the core first, by adding shader-side NRC record/encoding headers and ABI tests that mirror the C++ layout constants.

## 2026-07-08 - Milestone 2 Shader Record and Encoding ABI

### Implemented

- Added C++ record ABI in `include/nrc/nrc_records.hpp`.
- Updated resource layout constants to use the C++ record structs through `sizeof`.
- Added shader constants in `shaders/nrc/common/NrcConstants.glsl`.
- Added shader record/destination encoding in `shaders/nrc/common/NrcTypes.glsl`.
- Added shader input encoding in `shaders/nrc/common/NrcEncoding.glsl`.
- Added basic shader queue record construction helpers in `shaders/nrc/common/NrcQueue.glsl`.
- Added shader compile smoke test: `tests/shader_compile/nrc_common_smoke.comp`.
- Registered a `glslc`-based CTest test named `nrc_shader_common_compile`.

### VkNRC Compatibility Decisions

- Destination packing follows VkNRC:
  - screen destination: `(x | (y << 15)) << 1`.
  - train destination: `(batch | (left << 2) | (right << 16)) << 1 | 1`.
- Input encoding follows VkNRC's current shader behavior:
  - 3D position frequency encoding.
  - scattered direction one-blob encoding.
  - normal one-blob encoding.
  - roughness encoded as `1 - exp(-roughness)`.
  - diffuse/specular packed into the final encoded block.
- The shader-side common headers currently avoid binding declarations so downstream renderers can decide descriptor layout.

### Validation

- Full Debug build succeeded:
  - `cmake --build build --config Debug`
- Test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `2/2 tests passed`.
- The shader compile test uses:
  - `D:/VulkanSDK/1.4.335.0/Bin/glslc.exe`
  - `--target-env=vulkan1.3`

### Issues Found and Fixed

- GLSL reserves `input` as a keyword. The first compile test caught this in `NrcEncoding.glsl` and `NrcQueue.glsl`; parameters and local variables were renamed to avoid the reserved word.

### Current Limitations

- Shader queue helpers only construct records and validate indices; they do not yet provide atomic reservation helpers because the final descriptor/buffer ownership contract is still being designed.
- Scene unpacking is not implemented yet. The reusable shader layer only defines `NrcUnpackedInput`; each renderer or sample must still translate its own hit/material data into that form.
- No inference, gradient, optimizer, or KHR cooperative matrix backend shader has been implemented yet.
- No runtime Vulkan capability query has been implemented yet.

### Next Step

Start Milestone 3 by adding either:

- Vulkan capability query translation into `nrc::DeviceCapabilitiesView`, or
- the scalar/reference NN backend and CPU-side reference tests for forward inference.

## 2026-07-08 - Milestone 3 Vulkan Capability Query Contract

### Implemented

- Added optional `nrc_vulkan` static library target behind `NRC_BUILD_VULKAN_RUNTIME`.
- Added Vulkan SDK discovery fallback for the local SDK:
  - `D:/VulkanSDK/1.4.335.0/Include`
  - `D:/VulkanSDK/1.4.335.0/Lib/vulkan-1.lib`
- Added public Vulkan helper API:
  - `include/nrc/vulkan/nrc_vulkan_capabilities.hpp`
- Added implementation:
  - `runtime/vulkan/nrc_vulkan_capabilities.cpp`
- Added Vulkan capability translation test:
  - `tests/vulkan_capabilities_test.cpp`

### Capability Query Behavior

- Queries device extension availability for:
  - `VK_KHR_cooperative_matrix`
  - `VK_EXT_shader_atomic_float`
- Queries feature support for:
  - shader float16.
  - storage buffer 16-bit access.
  - shader buffer float32 atomic add.
  - subgroup size control.
  - full compute subgroups.
  - KHR cooperative matrix feature.
- Queries subgroup size properties and translates min/max subgroup size into a list of powers of two.
- Translates `VkCooperativeMatrixPropertiesKHR` into `nrc::CooperativeMatrixCapability`.
- Rejects cooperative matrix properties that are not compute-stage capable or not subgroup-scope.

### Integration Decision

The query function does not directly link against `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`. Instead, it accepts an optional `PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` and otherwise attempts to load it through `vkGetInstanceProcAddr`.

Reason: extension commands are not guaranteed to be exported through `vulkan-1.lib`. This also fits downstream projects that already use volk or another Vulkan loader.

### Validation

- Full Debug build succeeded:
  - `cmake --build build --config Debug`
- Test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `3/3 tests passed`.

### Issues Found and Fixed

- Initial build missed `<string_view>` and `<limits>` includes in the Vulkan implementation; fixed.
- Initial link used the extension command directly and failed with an unresolved external symbol for `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`; fixed by switching to function pointer loading/injection.

### Current Limitations

- `query_device_capabilities` has not yet been run against a real physical device in a sample executable.
- The fallback `vkGetInstanceProcAddr(VK_NULL_HANDLE, ...)` may not work with all loaders. Robust integrations should pass the function pointer explicitly from their Vulkan loader.
- No profile is compiled or dispatched yet; capability selection is still host-side only.
- No KHR cooperative matrix shader backend exists yet.

### Next Step

Start Milestone 4 with a scalar/reference NN backend and CPU tests, or begin KHR inference shader scaffolding once the reference path is available for correctness comparison.

## 2026-07-08 - Milestone 4 Reference NN and Scalar Shader Backend

### Implemented

- Added CPU reference NN API:
  - `include/nrc/nrc_reference_nn.hpp`
  - `src/nrc_reference_nn.cpp`
- Added GLSL scalar NN backend:
  - `shaders/nrc/backend/scalar/NrcNNScalar.glsl`
- Added CPU reference NN unit test:
  - `tests/reference_nn_test.cpp`
- Added scalar shader compile smoke test:
  - `tests/shader_compile/nrc_scalar_smoke.comp`
- Registered new CTest tests:
  - `nrc_reference_nn_test`
  - `nrc_shader_scalar_compile`

### VkNRC Compatibility Decisions

- The reference path uses the same topology as VkNRC:
  - 64 encoded inputs.
  - 5 hidden layers.
  - 64 hidden neurons per hidden layer.
  - ReLU after hidden layers.
  - 3 raw output values.
- Weight layout is row-major by layer:
  - hidden layer offset: `layer * 64 * 64`.
  - output layer offset: `5 * 64 * 64`.
  - row-major index: `offset + row * 64 + column`.
- CPU reference uses float weights for clarity and deterministic tests.
- GLSL scalar backend reads packed fp16 weights from `uvec4` storage, matching the future GPU weight buffer format.

### Validation

- Full Debug build succeeded:
  - `cmake --build build --config Debug`
- Test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `5/5 tests passed`.

### Current Limitations

- CPU reference and GLSL scalar backend are not yet cross-validated numerically against each other at runtime. Current shader coverage is compile-only.
- No fp32-to-fp16 weight packing utility exists yet.
- No KHR cooperative matrix backend exists yet.
- No inference pass consumes `NrcEvalRecord` queues yet.
- No gradient/backpropagation path has been implemented yet.

### Next Step

Start Milestone 5 by adding weight initialization/packing utilities and a deterministic packed-weight fixture, then use that fixture for scalar GPU and future KHR backend validation.

## 2026-07-08 - Milestone 5 Weight Initialization and Packing

### Implemented

- Added public weight utility API:
  - `include/nrc/nrc_weights.hpp`
- Added implementation:
  - `src/nrc_weights.cpp`
- Added test:
  - `tests/weights_test.cpp`
- Registered CTest test:
  - `nrc_weights_test`

### Behavior

- Added self-contained float32 to IEEE 754 binary16 conversion.
- Added binary16 to float32 conversion.
- Added deterministic Kaiming/He initialization using:
  - `std::mt19937`
  - normal distribution with standard deviation `sqrt(2 / 64)`.
- Added packed fp16 weight layout:
  - 8 fp16 values per `uvec4`.
  - Each `uint` stores two fp16 values.
  - Even lane uses low 16 bits.
  - Odd lane uses high 16 bits.

### VkNRC Compatibility Decisions

- Weight count remains `20672`.
- Packed weight block count is `20672 / 8 = 2584`.
- Packed byte count is `2584 * sizeof(uvec4) = 41344`, matching the resource layout test.
- The packed lane order matches `NrcScalarLoadWeight` in `NrcNNScalar.glsl`.

### Validation

- Full Debug build succeeded:
  - `cmake --build build --config Debug`
- Test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `6/6 tests passed`.

### Current Limitations

- Kaiming initialization currently uses the C++ standard library RNG. This is deterministic for a given implementation, but may not be bit-identical across all C++ standard library implementations.
- No GPU runtime test uploads packed weights yet.
- No scalar GPU dispatch test compares shader output to CPU reference output yet.
- No optimizer state initialization helper has been added yet.

### Next Step

Start Milestone 6 by adding a minimal compute dispatch harness for the scalar shader or by implementing the first KHR cooperative matrix inference shader scaffold against the same packed weight layout.

## 2026-07-08 - Milestone 6 KHR Cooperative Matrix Inference Scaffold

### Implemented

- Added KHR cooperative matrix inference backend scaffold:
  - `shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl`
- Added KHR shader compile smoke test:
  - `tests/shader_compile/nrc_khr_smoke.comp`
- Registered shader compile tests for:
  - `NRC_KHR_SUBGROUP_SIZE=32`
  - `NRC_KHR_SUBGROUP_SIZE=16`
- Added profile `implemented` flag so profile selection only returns backends that have an implemented shader path.
- Marked `KHR_FP16_M8N16K16_SG32_WG128` as planned but not currently implemented.
- Added profile test coverage to ensure an M8-only device is not selected until that backend exists.

### Backend Behavior

- The KHR backend currently implements forward inference for the VkNRC-compatible topology:
  - 64 encoded fp16 inputs.
  - 5 hidden 64-wide ReLU layers.
  - 3 output values.
- It reads packed fp16 `uvec4` weights using the Milestone 5 layout.
- It uses `coopmat<float16_t, gl_ScopeSubgroup, 16, 16, ...>` and supports compile-time subgroup sizes 16 and 32.
- It follows VkNRC's output-layer behavior by loading only the 3x64 output rows and zero-filling the remaining rows of the 16x64 cooperative matrix tile.

### Validation

- Full Debug build succeeded:
  - `cmake --build build --config Debug`
- Test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `8/8 tests passed`.
- KHR smoke tests compile with:
  - `--target-env=vulkan1.3`
  - `-DNRC_KHR_SUBGROUP_SIZE=32`
  - `-DNRC_KHR_SUBGROUP_SIZE=16`

### Current Limitations

- KHR backend is compile-validated only. It has not yet been dispatched on a GPU.
- No numerical comparison exists yet between CPU reference, scalar shader, and KHR shader outputs.
- Only 16x16x16 cooperative matrix profiles are implemented. The M8 profile remains planned and intentionally disabled in profile selection.
- No gradient/backpropagation KHR backend exists yet.
- No inference pass consumes `NrcEvalRecord` queues yet.

### Next Step

Start Milestone 7 by building a minimal Vulkan compute dispatch test harness. The first target should dispatch the scalar backend with deterministic packed weights and compare against `nrc::reference_forward`; after that, reuse the harness for KHR backend validation on devices that support the selected profile.

## 2026-07-08 - Milestone 7 Vulkan Scalar Dispatch Harness

### Implemented

- Added scalar dispatch shader:
  - `tests/shader_compile/nrc_scalar_dispatch.comp`
- Added Vulkan compute runtime validation test:
  - `tests/vulkan_scalar_dispatch_test.cpp`
- Registered generated SPIR-V build target:
  - `nrc_scalar_dispatch_shader`
- Registered CTest test:
  - `nrc_vulkan_scalar_dispatch_test`
- Added design note:
  - `docs/vulkan_dispatch_harness.md`

### Behavior

- The dispatch shader binds:
  - binding 0: packed fp16 weights, using `NrcNNScalar.glsl`'s existing layout.
  - binding 1: packed fp16 encoded inputs.
  - binding 2: output `vec4` values.
- The shader uses `local_size_x = 64` and one invocation per sample.
- The test dispatches 73 samples, intentionally not a multiple of the workgroup size, to validate the sample-count guard.
- CPU expected values are computed with `nrc::reference_forward` after both weights and inputs are rounded through the same fp16 packing path that the shader sees.
- The test creates a minimal Vulkan 1.1 compute pipeline, host-visible coherent storage buffers, descriptor set, push constant, command buffer, and host/GPU memory barriers.
- If no compatible Vulkan compute device or host-visible coherent storage buffer memory is available, the test prints `SKIP:` and exits successfully.

### Design Decision

This is deliberately a test harness, not the final reusable Vulkan runtime abstraction. The reusable runtime should be extracted only after the inference queue pass, KHR profile pipeline creation, and barrier/resource ownership contracts are clearer.

### Validation

- Configure command succeeded:
  - `cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DNRC_BUILD_SAMPLE_PATH_TRACER=ON -DNRC_BUILD_TESTS=ON -DNRC_BUILD_VULKAN_RUNTIME=ON`
- Full Debug build succeeded:
  - `cmake --build build --config Debug`
- Build generated:
  - `G:/cpp/radiance/NRC/build/tests/nrc_scalar_dispatch.spv`
  - `G:/cpp/radiance/NRC/build/tests/Debug/nrc_vulkan_scalar_dispatch_test.exe`
- Test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `9/9 tests passed`.
- The scalar Vulkan dispatch test passed on the current machine as a real Vulkan dispatch, not as a skip.

### Current Limitations

- The Vulkan harness validates only the scalar backend.
- KHR cooperative matrix backend is still compile-validated only.
- The harness uses host-visible coherent buffers for simplicity; it does not yet validate staging buffers or device-local production resources.
- No reusable descriptor/pipeline runtime API has been promoted from the test yet.
- No inference pass consumes `NrcEvalRecord` queues yet.
- No gradient/backpropagation or optimizer GPU pass has been implemented yet.

### Next Step

Use the dispatch harness pattern for KHR cooperative matrix numerical validation on devices that support an implemented profile. After that, implement the first inference pass that consumes `NrcEvalRecord` records and writes results through an integration-ready descriptor contract.

## 2026-07-08 - Milestone 8 KHR Dispatch Pipeline Gate

### Implemented

- Added KHR dispatch shader:
  - `tests/shader_compile/nrc_khr_dispatch.comp`
- Added Vulkan KHR dispatch validation test:
  - `tests/vulkan_khr_dispatch_test.cpp`
- Registered generated SPIR-V files:
  - `nrc_khr_dispatch_sg32.spv`
  - `nrc_khr_dispatch_sg16.spv`
- Registered build target:
  - `nrc_khr_dispatch_shaders`
- Registered CTest test:
  - `nrc_vulkan_khr_dispatch_test`
- Updated documentation:
  - `docs/khr_coopmat_backend.md`
  - `docs/vulkan_dispatch_harness.md`

### Behavior

- The dispatch shader binds:
  - binding 0: packed fp16 weights.
  - binding 1: packed fp16 encoded inputs.
  - binding 2: output `vec4` values.
- It uses `local_size_x = NRC_KHR_WORKGROUP_SIZE`, currently 128.
- It pads the final workgroup by feeding zero inputs to invocations whose global sample index is outside `sampleCount`.
- The C++ test queries real Vulkan device capabilities through `nrc::vulkan::query_device_capabilities`.
- It selects an implemented KHR cooperative matrix inference profile with:
  - `ProfileRequirements::requireTraining = false`
  - `ProfileRequirements::allowReferenceBackend = false`
- Device creation enables:
  - `VK_KHR_cooperative_matrix`
  - `VkPhysicalDeviceCooperativeMatrixFeaturesKHR::cooperativeMatrix`
  - shader float16
  - storage buffer 16-bit access
  - subgroup size control
  - full compute subgroups
- Pipeline creation uses:
  - `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`
  - `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT`
- If pipeline creation and dispatch succeed, the test compares 149 GPU samples against a CPU reference that rounds hidden activations and output values through fp16 to better match cooperative matrix behavior.

### Local Device Finding

The current machine has:

- Device: AMD Radeon RX 9070 XT.
- Driver: AMD proprietary driver 26.6.4 (LLPC).
- Vulkan API: 1.4.349.

`vulkaninfoSDK.exe` reports:

- `VK_KHR_cooperative_matrix`
- cooperative matrix feature enabled by the driver
- shader float16
- storage buffer 16-bit access
- subgroup size control
- full compute subgroups

The NRC capability/profile query selects:

- `KHR_FP16_M16N16K16_SG32_WG128`

However, `vkCreateComputePipelines` raises a Windows structured exception when compiling the VkNRC-style KHR cooperative matrix shader. The same failure occurs when using `G:/cpp/radiance/VkNRC/test/evaluate.comp` compiled with `SUBGROUP_SIZE=32`, so this is not specific to the new dispatch wrapper.

The test now guards `vkCreateComputePipelines` on MSVC/Windows and converts this driver-side crash into:

```text
SKIP: vkCreateComputePipelines raised a structured exception for the KHR cooperative matrix shader
```

### Validation

- `spirv-val` succeeded for the generated KHR SG32 dispatch SPIR-V:
  - `spirv-val --target-env vulkan1.3 build/tests/nrc_khr_dispatch_sg32.spv`
- VkNRC's original KHR experiment shader also compiled and validated:
  - `glslc --target-env=vulkan1.3 -DSUBGROUP_SIZE=32 G:/cpp/radiance/VkNRC/test/evaluate.comp`
  - `spirv-val --target-env vulkan1.3 build/tests/vknrc_evaluate_sg32.spv`
- Configure command succeeded:
  - `cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DNRC_BUILD_SAMPLE_PATH_TRACER=ON -DNRC_BUILD_TESTS=ON -DNRC_BUILD_VULKAN_RUNTIME=ON`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `10/10 tests passed`
- Important qualification:
  - `nrc_vulkan_khr_dispatch_test` passed on this machine by controlled skip, not by numerical GPU comparison, because the AMD driver raised a structured exception during KHR pipeline creation.

### Current Limitations

- KHR numerical validation has not completed on the current AMD driver.
- Capability/profile selection is not sufficient by itself to prove a KHR backend can be enabled in production. Pipeline creation must also be validated.
- The KHR backend remains a forward inference scaffold until it is run successfully and compared against CPU/scalar reference output on at least one conforming driver.
- The MSVC structured-exception guard is Windows-specific. Other platforms still need process-level isolation or another strategy if a driver crashes during pipeline creation.
- No reusable runtime API has been promoted for pipeline validation or backend fallback policy yet.
- No inference pass consumes `NrcEvalRecord` queues yet.
- No gradient/backpropagation or optimizer GPU pass has been implemented yet.

### Next Step

Add a reusable backend validation/runtime contract that distinguishes capability support, pipeline creation success, and numerical validation status. Then continue toward the first queue-consuming inference pass, using scalar dispatch as the correctness anchor while the KHR backend remains driver-gated.

## 2026-07-08 - Milestone 9 Scalar Queue Inference Pass Header

### Implemented

- Added first queue-consuming inference pass header:
  - `shaders/nrc/passes/NrcInferenceScalar.glsl`
- Added shader compile smoke test:
  - `tests/shader_compile/nrc_inference_scalar_smoke.comp`
- Registered CTest test:
  - `nrc_shader_inference_scalar_compile`
- Added design note:
  - `docs/inference_pass.md`
- Updated project structure documentation to describe shader-header passes.

### VkNRC Compatibility Decisions

- The pass mirrors VkNRC's inference responsibilities:
  - consume `NrcEvalRecord`;
  - unpack packed renderer input;
  - encode the 64-value NRC input;
  - run NN inference;
  - clamp prediction to non-negative values;
  - apply `bias + factor * prediction` to either screen output or train-record bias.
- Screen output keeps VkNRC's split:
  - `rgba32f` image stores `bias.rgb` and `factor.r`;
  - `rg32f` image stores `factor.g` and `factor.b`.
- Train output uses this repository's flat train-record resource layout:
  - `batch * NRC_TRAIN_BATCH_SIZE + index`
  - This differs from VkNRC's descriptor-array-of-batches model but preserves the data semantics and is easier for downstream engines that do not want descriptor arrays.

### Integration Boundary

The pass deliberately does not know scene or material layout. Integrations must provide:

```glsl
NrcUnpackedInput NrcUnpackPackedInput(PackedNrcInput packedInput);
```

This keeps the NRC library reusable while preserving VkNRC's packed input record and encoding behavior.

### Validation

- Configure command succeeded:
  - `cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DNRC_BUILD_SAMPLE_PATH_TRACER=ON -DNRC_BUILD_TESTS=ON -DNRC_BUILD_VULKAN_RUNTIME=ON`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `11/11 tests passed`
- The new inference pass is compile-validated through a smoke shader that provides a fake deterministic scene unpack function.

### Current Limitations

- The scalar inference pass has not yet been dispatched on Vulkan.
- No test yet verifies actual screen image or train-record mutation.
- No KHR inference pass exists yet; KHR remains blocked on driver pipeline creation on the current AMD machine.
- Scene/material unpacking is still only represented by a test fake. The reference path tracer must provide the real implementation.
- No indirect dispatch generation or eval count reset/update pass exists yet.
- No gradient/backpropagation or optimizer GPU pass has been implemented yet.

### Next Step

Add a Vulkan dispatch test for `NrcInferenceScalar.glsl` with a fake scene unpack and small screen/train buffers. That test should verify that screen destinations and train destinations are both updated exactly as expected before moving on to the reference path tracer integration.

## 2026-07-08 - Milestone 10 Scalar Inference Vulkan Dispatch Test

### Implemented

- Added Vulkan dispatch validation test for the scalar inference pass:
  - `tests/vulkan_inference_scalar_test.cpp`
- Added generated SPIR-V target:
  - `nrc_inference_scalar_dispatch_shader`
- Registered CTest test:
  - `nrc_vulkan_inference_scalar_test`
- Updated inference pass documentation:
  - `docs/inference_pass.md`

### Test Design

The test uses the existing smoke shader wrapper:

- `tests/shader_compile/nrc_inference_scalar_smoke.comp`

That shader provides a deterministic fake implementation of:

```glsl
NrcUnpackPackedInput(PackedNrcInput packedInput)
```

The test configures the MLP with:

- identity hidden layers;
- output weights reading the final two encoded constant `1.0` lanes;
- constant prediction:
  - `vec3(0.25, 0.5, 0.75)`.

This makes validation focus on inference-pass resource behavior instead of CPU-side duplication of the GLSL input encoder.

### Runtime Coverage

The test creates real Vulkan resources:

- packed fp16 weight storage buffer;
- `NrcEvalRecord` storage buffer;
- eval-count storage buffer;
- `rgba32f` storage image for `bias.rgb` and `factor.r`;
- `rg32f` storage image for `factor.g` and `factor.b`;
- flat `NrcTrainRecord` storage buffer sized for all NRC train batches;
- upload/readback staging buffers for the storage images.

It records barriers and copies for:

- host-to-transfer image upload;
- transfer-to-compute image layout transition;
- host-to-compute storage-buffer visibility;
- compute-to-transfer image readback;
- compute-to-host train-buffer visibility.

### Validated Behavior

The test verifies:

- screen destination `(1, 1)` is updated with `bias + factor * prediction`;
- screen destination `(2, 0)` is updated independently;
- screen alpha is written as `0.0`, matching `NrcInferenceApplyScreen`;
- train destination `batch 0, [2, 4]` updates only train-record bias values;
- train-record factors remain unchanged;
- an invalid eval destination does not update any resource;
- untouched screen pixels and nearby train records remain unchanged.

### Validation

- Configure command succeeded:
  - `cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DNRC_BUILD_SAMPLE_PATH_TRACER=ON -DNRC_BUILD_TESTS=ON -DNRC_BUILD_VULKAN_RUNTIME=ON`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Build generated:
  - `G:/cpp/radiance/NRC/build/tests/nrc_inference_scalar_dispatch.spv`
  - `G:/cpp/radiance/NRC/build/tests/Debug/nrc_vulkan_inference_scalar_test.exe`
- Test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `12/12 tests passed`
- The new scalar inference Vulkan test passed as a real dispatch on the current machine, not as a skip.

### Current Limitations

- The inference dispatch test uses a fake scene unpack function; the reference path tracer still needs a real scene/material implementation.
- The test uses host-visible buffers and staging for clarity, not a production runtime allocator.
- Only the scalar inference pass is runtime-validated.
- No KHR inference pass exists yet; KHR cooperative matrix pipeline creation is still driver-gated on the current AMD machine.
- No indirect dispatch generation or eval-count reset/update pass exists yet.
- No gradient/backpropagation or optimizer GPU pass has been implemented yet.

### Next Step

Add queue-management helpers/passes for eval count reset and eval-record reservation, then use those pieces in the reference path tracer sample. The scalar inference dispatch test should remain the correctness anchor while the sample grows toward real scene integration.

## 2026-07-08 - Milestone 11 Queue State and Reservation Ops

### Implemented

- Added host queue state ABI:
  - `nrc::NrcQueueState`
- Added queue state resource layout field:
  - `ResourceLayout::queueStateBytes`
- Updated train count layout semantics:
  - `ResourceLayout::trainBatchCountBytes` now covers all train batch counters.
- Added shader queue ops header:
  - `shaders/nrc/passes/NrcQueueOps.glsl`
- Added queue ops dispatch shader:
  - `tests/shader_compile/nrc_queue_ops_dispatch.comp`
- Added Vulkan queue ops runtime test:
  - `tests/vulkan_queue_ops_test.cpp`
- Registered CTest tests:
  - `nrc_shader_queue_ops_compile`
  - `nrc_vulkan_queue_ops_test`
- Added documentation:
  - `docs/queue_ops.md`
- Updated:
  - `docs/shader_abi.md`
  - `docs/profile_and_layout.md`

### ABI and Layout

`NrcQueueState` is:

```text
uint evalCount
uint trainCounts[4]
```

Size:

- `sizeof(NrcQueueState) == 20`

For the `1280x720` layout test:

- `evalCountBytes == 4`
- `trainBatchCountBytes == 16`
- `queueStateBytes == 20`

### VkNRC Compatibility and Deliberate Difference

VkNRC's path tracer directly uses:

```glsl
atomicAdd(uEvalCount, 1u, ...)
atomicAdd(uBatchTrainCounts[batch].count, bounce, ...)
```

That means counters can exceed buffer capacity unless the caller clamps later dispatches.

This reusable library uses compare-and-swap reservation helpers instead:

- `NrcReserveEvalRecord(capacity, out index)`
- `NrcPushEvalRecord(record, capacity, out index)`
- `NrcReserveTrainRecords(batch, requestedCount, out firstIndex, out reservedCount)`

These helpers clamp counters at their configured capacities. Overflow work is dropped by returning `false` or a truncated `reservedCount`. This is safer for library-style integration because a later inference pass can trust the clamped eval count.

### Runtime Validation

The Vulkan queue ops test:

- seeds queue counters with non-zero values;
- dispatches `NrcQueueResetMain`;
- dispatches 64 synthetic queue producers;
- lets all producers compete for 17 eval-record slots;
- lets all producers reserve 300 train records each in batch 0, which exceeds `NRC_TRAIN_BATCH_SIZE`;
- verifies:
  - eval count clamps to 17;
  - train batch 0 count clamps to 16384;
  - train batches 1-3 remain 0;
  - all reserved eval records contain the expected screen destination and packed input payload;
  - all reserved train records contain the expected bias/factor/input payload;
  - batch 1 remains untouched.

### Validation

- Configure command succeeded:
  - `cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DNRC_BUILD_SAMPLE_PATH_TRACER=ON -DNRC_BUILD_TESTS=ON -DNRC_BUILD_VULKAN_RUNTIME=ON`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Build generated:
  - `G:/cpp/radiance/NRC/build/tests/nrc_queue_ops_dispatch.spv`
  - `G:/cpp/radiance/NRC/build/tests/Debug/nrc_vulkan_queue_ops_test.exe`
- Test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `14/14 tests passed`
- The queue ops Vulkan test passed as a real dispatch on the current machine, not as a skip.

### Current Limitations

- Queue ops are validated with a synthetic producer shader, not yet the reference path tracer.
- No indirect dispatch argument generation exists yet.
- No frame/sample orchestration helper exists yet for queue reset, path tracing, inference, gradient, and optimizer order.
- Overflow work is intentionally dropped by clamped reservation. Integrations that need diagnostics must compare requested work against returned counts or final queue counts.
- No gradient/backpropagation or optimizer GPU pass has been implemented yet.

### Next Step

Start turning `sample/path_tracer` from a placeholder into a minimal Vulkan reference integration. The first useful sample milestone should use queue reset, synthetic/simple eval record production, scalar inference, and readback so the sample exercises the same integration seams a real renderer will use before full ray tracing and training are added.

## 2026-07-08 - Milestone 12 Minimal Reference Sample Integration

### Implemented

- Replaced the reference path tracer placeholder with a minimal Vulkan integration skeleton:
  - `sample/path_tracer/sample.cpp`
- Added sample producer shader:
  - `sample/path_tracer/shaders/sample_producer.comp`
- Added sample scalar inference wrapper:
  - `sample/path_tracer/shaders/sample_inference_scalar.comp`
- Added generated sample config template:
  - `sample/path_tracer/sample_config.hpp.in`
- Updated sample CMake to:
  - compile sample SPIR-V with `glslc`;
  - generate `sample_config.hpp` with SPIR-V paths;
  - link the real sample against `nrc_vulkan`;
  - fall back to the placeholder only when Vulkan/glslc is unavailable.
- Registered CTest smoke test:
  - `nrc_sample_reference_path_tracer_smoke`
- Added documentation:
  - `docs/sample_path_tracer.md`
- Updated:
  - `docs/project_structure.md`

### Runtime Flow

The sample now records a real Vulkan compute command buffer:

1. Upload a `4x4` bias/factor-R image and factor-GB image.
2. Dispatch `sample_producer.comp` in reset mode.
3. Dispatch `sample_producer.comp` in producer mode.
4. Dispatch `sample_inference_scalar.comp`.
5. Copy the resolved screen image back to the CPU.

The producer emits one screen eval record per pixel using `NrcQueueOps.glsl`.

The inference wrapper provides a deterministic fake `NrcUnpackPackedInput` and uses `NrcInferenceScalar.glsl`.

The host configures deterministic weights so the scalar MLP predicts:

```text
vec3(0.25, 0.5, 0.75)
```

### Validated Behavior

The sample verifies:

- `evalCount == 16`;
- all train batch counts remain zero;
- every `4x4` pixel has exactly one eval record;
- scalar inference resolves every screen pixel as `bias + factor * prediction`;
- output alpha is `0.0`, matching the inference pass contract.

### Validation

- Configure command succeeded:
  - `cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DNRC_BUILD_SAMPLE_PATH_TRACER=ON -DNRC_BUILD_TESTS=ON -DNRC_BUILD_VULKAN_RUNTIME=ON`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Manual sample run succeeded:
  - `build/sample/path_tracer/Debug/nrc_reference_path_tracer.exe`
- Manual sample output:
  - `NRC reference path tracer minimal integration passed: 16 eval records, scalar inference screen resolve.`
  - `This milestone is a synthetic integration skeleton; full ray tracing is pending.`
- Test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `15/15 tests passed`

### Current Limitations

- This is not yet a full path tracer.
- No acceleration structure, ray query, material system, BRDF sampling, scene loading, or visual output file exists yet.
- The scene unpack function is deterministic fake data.
- The sample produces screen eval records only; it does not produce train records yet.
- KHR cooperative matrix inference is not used in the sample because KHR pipeline creation is driver-gated on the current AMD machine.
- No gradient/backpropagation or optimizer pass is connected yet.

### Next Step

Add a small scene/hit representation to the sample and replace the synthetic producer with real `PackedNrcInput` generation. After that, add train-record production so the sample begins covering the path tracer-to-training side of VkNRC's workflow.

## 2026-07-08 - Milestone 13 Sample Train Record Integration

### Implemented

- Extended sample producer shader:
  - `sample/path_tracer/shaders/sample_producer.comp`
- Updated sample host validation:
  - `sample/path_tracer/sample.cpp`
- Updated sample documentation:
  - `docs/sample_path_tracer.md`

### Runtime Behavior

The sample still emits one screen eval record for each `4x4` pixel.

It now also:

- reserves two train records in batch 0 through `NrcReserveTrainRecords`;
- writes synthetic train records with:
  - bias `vec3(0.5, 1.0, 1.5)`;
  - factor `vec3(2.0, 3.0, 4.0)`;
  - a packed input payload;
- pushes one train eval record targeting batch 0 records `[0, 1]`.

The scalar inference pass then resolves:

- all screen eval records into the screen image;
- the train eval record into the referenced train-record bias fields.

### Validated Behavior

The sample now verifies:

- `evalCount == 17`;
- all 16 pixels have exactly one screen eval record;
- exactly one train eval record exists;
- the train eval record targets batch 0 records `[0, 1]`;
- train batch 0 count is 2;
- train batches 1-3 remain zero;
- both train records preserve their factor values;
- both train-record bias values are updated with `bias + factor * prediction`;
- all screen pixels still resolve correctly.

### Validation

- Debug build succeeded:
  - `cmake --build build --config Debug`
- Manual sample run succeeded:
  - `build/sample/path_tracer/Debug/nrc_reference_path_tracer.exe`
- Manual sample output:
  - `NRC reference path tracer minimal integration passed: 17 eval records, scalar inference screen/train resolve.`
  - `This milestone is a synthetic integration skeleton; full ray tracing is pending.`
- Test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `15/15 tests passed`

### Current Limitations

- Train records are still synthetic and not generated from real path-traced samples.
- No acceleration structure, ray query, material system, BRDF sampling, scene loading, or visual output file exists yet.
- The scene unpack function remains deterministic fake data.
- KHR cooperative matrix inference is not used in the sample because KHR pipeline creation is driver-gated on the current AMD machine.
- No gradient/backpropagation or optimizer pass is connected yet.

### Next Step

Add a small scene/hit representation to the sample so `PackedNrcInput` comes from a real geometry/material model instead of synthetic pixel data. After that, add a simple path tracing loop and use the generated train records as inputs for scalar gradient/optimizer once those passes exist.

## 2026-07-08 - Milestone 14 Scene-Backed Sample Input Generation

### Implemented

- Added shared sample scene shader header:
  - `sample/path_tracer/shaders/sample_scene.glsl`
- Updated sample producer shader:
  - `sample/path_tracer/shaders/sample_producer.comp`
- Updated sample inference wrapper:
  - `sample/path_tracer/shaders/sample_inference_scalar.comp`
- Updated sample host app:
  - `sample/path_tracer/sample.cpp`
- Updated sample CMake dependencies:
  - `sample/path_tracer/CMakeLists.txt`
- Updated documentation:
  - `docs/sample_path_tracer.md`
  - `docs/project_structure.md`

### Runtime Behavior

The sample now uploads a small scene buffer containing two triangles. Each vertex carries:

- position;
- normal plus roughness;
- diffuse color;
- specular color.

`sample_scene.glsl` provides the sample integration boundary:

- maps a `4x4` pixel to a triangle hit;
- packs `primitiveId`, instance id, barycentrics, and scattered direction into `PackedNrcInput`;
- unpacks the same payload through the scene buffer into `NrcUnpackedInput` for `NrcInferenceScalar.glsl`.

The reusable NRC inference pass remains scene-agnostic. The sample still provides:

```glsl
NrcUnpackedInput NrcUnpackPackedInput(PackedNrcInput packedInput);
```

This preserves the intended integration shape for engines such as MCVR-custom: renderer-specific hit/material data stays outside `shaders/nrc`, while NRC passes consume the stable packed record ABI.

### Descriptor Layout Changes

The sample now binds the scene buffer as:

- producer set binding 3;
- inference set binding 6.

The sample producer and inference pipelines have separate descriptor set layouts, so the shared scene shader keeps the binding configurable through:

```glsl
#define NRC_SAMPLE_SCENE_BINDING ...
```

### Validated Behavior

The sample still verifies all Milestone 13 behavior:

- `evalCount == 17`;
- one screen eval record per pixel;
- one train eval record targeting batch 0 records `[0, 1]`;
- train batch 0 count is 2;
- train batches 1-3 remain zero;
- screen and train resolves still produce `bias + factor * prediction`.

It now additionally verifies:

- every screen eval record contains the expected scene-backed `PackedNrcInput`;
- the train eval record contains the expected scene-backed payload;
- both train records preserve the expected packed scene input payload after scalar inference mutates their bias fields.

### Validation

- Configure command succeeded:
  - `cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DNRC_BUILD_SAMPLE_PATH_TRACER=ON -DNRC_BUILD_TESTS=ON -DNRC_BUILD_VULKAN_RUNTIME=ON`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Manual sample run succeeded:
  - `build/sample/path_tracer/Debug/nrc_reference_path_tracer.exe`
- Manual sample output:
  - `NRC reference path tracer minimal integration passed: 17 eval records, scalar inference screen/train resolve.`
  - `This milestone uses a scene-backed hit/input path; full ray tracing is pending.`
- Test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `15/15 tests passed`

### Current Limitations

- This is still not a full path tracer.
- There is no acceleration structure, ray query, direct lighting, BRDF sampling, scene loading, or visual output artifact yet.
- Train records still use synthetic radiance/factor values, although their packed input now comes from the sample scene hit path.
- KHR cooperative matrix inference is not used in the sample because KHR pipeline creation remains driver-gated on the current AMD machine.
- No gradient/backpropagation or optimizer pass is connected yet.

### Next Step

Add a simple path tracing or direct-lighting stage so train-record bias/factor values come from actual sampled scene transport rather than constants. After that, use those records as the validation target for scalar gradient and optimizer passes.

## 2026-07-08 - Milestone 15 Direct-Lighting Train Payloads

### Implemented

- Extended sample scene shader:
  - `sample/path_tracer/shaders/sample_scene.glsl`
- Updated sample producer shader:
  - `sample/path_tracer/shaders/sample_producer.comp`
- Updated sample host validation:
  - `sample/path_tracer/sample.cpp`
- Updated sample documentation:
  - `docs/sample_path_tracer.md`

### Runtime Behavior

Train records no longer use constant bias/factor values. The producer now:

1. creates the same scene-backed `PackedNrcInput` for the sample training hit;
2. unpacks it through the sample scene buffer;
3. evaluates a deterministic direct-lighting payload;
4. stores train records with:
   - `bias = diffuse * lightColor * max(dot(normal, lightDir), 0)`;
   - `factor = max(diffuse + specular * (1 - roughness), 0.05)`;
5. emits the train eval record targeting the reserved batch 0 train records.

The scalar inference pass still resolves train records as:

```text
bias = directLightingBias + directLightingFactor * prediction
```

### Integration Boundary

The direct-lighting function is intentionally sample-local. The reusable NRC pass still only consumes:

- `PackedNrcInput`;
- renderer-provided `NrcUnpackPackedInput`;
- standard eval/train queues.

This keeps the library boundary suitable for MCVR-style integration while making the reference sample less synthetic.

### Validated Behavior

The host now mirrors the shader-side direct-lighting math and validates:

- screen eval payloads still match scene-backed packed inputs;
- train eval payload still matches the sample training hit;
- train records preserve the expected packed input payload;
- train-record factors match the CPU direct-lighting reference;
- train-record bias fields equal `directLightingBias + directLightingFactor * vec3(0.25, 0.5, 0.75)` after scalar inference.

### Validation

- Debug build succeeded:
  - `cmake --build build --config Debug`
- Manual sample run succeeded:
  - `build/sample/path_tracer/Debug/nrc_reference_path_tracer.exe`
- Manual sample output:
  - `NRC reference path tracer minimal integration passed: 17 eval records, scalar inference screen/train resolve.`
  - `This milestone uses scene-backed hits and direct-lighting train payloads; full ray tracing is pending.`
- Test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `15/15 tests passed`

### Current Limitations

- The lighting model is deterministic direct lighting, not ray-query or multi-bounce path tracing.
- No acceleration structure, ray query, BRDF sampling, scene loading, or visual output artifact exists yet.
- KHR cooperative matrix inference is not used in the sample because KHR pipeline creation remains driver-gated on the current AMD machine.
- No gradient/backpropagation or optimizer pass is connected yet.

### Next Step

Add a ray-query or simple path tracing stage so the sample can generate train records from actual traced paths rather than direct-lighting helper math.

## 2026-07-08 - Milestone 16 Primary-Ray Triangle Tracing Sample

### Implemented

- Extended sample scene shader:
  - `sample/path_tracer/shaders/sample_scene.glsl`
- Updated sample producer shader:
  - `sample/path_tracer/shaders/sample_producer.comp`
- Updated sample host validation:
  - `sample/path_tracer/sample.cpp`
- Updated documentation:
  - `docs/sample_path_tracer.md`
  - `docs/project_structure.md`

### Runtime Behavior

The sample producer no longer maps pixels analytically to triangle ids and barycentrics. It now:

1. builds a primary camera ray for each pixel;
2. intersects that ray against the two sample triangles with a local Moller-Trumbore triangle test;
3. packs the traced hit's primitive id and barycentrics into `PackedNrcInput`;
4. emits screen eval records only for traced hits;
5. uses the traced hit for the sample train records and direct-lighting payload.

The two triangles cover the full `4x4` sample image, so the expected eval count remains:

```text
16 screen eval records + 1 train eval record = 17
```

### Validation Behavior

The host-side verifier now mirrors the same primary-ray triangle intersection. It no longer relies on the older analytic pixel-to-triangle rule.

The sample validates:

- every screen eval record maps to a CPU-traced primary hit;
- every packed primitive id, instance id, barycentric payload, and scattered direction payload matches the CPU reference;
- the train eval record and train records use the traced primary hit for pixel `(0, 0)`;
- scalar inference still resolves screen and train outputs correctly.

### Design Note

This milestone deliberately avoids Vulkan acceleration structures and ray query. The goal is to tighten the NRC integration shape first:

- traced hit -> packed NRC input;
- scene/material unpack -> encoded NRC input;
- train record payload -> scalar inference resolve.

Vulkan ray query can replace the sample-local triangle loop later without changing the reusable NRC queue or inference ABI.

### Validation

- Debug build succeeded:
  - `cmake --build build --config Debug`
- Manual sample run succeeded:
  - `build/sample/path_tracer/Debug/nrc_reference_path_tracer.exe`
- Manual sample output:
  - `NRC reference path tracer minimal integration passed: 17 eval records, scalar inference screen/train resolve.`
  - `This milestone uses primary-ray traced hits and direct-lighting train payloads; full ray tracing is pending.`
- Test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `15/15 tests passed`

### Current Limitations

- The sample uses a local triangle loop, not Vulkan ray query or hardware acceleration structures.
- There are no secondary rays, shadow rays, BRDF sampling, scene loading, or visual output artifacts.
- Train records still use deterministic direct lighting rather than multi-bounce path-traced radiance.
- KHR cooperative matrix inference is not used in the sample because KHR pipeline creation remains driver-gated on the current AMD machine.
- No gradient/backpropagation or optimizer pass is connected yet.

### Next Step

Add either a tiny visual output path or start the scalar gradient/backpropagation pass. The visual path would make integration regressions easier to inspect; the gradient pass would move the core NRC algorithm closer to VkNRC coverage.

## 2026-07-08 - Milestone 17 Training Prepare and Optimizer Foundation

### Implemented

- Added host optimizer API:
  - `include/nrc/nrc_optimizer.hpp`
  - `src/nrc_optimizer.cpp`
- Added optimizer constants to:
  - `include/nrc/nrc_constants.hpp`
  - `shaders/nrc/common/NrcConstants.glsl`
- Added shader ABI structs:
  - `NrcOptimizerState`
  - `NrcOptimizerEntry`
- Added shader-header passes:
  - `shaders/nrc/passes/NrcTrainPrepare.glsl`
  - `shaders/nrc/passes/NrcOptimize.glsl`
- Added tests:
  - `tests/optimizer_test.cpp`
  - `tests/shader_compile/nrc_train_prepare_dispatch.comp`
  - `tests/shader_compile/nrc_optimize_dispatch.comp`
- Updated documentation:
  - `docs/training_optimizer.md`
  - `docs/profile_and_layout.md`
  - `docs/project_structure.md`

### VkNRC Behavior Preserved

The new optimizer foundation follows VkNRC's `nrc_train_prepare.comp` and `nrc_optimize.comp` behavior:

- clamp train count to `NRC_TRAIN_BATCH_SIZE`;
- advance optimizer state only for non-empty batches;
- update `step`, `beta1T`, `beta2T`, `alphaTMinus1`, and `alphaT`;
- divide gradients by `trainCount * lossScale`;
- treat NaN/Inf gradients as zero;
- apply Adam with:
  - `beta1 = 0.9`;
  - `beta2 = 0.999`;
  - learning rate `0.002`;
  - epsilon `1e-8`;
- update EMA with alpha `0.99`;
- write packed fp16 weights and optionally packed fp16 use-weights.

### Deliberate ABI Difference

VkNRC uses one train-count buffer per batch. This repository currently uses a compact queue-state buffer:

```text
uint evalCount
uint trainCounts[4]
```

The prepare and optimizer shader headers therefore select the active batch through push constants and read `trainCounts[batch]`. This keeps the current queue ABI suitable for MCVR-style integration while retaining VkNRC's optimizer math.

### Validation

- Configure command succeeded:
  - `cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DNRC_BUILD_SAMPLE_PATH_TRACER=ON -DNRC_BUILD_TESTS=ON -DNRC_BUILD_VULKAN_RUNTIME=ON`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `19/19 tests passed`

New validation coverage:

- `nrc_optimizer_test` checks CPU optimizer initialization, batch clamp, zero-batch no-op behavior, state advancement, Adam first-step math, non-finite gradient handling, and first-step EMA use-weight behavior.
- `nrc_shader_train_prepare_compile` compiles the train-prepare shader header through a dispatch wrapper.
- `nrc_shader_optimize_compile` compiles the optimizer shader header without `useWeights`.
- `nrc_shader_optimize_use_weights_compile` compiles the optimizer shader header with `NRC_OPTIMIZE_WRITE_USE_WEIGHTS=1`.

### Current Limitations

- The optimizer pass consumes externally supplied gradients. No scalar GPU gradient/backpropagation pass has been implemented yet.
- Optimizer GPU dispatch is compile-validated at this milestone, not numerically validated through a full Vulkan training dispatch.
- KHR/NV cooperative-matrix gradient paths are not implemented yet.
- The sample still does not run a training update; it only produces train records and runs scalar inference resolve.

### Next Step

Add a scalar gradient/backpropagation reference path that consumes `NrcTrainRecord` batches and produces the gradient buffer expected by the optimizer. Once that exists, add a Vulkan training dispatch test that runs prepare -> gradient -> optimize and compares updated weights against the CPU reference.

## 2026-07-08 - Milestone 18 CPU Reference Backpropagation

### Implemented

- Extended CPU reference NN API:
  - `include/nrc/nrc_reference_nn.hpp`
  - `src/nrc_reference_nn.cpp`
- Added reference gradient test:
  - `tests/reference_gradient_test.cpp`
- Updated CTest registration:
  - `tests/CMakeLists.txt`
- Updated documentation:
  - `docs/reference_nn.md`

### Behavior

`nrc::reference_backward` now:

- runs the same 64 -> 5x64 ReLU -> 3 forward topology as `reference_forward`;
- stores hidden-layer pre-activations for ReLU masks;
- computes output gradients for:
  - plain L2 loss;
  - VkNRC-style relative L2 luminance loss;
- accumulates gradients into the same row-major weight layout used by all weight utilities;
- optionally clears or accumulates into an existing gradient buffer.

The relative luminance loss gradient follows VkNRC's shader expression:

```text
2 * lossScale * (prediction - target)
/ (luminance(max(prediction, 0))^2 + 0.01)
```

As in VkNRC's backprop shader, the luminance denominator is used as the current prediction scale and is not itself differentiated.

### Validation

- Configure command succeeded:
  - `cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DNRC_BUILD_SAMPLE_PATH_TRACER=ON -DNRC_BUILD_TESTS=ON -DNRC_BUILD_VULKAN_RUNTIME=ON`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `20/20 tests passed`

New validation coverage:

- `nrc_reference_gradient_test` checks selected hidden and output weights against central finite differences.
- The finite-difference loss holds the relative-luminance denominator fixed, matching VkNRC's shader backprop behavior.
- The test checks accumulation mode and invalid weight-count validation.

### Current Limitations

- This is CPU reference backpropagation only. No GPU scalar gradient shader has been added yet.
- The reference function consumes already encoded `ReferenceInput` values. A later batch helper should bridge `NrcTrainRecord` plus renderer unpacking into encoded input batches for CPU/GPU comparison.
- No full training loop test exists yet.
- KHR/NV cooperative-matrix gradient paths are still missing.

### Next Step

Add a scalar GLSL gradient pass that consumes `NrcTrainRecord` batches, uses renderer-provided unpacking, accumulates fp32 gradients, and can be validated against `nrc::reference_backward` through a Vulkan dispatch test.

## 2026-07-08 - Milestone 19 Scalar Gradient Shader Header

### Implemented

- Added scalar gradient shader-header pass:
  - `shaders/nrc/passes/NrcGradientScalar.glsl`
- Added shader compile wrapper:
  - `tests/shader_compile/nrc_gradient_scalar_dispatch.comp`
- Registered CTest compile coverage:
  - `nrc_shader_gradient_scalar_compile`
- Added documentation:
  - `docs/gradient_pass.md`
- Updated:
  - `docs/shader_abi.md`

### Behavior

`NrcGradientScalar.glsl`:

- consumes flat `NrcTrainRecord` storage;
- reads the active train count from `NrcQueueState.trainCounts[batch]`;
- uses renderer-provided `NrcUnpackPackedInput`;
- encodes input through `NrcEncodeInput`;
- runs scalar forward propagation while storing activations and ReLU pre-activations;
- computes VkNRC-style relative L2 luminance output derivatives;
- backpropagates through the output and five hidden ReLU layers;
- atomically accumulates fp32 gradients in row-major weight layout.

The pass uses the same packed fp16 weight buffer layout as scalar inference.

### Validation

- Configure command succeeded:
  - `cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DNRC_BUILD_SAMPLE_PATH_TRACER=ON -DNRC_BUILD_TESTS=ON -DNRC_BUILD_VULKAN_RUNTIME=ON`
- Test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `21/21 tests passed`

New validation coverage:

- `nrc_shader_gradient_scalar_compile` compiles the scalar gradient pass through a wrapper that provides a fake unpack function.
- Existing `nrc_reference_gradient_test` remains the CPU numerical oracle for future GPU gradient validation.

### Current Limitations

- The scalar gradient shader is compile-validated only. It has not yet been dispatched and compared against the CPU reference.
- Runtime dispatch requires `VK_EXT_shader_atomic_float` with fp32 storage-buffer atomic add support.
- The pass is intentionally scalar and slow. It is for correctness/debugging, not production performance.
- KHR cooperative-matrix backpropagation is still missing.
- The sample still does not run prepare -> gradient -> optimize.

### Next Step

Add a Vulkan scalar gradient dispatch test with feature gating for `shaderBufferFloat32AtomicAdd`, compare selected gradients against `nrc::reference_backward`, then wire prepare -> gradient -> optimize into a full training dispatch test.

## 2026-07-08 - Milestone 20 Scalar Gradient Runtime Validation

### Implemented

- Added host-side input encoding mirror:
  - `include/nrc/nrc_encoding.hpp`
  - `src/nrc_encoding.cpp`
- Added encoding test:
  - `tests/encoding_test.cpp`
- Added Vulkan scalar gradient runtime test:
  - `tests/vulkan_gradient_scalar_test.cpp`
- Updated CMake registration:
  - `nrc_encoding_test`
  - `nrc_gradient_scalar_dispatch_shader`
  - `nrc_vulkan_gradient_scalar_test`
- Added `kPi` to `include/nrc/nrc_constants.hpp` for CPU spherical encoding.
- Updated documentation:
  - `docs/gradient_pass.md`
  - `docs/training_optimizer.md`
  - `docs/reference_nn.md`
  - `docs/shader_abi.md`
  - `docs/project_structure.md`

### Behavior

`nrc::encode_input` mirrors shader-side `NrcEncodeInput` for CPU reference and Vulkan dispatch tests. It accepts an `UnpackedNrcInput`, applies the same position frequency encoding, one-blob direction/normal/roughness encoding, diffuse/specular lane packing, and rounds each lane through the existing fp32<->fp16 conversion helpers so the CPU reference consumes the same effective inputs as the scalar GPU path.

`nrc_vulkan_gradient_scalar_test` now:

- builds a deterministic train batch with three records;
- packs fp32 weights into the same fp16 `uvec4` layout used by shaders;
- unpacks those packed weights back to fp32 for the CPU oracle;
- mirrors the test shader's fake `NrcUnpackPackedInput` on the host;
- encodes inputs through `nrc::encode_input`;
- accumulates expected gradients with `nrc::reference_backward`;
- dispatches `NrcGradientScalar.glsl` through Vulkan;
- compares the full fp32 gradient buffer against the CPU result.

The test requires `VK_EXT_shader_atomic_float` and `shaderBufferFloat32AtomicAdd`. If no Vulkan compute device exposes that feature, it reports a controlled skip instead of failing the suite.

### Validation

- Configure command succeeded:
  - `cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DNRC_BUILD_SAMPLE_PATH_TRACER=ON -DNRC_BUILD_TESTS=ON -DNRC_BUILD_VULKAN_RUNTIME=ON`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `23/23 tests passed`

New validation coverage:

- `nrc_encoding_test` checks selected spherical encoding and 64-lane encoded input behavior.
- `nrc_shader_gradient_scalar_compile` continues to compile-check the shader-header integration boundary.
- `nrc_vulkan_gradient_scalar_test` numerically validates the dispatched scalar gradient pass against the CPU backpropagation reference.

### Current Limitations

- This is still the scalar correctness path. It is useful for validation and debugging, not the final high-performance training backend.
- The gradient pass depends on fp32 storage-buffer atomic add support.
- Full prepare -> gradient -> optimize Vulkan training loop validation is still pending.
- The sample path tracer still produces train records and runs scalar inference resolve, but does not yet update NRC weights.
- KHR cooperative-matrix gradient/backpropagation is still missing.

### Next Step

Add a full Vulkan training-step test that runs prepare -> scalar gradient -> optimize from `NrcTrainRecord` batches and compares updated fp32/fp16 weights against the CPU optimizer reference. After that, wire the same loop into the sample path tracer as a reference integration path.

## 2026-07-08 - Milestone 21 Full Scalar Vulkan Training Step

### Implemented

- Added Vulkan full training-step test:
  - `tests/vulkan_training_step_test.cpp`
- Added CMake shader build/test registration for:
  - `nrc_training_step_prepare.spv`
  - existing `nrc_gradient_scalar_dispatch.spv`
  - `nrc_training_step_optimize.spv`
  - `nrc_training_step_shaders`
  - `nrc_vulkan_training_step_test`
- Updated documentation:
  - `docs/training_optimizer.md`
  - `docs/gradient_pass.md`
  - `docs/sample_path_tracer.md`
  - `docs/project_structure.md`

### Behavior

`nrc_vulkan_training_step_test` validates the reusable scalar training sequence as an actual multi-pass Vulkan flow:

```text
NrcTrainPrepare -> NrcGradientScalar -> NrcOptimize
```

The test uses separate descriptor sets for each pass while keeping the buffers shared in the same way an integration would:

- prepare set:
  - queue state;
  - dispatch-indirect command;
  - optimizer state;
- gradient set:
  - train records;
  - queue state;
  - packed fp16 weights;
  - fp32 gradients;
- optimize set:
  - packed fp16 weights output;
  - packed fp16 use-weights output;
  - fp32 gradients;
  - fp32 optimizer entries;
  - queue state;
  - optimizer state.

The command buffer records:

1. host-to-compute barrier for uploaded buffers;
2. `NrcTrainPrepareMain`;
3. compute-to-indirect/compute barrier;
4. `vkCmdDispatchIndirect` for `NrcGradientScalarMain`;
5. compute-to-compute barrier;
6. `NrcOptimizeMain` with `NRC_OPTIMIZE_WRITE_USE_WEIGHTS=1`;
7. compute-to-host barrier for readback.

### CPU Oracle

The CPU reference path mirrors the shader sequence:

1. pack initial weights to fp16 and unpack them back to fp32 so the CPU oracle sees the same weights as the GPU;
2. initialize optimizer entries from the GPU-visible weights;
3. call `prepare_training_batch`;
4. encode train records through `nrc::encode_input`;
5. accumulate gradients through `nrc::reference_backward`;
6. update entries and weights through `nrc::optimize_weights`;
7. pack expected weights and use-weights to fp16 for comparison against shader output.

### Validation

- Configure command succeeded:
  - `cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DNRC_BUILD_SAMPLE_PATH_TRACER=ON -DNRC_BUILD_TESTS=ON -DNRC_BUILD_VULKAN_RUNTIME=ON`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Targeted test command succeeded:
  - `ctest --test-dir build -C Debug -R nrc_vulkan_training_step_test --output-on-failure`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `24/24 tests passed`

New validation coverage:

- prepare clamps and writes the train count used by later passes;
- prepare advances optimizer state;
- prepare writes the dispatch-indirect command consumed by gradient;
- scalar gradient produces the expected fp32 gradient buffer;
- optimize consumes those gradients and writes updated fp32 optimizer entries;
- optimize writes packed fp16 weights;
- optimize writes packed fp16 use-weights.

### Current Limitations

- The full training-step validation is scalar and correctness-oriented. It is not the final high-performance KHR cooperative-matrix training backend.
- The training-step test requires `VK_EXT_shader_atomic_float` / `shaderBufferFloat32AtomicAdd` because the scalar gradient pass accumulates fp32 gradients with storage-buffer atomics.
- The sample path tracer still does not run the training update loop.
- KHR cooperative-matrix gradient/backpropagation remains missing.

### Next Step

Wire the scalar prepare -> gradient -> optimize sequence into the sample path tracer so the reference integration performs an actual weight update after generating train records. After that, continue toward a KHR cooperative-matrix training backend and a richer path-tracing sample.

## 2026-07-09 - Milestone 22 Sample Scalar Training Integration

### Implemented

- Added sample training shader wrappers:
  - `sample/path_tracer/shaders/sample_train_prepare.comp`
  - `sample/path_tracer/shaders/sample_gradient_scalar.comp`
  - `sample/path_tracer/shaders/sample_optimize.comp`
- Updated sample CMake shader build:
  - `nrc_sample_train_prepare.spv`
  - `nrc_sample_gradient_scalar.spv`
  - `nrc_sample_optimize.spv`
- Updated sample config generation:
  - `NRC_SAMPLE_TRAIN_PREPARE_SPV`
  - `NRC_SAMPLE_GRADIENT_SPV`
  - `NRC_SAMPLE_OPTIMIZE_SPV`
- Extended `sample/path_tracer/sample.cpp` to run:
  - queue reset;
  - scene-backed eval/train production;
  - scalar inference resolve;
  - train prepare;
  - scalar gradient through indirect dispatch;
  - optimizer;
  - readback validation.
- Updated documentation:
  - `docs/sample_path_tracer.md`
  - `docs/training_optimizer.md`
  - `docs/gradient_pass.md`
  - `docs/project_structure.md`

### Behavior

The sample now performs one real scalar NRC training update after generating train records and resolving their target radiance through scalar inference.

Important integration changes:

- the sample train-record buffer now uses the full flat layout expected by reusable training passes:
  - `NRC_TRAIN_BATCH_COUNT * NRC_TRAIN_BATCH_SIZE`;
- the sample creates fp32 optimizer state and entries initialized from fp16-rounded GPU-visible weights;
- the sample creates fp32 gradient storage and an indirect dispatch command buffer;
- the sample enables `VK_EXT_shader_atomic_float` and `shaderBufferFloat32AtomicAdd` for scalar gradient accumulation;
- `sample_gradient_scalar.comp` uses `sample_scene.glsl` for `NrcUnpackPackedInput`, so gradient validation exercises the same scene/material unpacking path as inference.

The command buffer sequence is now:

```text
upload screen images
producer reset
producer emit eval/train records
scalar inference resolve
train prepare
scalar gradient via vkCmdDispatchIndirect
optimizer
screen image readback
```

### CPU Oracle

The host verifier mirrors the sample scene path and training math:

1. unpack initial packed fp16 weights back to fp32;
2. initialize optimizer entries from the GPU-visible weights;
3. verify screen and train resolve as before;
4. encode the final train records with the CPU `nrc::encode_input` mirror and sample scene unpacking;
5. accumulate gradients with `nrc::reference_backward`;
6. update weights with `nrc::optimize_weights`;
7. compare actual gradients, optimizer entries, packed weights, and packed use-weights against the CPU result;
8. verify at least one packed weight changes.

### Validation

- Configure command succeeded:
  - `cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DNRC_BUILD_SAMPLE_PATH_TRACER=ON -DNRC_BUILD_TESTS=ON -DNRC_BUILD_VULKAN_RUNTIME=ON`
- Targeted sample build succeeded:
  - `cmake --build build --config Debug --target nrc_reference_path_tracer`
- Manual sample run succeeded:
  - `build/sample/path_tracer/Debug/nrc_reference_path_tracer.exe`
- Manual sample output:
  - `NRC reference path tracer minimal integration passed: 17 eval records, scalar inference resolve, and one scalar training update.`
  - `This milestone uses primary-ray traced hits and direct-lighting train payloads; full ray tracing is pending.`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `24/24 tests passed`

### Current Limitations

- The sample still uses a local primary-ray triangle loop, not Vulkan ray query or acceleration structures.
- Training records still use deterministic direct lighting rather than multi-bounce path-traced radiance.
- The sample runs one validated update, not an iterative per-frame training loop.
- Sample training uses the scalar correctness path and therefore requires fp32 storage-buffer atomic add support.
- KHR cooperative-matrix gradient/backpropagation remains missing.

### Next Step

Turn the single sample update into a small iterative training loop with repeated frame-like steps and then add a visual output artifact. In parallel, continue toward the KHR cooperative-matrix training backend so the reusable implementation can move beyond the scalar correctness path.

## 2026-07-09 - Milestone 23 Sample Visual Output Artifact

### Implemented

- Added deterministic sample output path to generated config:
  - `NRC_SAMPLE_OUTPUT_PPM`
  - `build/sample/path_tracer/nrc_reference_path_tracer.ppm`
- Extended `sample/path_tracer/sample.cpp` to:
  - tone-map the resolved `4x4` screen readback;
  - write a binary PPM image;
  - re-read and byte-compare the written file against the readback-derived bytes;
  - validate the image has visible non-uniform pixel data.
- Updated documentation:
  - `docs/sample_path_tracer.md`
  - `docs/project_structure.md`

### Behavior

The reference sample now leaves behind an inspectable visual artifact after its integration smoke run. The image is intentionally tiny because the sample is still a deterministic integration harness:

```text
P6
4 4
255
```

The PPM payload is generated from the scalar-inference-resolved screen image after a simple Reinhard tone map and gamma transform. The artifact is written under the CMake binary directory so CTest's working directory does not affect where the file lands.

### Validation

- Configure command succeeded:
  - `cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DNRC_BUILD_SAMPLE_PATH_TRACER=ON -DNRC_BUILD_TESTS=ON -DNRC_BUILD_VULKAN_RUNTIME=ON`
- Targeted sample build succeeded:
  - `cmake --build build --config Debug --target nrc_reference_path_tracer`
- Manual sample run succeeded:
  - `build/sample/path_tracer/Debug/nrc_reference_path_tracer.exe`
- Manual sample output:
  - `NRC reference path tracer minimal integration passed: 17 eval records, scalar inference resolve, and one scalar training update.`
  - `Wrote sample image: G:/cpp/radiance/NRC/build/sample/path_tracer/nrc_reference_path_tracer.ppm`
  - `This milestone uses primary-ray traced hits and direct-lighting train payloads; full ray tracing is pending.`
- External artifact check:
  - `build/sample/path_tracer/nrc_reference_path_tracer.ppm`
  - size: `59` bytes;
  - header bytes decode to `P6`, `4 4`, `255`.
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `24/24 tests passed`

### Current Limitations

- The visual output is a `4x4` smoke artifact, not a useful rendered image for quality inspection yet.
- The sample still uses a local primary-ray triangle loop and deterministic direct lighting.
- The sample still runs only one scalar training update.
- KHR cooperative-matrix training is still missing.

### Next Step

Grow the sample from a single-frame smoke artifact into a small iterative reference run with a larger output image. The next useful sample milestone should either repeat train/infer/update over several frame-like iterations or replace the local triangle loop with a more realistic ray-query/path-tracing stage.

## 2026-07-09 - Milestone 24 KHR M8 Cooperative Matrix Inference Profile

### Implemented

- Generalized the KHR cooperative matrix inference backend:
  - `shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl`
- Marked the M8 KHR inference profile implemented:
  - `KHR_FP16_M8N16K16_SG32_WG128`
- Updated profile selection test:
  - M8-only device capabilities now select `KhrFp16M8N16K16Sg32Wg128`.
- Added shader compile coverage:
  - `nrc_shader_khr_m8_sg32_compile`
- Added M8 dispatch SPIR-V build path:
  - `nrc_khr_dispatch_m8_sg32.spv`
- Updated Vulkan KHR dispatch test:
  - accepts SG32, SG16, and M8/SG32 SPIR-V variants;
  - chooses the dispatch shader matching the selected implemented profile.
- Updated documentation:
  - `docs/khr_coopmat_backend.md`
  - `docs/profile_and_layout.md`
  - `docs/vulkan_dispatch_harness.md`

### Backend Behavior

The previous KHR backend implicitly relied on `M == K == 16`: the hidden-layer accumulator could be assigned directly back to the activation cooperative matrix because both matrices had the same shape.

The backend now treats matrix M and K separately:

- activation tiles remain `K x N`;
- hidden/output weight tiles are `M x K`;
- accumulators are `M x N`;
- hidden-layer outputs are ReLU-applied, stored back into the shared activation buffer, and reloaded as K-row activation tiles for the next layer.

This supports:

```text
M16N16K16
M8N16K16
```

The workgroup policy remains deliberately constrained:

```text
workgroup size: 128
subgroup size: 16 or 32
N: 16
K: 16
M: 8 or 16
```

### Validation

- Manual shader compile checks succeeded:
  - `NRC_KHR_SUBGROUP_SIZE=32`
  - `NRC_KHR_SUBGROUP_SIZE=16`
  - `NRC_KHR_SUBGROUP_SIZE=32 NRC_KHR_COOPMAT_M=8`
- Configure command succeeded:
  - `cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DNRC_BUILD_SAMPLE_PATH_TRACER=ON -DNRC_BUILD_TESTS=ON -DNRC_BUILD_VULKAN_RUNTIME=ON`
- Targeted build succeeded:
  - `cmake --build build --config Debug --target nrc_profile_layout_test nrc_vulkan_khr_dispatch_test`
- Targeted KHR/profile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_profile_layout_test|nrc_shader_khr|nrc_vulkan_khr_dispatch_test" --output-on-failure`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `25/25 tests passed`

### Current Limitations

- KHR runtime numerical validation still depends on a driver that can create and run the cooperative matrix pipeline. The current local AMD driver path remains protected by the existing structured-exception skip.
- The new M8 profile is forward inference only.
- KHR cooperative-matrix gradient/backpropagation remains missing.
- The sample path tracer still uses scalar inference/training instead of KHR inference.

### Next Step

Start the KHR training backend design, or add a profile-specific validation layer that records pipeline creation success and numerical validation status separately from capability selection. The latter is useful before allowing downstream integrations to enable a KHR profile automatically.

## 2026-07-09 - Milestone 25 Profile Shader Configuration Contract

### Implemented

- Added public shader configuration ABI:
  - `nrc::ProfileShaderConfig`
  - `nrc::profile_shader_configs`
  - `nrc::find_profile_shader_config`
- Added profile-to-shader config records for:
  - `KHR_FP16_M16N16K16_SG32_WG128`
  - `KHR_FP16_M16N16K16_SG16_WG128`
  - `KHR_FP16_M8N16K16_SG32_WG128`
  - `SCALAR_REFERENCE_WG64`
- Extended `nrc_profile_layout_test` to verify:
  - every implemented profile has a shader config;
  - config backend matches profile backend;
  - config workgroup size matches profile workgroup size;
  - KHR config subgroup and cooperative matrix dimensions match the selected profile;
  - scalar config does not claim subgroup/cooperative-matrix requirements.
- Updated `nrc_vulkan_khr_dispatch_test` to choose SG32, SG16, or M8/SG32 SPIR-V through `find_profile_shader_config`.
- Updated documentation:
  - `docs/profile_and_layout.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/vulkan_dispatch_harness.md`

### Behavior

Downstream integrations can now use one profile-selection path:

1. query capabilities;
2. call `choose_profile`;
3. call `find_profile_shader_config`;
4. select or compile the shader variant using the returned workgroup, subgroup, and cooperative-matrix dimensions.

For KHR cooperative matrix profiles, the config corresponds to shader defines:

```text
NRC_KHR_WORKGROUP_SIZE
NRC_KHR_SUBGROUP_SIZE
NRC_KHR_COOPMAT_M
NRC_KHR_COOPMAT_N
NRC_KHR_COOPMAT_K
```

This keeps C++ profile selection synchronized with shader compilation as matrix-size support grows.

### Validation

- Targeted build succeeded:
  - `cmake --build build --config Debug --target nrc_profile_layout_test`
- Targeted test succeeded:
  - `ctest --test-dir build -C Debug -R nrc_profile_layout_test --output-on-failure`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `25/25 tests passed`

### Current Limitations

- The config describes shader variants; it still does not prove the selected driver's pipeline creation or numerical execution succeeds.
- There is not yet a reusable Vulkan pipeline-validation API that records capability support, shader compile variant, pipeline creation status, and numerical validation status in one report.
- KHR cooperative-matrix training remains missing.

### Next Step

Add a profile validation report API, preferably in `runtime/vulkan`, that consumes queried capabilities plus profile shader config and reports why a profile is unavailable or whether it has passed pipeline creation/numerical validation. This will make downstream automatic profile enablement less brittle.

## 2026-07-09 - Milestone 26 Profile Support Diagnostics

### Implemented

- Added public capability-level profile diagnostics:
  - `nrc::ProfileSupportStatus`
  - `nrc::ProfileSupportReport`
  - `nrc::evaluate_profile_support`
- Kept `nrc::supports_profile` behavior synchronized by delegating it to `evaluate_profile_support(...).supported`.
- Added diagnostic statuses for:
  - supported profile;
  - unimplemented profile;
  - scalar reference backend not explicitly allowed;
  - unsupported backend;
  - missing required feature bundle;
  - missing training float atomic add;
  - missing required subgroup size;
  - missing cooperative matrix shape.
- Updated `docs/profile_and_layout.md` with the support diagnostics contract.

### Behavior

`evaluate_profile_support` reports the first decisive rejection reason for one profile and carries the matched profile id/name plus the final supported bool. This keeps automatic profile selection simple while giving downstream integrations enough information to explain why a vendor/device profile was rejected.

The check order is intentionally conservative:

1. scalar reference backend requires explicit opt-in;
2. unsupported backend/profile implementation status are rejected before capability checks;
3. KHR profiles require the common feature bundle before training atomics;
4. subgroup size is checked before matrix shape.

### Validation

- Targeted build succeeded:
  - `cmake --build build --config Debug --target nrc_profile_layout_test`
- Targeted test succeeded:
  - `ctest --test-dir build -C Debug -R nrc_profile_layout_test --output-on-failure`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `25/25 tests passed`

### Current Limitations

- Diagnostics are still capability-level only.
- The report does not include Vulkan shader module creation, pipeline creation, dispatch success, or numerical validation status.
- KHR cooperative-matrix training remains missing.

### Next Step

Add a Vulkan runtime validation report that combines profile support diagnostics, shader config selection, pipeline creation status, and optional numerical validation into one integration-facing result.

## 2026-07-09 - Milestone 27 Vulkan Profile Validation Report

### Implemented

- Added Vulkan-facing validation report API in `nrc_vulkan`:
  - `nrc::vulkan::VulkanValidationStepStatus`
  - `nrc::vulkan::VulkanProfileValidationStatus`
  - `nrc::vulkan::VulkanValidationStep`
  - `nrc::vulkan::VulkanProfileValidationRequirements`
  - `nrc::vulkan::VulkanProfileValidationReport`
  - `nrc::vulkan::evaluate_vulkan_profile_validation`
- The report combines:
  - capability-level `nrc::ProfileSupportReport`;
  - selected profile shader config;
  - caller-supplied pipeline creation status;
  - caller-supplied numerical validation status;
  - final `usable` decision.
- Extended `nrc_vulkan_capabilities_test` to cover:
  - supported KHR profile with shader config;
  - unsupported capability report forwarding;
  - missing shader config;
  - shader config/backend mismatch;
  - required pipeline creation not run, skipped, passed, and failed;
  - required numerical validation not run and failed.
- Updated documentation:
  - `docs/vulkan_capability_query.md`
  - `docs/vulkan_dispatch_harness.md`
  - `README.md`

### Behavior

The new report is deliberately integration-facing rather than a pipeline builder. Applications such as MCVR can create devices, shader modules, and pipelines using their own renderer infrastructure, then feed the outcome back through `VulkanProfileValidationRequirements`.

The final status is selected in this order:

1. profile capability support must pass;
2. shader config must exist;
3. shader config backend must match the selected profile;
4. failed pipeline creation always rejects the profile;
5. required pipeline creation rejects `NotRun` and `Skipped`;
6. failed numerical validation always rejects the profile;
7. required numerical validation rejects `NotRun` and `Skipped`.

`report.usable` is true only when all required checks are satisfied.

### Validation

- Targeted build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_capabilities_test`
- Targeted test succeeded:
  - `ctest --test-dir build -C Debug -R nrc_vulkan_capabilities_test --output-on-failure`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `25/25 tests passed`

### Current Limitations

- The Vulkan validation report does not create Vulkan objects by itself.
- The report stores a coarse numerical validation status, but not numerical error metrics, sample counts, or per-profile tolerance details.
- The KHR dispatch test still owns the concrete shader module, pipeline layout, descriptor, dispatch, and CPU/GPU comparison harness.
- KHR cooperative-matrix training remains missing.

### Next Step

Promote reusable pieces from the KHR dispatch harness into `runtime/vulkan`: a guarded compute pipeline creation helper that consumes `ProfileShaderConfig`, records `VkResult`/structured-exception outcomes into `VulkanValidationStep`, and remains independent from the sample/test descriptor layout.

## 2026-07-09 - Milestone 28 Vulkan Profile Compute Pipeline Helper

### Implemented

- Added reusable Vulkan pipeline helper API:
  - `include/nrc/vulkan/nrc_vulkan_pipeline.hpp`
  - `runtime/vulkan/nrc_vulkan_pipeline.cpp`
  - `nrc::vulkan::VulkanComputePipelineCreateInfo`
  - `nrc::vulkan::VulkanComputePipelineCreateResult`
  - `nrc::vulkan::create_profile_compute_pipeline`
- Added `structuredException` to `nrc::vulkan::VulkanValidationStep`.
- The helper consumes a caller-owned:
  - `VkDevice`;
  - `VkShaderModule`;
  - `VkPipelineLayout`;
  - `nrc::ProfileShaderConfig`.
- For KHR cooperative matrix profiles, the helper applies:
  - `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`;
  - `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT`.
- The helper returns:
  - created `VkPipeline`;
  - `VulkanValidationStepStatus`;
  - `VkResult`;
  - whether MSVC/Windows caught a structured exception from `vkCreateComputePipelines`.
- Refactored `nrc_vulkan_khr_dispatch_test` to use the helper instead of its local guarded pipeline creation code.
- Extended `nrc_vulkan_capabilities_test` to verify invalid pipeline creation requests fail safely and preserve structured-exception flags through profile validation reports.
- Updated documentation:
  - `docs/vulkan_capability_query.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/vulkan_dispatch_harness.md`
  - `README.md`

### Behavior

The helper is profile-aware but still renderer-neutral. It does not create a device, shader module, descriptor layout, pipeline layout, descriptor sets, command buffers, or resources. This keeps the integration contract suitable for projects such as MCVR, where the host renderer should keep ownership of its Vulkan object model.

For scalar/reference profiles, the helper creates a normal compute pipeline without subgroup-size pNext state. For KHR profiles, it derives subgroup/full-subgroup requirements from `ProfileShaderConfig`, keeping C++ profile selection and Vulkan pipeline creation synchronized.

Structured exceptions are treated as `VulkanValidationStepStatus::Skipped` with `structuredException = true` and `vkResult = VK_ERROR_UNKNOWN`, so callers can turn known driver crashes into controlled skips or disable a profile without crashing the host process.

### Validation

- Targeted build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_capabilities_test nrc_vulkan_khr_dispatch_test`
- Targeted tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_capabilities_test|nrc_vulkan_khr_dispatch_test" --output-on-failure`
- Verbose KHR dispatch test still reports a controlled local skip:
  - `SKIP: vkCreateComputePipelines raised a structured exception for the KHR cooperative matrix shader`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `25/25 tests passed`

### Current Limitations

- The helper does not load SPIR-V or create shader modules.
- The helper does not own or destroy the returned pipeline.
- The helper does not yet feed its result directly into a higher-level Vulkan numerical validation routine; the KHR dispatch test still owns dispatch and CPU/GPU comparison.
- The local AMD KHR dispatch path still cannot numerically validate because pipeline creation raises a structured exception and is reported as a controlled skip.
- KHR cooperative-matrix training remains missing.

### Next Step

Promote reusable shader module loading and validation-step plumbing from the dispatch tests into `runtime/vulkan`, then wire KHR dispatch pipeline creation results into `VulkanProfileValidationReport` in the test path.

## 2026-07-09 - Milestone 29 Vulkan Shader Module Validation Plumbing

### Implemented

- Added reusable Vulkan shader module helper API:
  - `include/nrc/vulkan/nrc_vulkan_shader.hpp`
  - `runtime/vulkan/nrc_vulkan_shader.cpp`
  - `nrc::vulkan::VulkanShaderModuleCreateInfo`
  - `nrc::vulkan::VulkanShaderModuleCreateResult`
  - `nrc::vulkan::create_shader_module`
- Extended Vulkan profile validation reports with shader module stage state:
  - `requireShaderModuleCreation`
  - `shaderModuleCreation`
  - `ShaderModuleCreationNotRun`
  - `ShaderModuleCreationSkipped`
  - `ShaderModuleCreationFailed`
- Refactored `nrc_vulkan_khr_dispatch_test` so the KHR path now:
  - creates shader modules through `nrc::vulkan::create_shader_module`;
  - records shader module creation in `VulkanProfileValidationReport`;
  - creates profile compute pipelines through `nrc::vulkan::create_profile_compute_pipeline`;
  - records pipeline creation in the same validation report.
- Extended `nrc_vulkan_capabilities_test` to cover:
  - invalid shader module creation requests;
  - required shader module creation not run;
  - passed shader module creation;
  - skipped shader module creation;
  - failed shader module creation with preserved `VkResult`.
- Updated documentation:
  - `docs/vulkan_capability_query.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/vulkan_dispatch_harness.md`
  - `README.md`

### Behavior

The Vulkan validation chain is now:

1. profile capability support;
2. shader config lookup and backend match;
3. shader module creation;
4. profile compute pipeline creation;
5. numerical validation.

`create_shader_module` intentionally accepts already-loaded SPIR-V words instead of reading files. That keeps the helper usable for engines with custom package formats, asset caches, or hot-reload systems.

`evaluate_vulkan_profile_validation` treats failed shader module creation as a decisive profile rejection. `NotRun` and `Skipped` only block when `requireShaderModuleCreation` is enabled, matching the existing pipeline/numerical validation policy.

### Validation

- Targeted build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_capabilities_test nrc_vulkan_khr_dispatch_test`
- Targeted verbose tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_capabilities_test|nrc_vulkan_khr_dispatch_test" --output-on-failure -V`
- Verbose KHR dispatch test still reports a controlled local skip:
  - `SKIP: vkCreateComputePipelines raised a structured exception for the KHR cooperative matrix shader`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `25/25 tests passed`

### Current Limitations

- The helper does not read SPIR-V files, compile GLSL, or own/destroy the returned shader module.
- The dispatch tests still keep their local SPIR-V file loading helpers.
- The report stores coarse shader/pipeline stage state but not detailed driver diagnostic strings.
- The local AMD KHR dispatch path still cannot numerically validate because pipeline creation raises a structured exception and is reported as a controlled skip.
- KHR cooperative-matrix training remains missing.

### Next Step

Promote a small reusable SPIR-V file loading helper for tests/sample use, or move the next layer up and extract descriptor/resource barrier helpers needed by scalar and KHR dispatch validation.

## 2026-07-09 - Milestone 30 Vulkan Storage Buffer Descriptor Helpers

### Implemented

- Added reusable storage-buffer descriptor helper API:
  - `include/nrc/vulkan/nrc_vulkan_descriptors.hpp`
  - `runtime/vulkan/nrc_vulkan_descriptors.cpp`
  - `nrc::vulkan::VulkanStorageBufferBinding`
  - `nrc::vulkan::VulkanStorageBufferDescriptorSetLayoutCreateInfo`
  - `nrc::vulkan::VulkanDescriptorSetLayoutCreateResult`
  - `nrc::vulkan::VulkanStorageBufferDescriptorWrite`
  - `nrc::vulkan::VulkanStorageBufferDescriptorUpdate`
  - `nrc::vulkan::create_storage_buffer_descriptor_set_layout`
  - `nrc::vulkan::update_storage_buffer_descriptors`
- The layout helper validates:
  - non-null device;
  - non-empty binding list;
  - no duplicate bindings;
  - non-zero stage flags.
- The descriptor update helper validates:
  - non-null device;
  - non-null descriptor set;
  - non-empty descriptor list;
  - non-null buffers;
  - non-zero ranges.
- Refactored:
  - `nrc_vulkan_scalar_dispatch_test`
  - `nrc_vulkan_khr_dispatch_test`
- Extended `nrc_vulkan_capabilities_test` to cover invalid descriptor layout/update requests.
- Updated documentation:
  - `docs/vulkan_capability_query.md`
  - `docs/vulkan_dispatch_harness.md`
  - `docs/khr_coopmat_backend.md`
  - `README.md`

### Behavior

The helper intentionally covers only the common scalar/KHR validation shape:

- storage buffers only;
- one descriptor per binding;
- explicit binding numbers;
- caller-owned descriptor pools and descriptor sets;
- caller-owned descriptor set layout lifetime.

This avoids imposing an allocator or descriptor pool policy on integrations such as MCVR while still removing repeated layout/write boilerplate from the validation harness.

### Validation

- Targeted build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_capabilities_test nrc_vulkan_scalar_dispatch_test nrc_vulkan_khr_dispatch_test`
- Targeted verbose tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_capabilities_test|nrc_vulkan_scalar_dispatch_test|nrc_vulkan_khr_dispatch_test" --output-on-failure -V`
- Verbose KHR dispatch test still reports a controlled local skip:
  - `SKIP: vkCreateComputePipelines raised a structured exception for the KHR cooperative matrix shader`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `25/25 tests passed`

### Current Limitations

- The helper does not allocate descriptor pools or descriptor sets.
- The helper does not handle storage images, sampled images, samplers, acceleration structures, descriptor arrays, or bindless descriptors.
- The scalar inference sample/test path still has image descriptor boilerplate outside this helper.
- The local AMD KHR dispatch path still cannot numerically validate because pipeline creation raises a structured exception and is reported as a controlled skip.
- KHR cooperative-matrix training remains missing.

### Next Step

Promote resource barrier helpers shared by scalar/KHR/queue/training dispatch tests, or add image descriptor helpers for the scalar inference pass and reference path tracer sample.

## 2026-07-09 - Milestone 31 Vulkan Memory Barrier Helpers

### Implemented

- Added reusable memory barrier helper API:
  - `include/nrc/vulkan/nrc_vulkan_barriers.hpp`
  - `runtime/vulkan/nrc_vulkan_barriers.cpp`
  - `nrc::vulkan::VulkanMemoryBarrier`
  - `nrc::vulkan::host_write_to_compute_shader_barrier`
  - `nrc::vulkan::compute_shader_write_to_host_read_barrier`
  - `nrc::vulkan::compute_shader_write_to_compute_shader_barrier`
  - `nrc::vulkan::cmd_memory_barrier`
- `cmd_memory_barrier` validates:
  - non-null command buffer;
  - non-zero source stage mask;
  - non-zero destination stage mask.
- Refactored:
  - `nrc_vulkan_scalar_dispatch_test`
  - `nrc_vulkan_khr_dispatch_test`
- Extended `nrc_vulkan_capabilities_test` to cover:
  - host-write to compute-shader barrier preset;
  - compute-shader write to host-read barrier preset;
  - invalid memory barrier request handling.
- Updated documentation:
  - `docs/vulkan_capability_query.md`
  - `docs/vulkan_dispatch_harness.md`
  - `docs/khr_coopmat_backend.md`
  - `README.md`

### Behavior

The helper records one `VkMemoryBarrier` through `vkCmdPipelineBarrier` and returns a `VulkanValidationStep`. The scalar and KHR forward dispatch tests now use it for:

- host writes becoming compute shader reads;
- compute shader writes becoming host reads.

The helper intentionally remains low-level. It does not allocate command buffers, begin/end command buffers, submit work, or infer resource ownership.

### Validation

- Targeted build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_capabilities_test nrc_vulkan_scalar_dispatch_test nrc_vulkan_khr_dispatch_test`
- Targeted verbose tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_capabilities_test|nrc_vulkan_scalar_dispatch_test|nrc_vulkan_khr_dispatch_test" --output-on-failure -V`
- Verbose KHR dispatch test still reports a controlled local skip:
  - `SKIP: vkCreateComputePipelines raised a structured exception for the KHR cooperative matrix shader`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `25/25 tests passed`

### Current Limitations

- The helper only records `VkMemoryBarrier`.
- It does not yet cover buffer memory barriers, image memory barriers, or image layout transitions.
- The scalar inference test and sample still contain image transition boilerplate.
- Queue/training/gradient dispatch tests still contain local memory barrier wrappers or inline barriers.
- The local AMD KHR dispatch path still cannot numerically validate because pipeline creation raises a structured exception and is reported as a controlled skip.
- KHR cooperative-matrix training remains missing.

### Next Step

Extend the barrier helpers to cover image layout transitions used by scalar inference and the sample, or continue refactoring queue/training/gradient tests onto `cmd_memory_barrier` for the common compute/host barrier cases.

## 2026-07-09 - Milestone 32 Training Barrier Helper Adoption

### Implemented

- Refactored additional Vulkan validation tests onto `nrc::vulkan::cmd_memory_barrier`:
  - `nrc_vulkan_queue_ops_test`
  - `nrc_vulkan_gradient_scalar_test`
  - `nrc_vulkan_training_step_test`
- Queue ops now uses the reusable helper for:
  - host writes to compute shader reads/writes;
  - queue reset pass writes to queue push pass reads/writes;
  - compute shader writes to host reads.
- Scalar gradient now uses the reusable helper for:
  - host writes to compute shader reads/writes;
  - compute shader writes to host reads.
- Full scalar training step now uses the reusable helper for:
  - host writes to compute shader reads/writes;
  - prepare pass writes to gradient pass reads/writes and indirect dispatch reads;
  - gradient writes to optimizer reads/writes;
  - optimizer writes to host reads.
- Updated `docs/vulkan_dispatch_harness.md` to record the broader barrier helper coverage.

### Behavior

This milestone does not change the synchronization policy. It replaces local `VkMemoryBarrier` boilerplate and the training test's local `pipeline_barrier` wrapper with the shared runtime helper.

The training step still records custom stage/access masks where needed, especially for the compute-to-indirect-dispatch dependency:

```text
VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
-> VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT

VK_ACCESS_SHADER_WRITE_BIT
-> VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT
```

### Validation

- Targeted build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_queue_ops_test nrc_vulkan_gradient_scalar_test nrc_vulkan_training_step_test`
- Targeted verbose tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_queue_ops_test|nrc_vulkan_gradient_scalar_test|nrc_vulkan_training_step_test" --output-on-failure -V`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `25/25 tests passed`

### Current Limitations

- The helper still only records `VkMemoryBarrier`.
- Image layout transitions in scalar inference and the sample are still local boilerplate.
- Descriptor update helpers have not yet been adopted by queue/gradient/training tests.
- KHR cooperative-matrix training remains missing.

### Next Step

Either adopt storage-buffer descriptor helpers in queue/gradient/training tests, or add image layout transition helpers for scalar inference and the sample path tracer.
