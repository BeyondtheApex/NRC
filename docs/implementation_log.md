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

## 2026-07-09 - Milestone 33 Training Descriptor Helper Adoption

### Implemented

- Refactored additional Vulkan validation tests onto the reusable storage-buffer descriptor helpers:
  - `nrc_vulkan_queue_ops_test`
  - `nrc_vulkan_gradient_scalar_test`
  - `nrc_vulkan_training_step_test`
- Queue ops now uses:
  - `nrc::vulkan::create_storage_buffer_descriptor_set_layout`
  - `nrc::vulkan::update_storage_buffer_descriptors`
- Scalar gradient now uses the same descriptor helper pair for its 4-binding train/queue/weight/gradient layout.
- Full scalar training step now uses the same descriptor helper pair for its 6-binding prepare/gradient/optimize descriptor layout.
- Updated `docs/vulkan_dispatch_harness.md` to record descriptor helper coverage for queue, gradient, and full-training-step tests.

### Behavior

This milestone does not change descriptor binding numbers or resource ownership. Descriptor pools and descriptor sets are still allocated by the tests. The reusable helper only creates storage-buffer set layouts and writes storage-buffer descriptors.

The full scalar training step keeps the same binding order across prepare, gradient, and optimize passes. It now builds descriptor writes from the existing `std::array<const Buffer *, kDescriptorBindingCount>` input and delegates the Vulkan write boilerplate to `update_storage_buffer_descriptors`.

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

- The descriptor helper still covers storage buffers only.
- Image descriptors in scalar inference and the sample remain local boilerplate.
- Shader module and compute pipeline helpers have not yet been adopted by queue/gradient/training tests.
- KHR cooperative-matrix training remains missing.

### Next Step

Adopt shader module and compute pipeline helpers in queue/gradient/training tests, or add image descriptor/layout transition helpers for scalar inference and the sample path tracer.

## 2026-07-09 - Milestone 34 Basic Compute Pipeline Helper Adoption

### Implemented

- Added a plain compute pipeline helper for non-profile passes:
  - `nrc::vulkan::VulkanBasicComputePipelineCreateInfo`
  - `nrc::vulkan::create_compute_pipeline`
- Kept `nrc::vulkan::create_profile_compute_pipeline` for KHR/profile-aware dispatches and made it share the guarded creation path used by the plain helper.
- Extended `nrc_vulkan_capabilities_test` to cover invalid plain compute pipeline creation requests.
- Refactored additional Vulkan validation tests to use:
  - `nrc::vulkan::create_shader_module`
  - `nrc::vulkan::create_compute_pipeline`
- Updated tests:
  - `nrc_vulkan_queue_ops_test`
  - `nrc_vulkan_gradient_scalar_test`
  - `nrc_vulkan_training_step_test`
- Updated documentation:
  - `docs/vulkan_capability_query.md`
  - `docs/vulkan_dispatch_harness.md`
  - `README.md`

### Behavior

The new plain helper is for utility passes that are not NRC profile implementations: queue ops, train prepare, scalar gradient, and optimizer dispatches. It creates a standard compute pipeline and records `VkResult` plus the MSVC/Windows structured-exception flag in `VulkanValidationStep`.

Profile-aware KHR inference still uses `create_profile_compute_pipeline`, which applies subgroup size and full-subgroup requirements from `ProfileShaderConfig`.

### Validation

- Targeted build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_capabilities_test nrc_vulkan_queue_ops_test nrc_vulkan_gradient_scalar_test nrc_vulkan_training_step_test`
- Targeted verbose tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_capabilities_test|nrc_vulkan_queue_ops_test|nrc_vulkan_gradient_scalar_test|nrc_vulkan_training_step_test" --output-on-failure -V`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `25/25 tests passed`

### Current Limitations

- Scalar dispatch still uses local shader module and compute pipeline creation.
- Scalar inference still uses local shader module, pipeline, descriptor image, and image transition boilerplate.
- The helper creates one compute pipeline at a time and does not manage pipeline caches beyond accepting a caller-owned `VkPipelineCache`.
- KHR cooperative-matrix training remains missing.

### Next Step

Adopt shader module/plain compute pipeline helpers in scalar dispatch, or start the image descriptor and image layout transition helper needed by scalar inference and the sample path tracer.

## 2026-07-09 - Milestone 35 Scalar Dispatch Pipeline Helper Adoption

### Implemented

- Refactored `nrc_vulkan_scalar_dispatch_test` to create its shader module through `nrc::vulkan::create_shader_module`.
- Refactored the same scalar dispatch test to create its non-profile compute pipeline through `nrc::vulkan::create_compute_pipeline`.
- Preserved the existing shader file loading, pipeline layout, descriptor set layout/update, command buffer, dispatch, and CPU/GPU numerical comparison ownership in the test harness.
- Added the same structured-exception guard used by the other plain compute dispatch tests, converting a driver-side `vkCreateComputePipelines` structured exception into a controlled `SkipTest`.
- Updated `docs/vulkan_dispatch_harness.md` so the helper coverage now includes scalar dispatch.

### Behavior

This milestone does not change the scalar backend ABI, descriptor binding order, push constants, dispatch size, or expected numerical output. It only moves shader module and compute pipeline creation into reusable Vulkan helper calls so the scalar dispatch correctness anchor uses the same guarded path as queue, gradient, and training-step tests.

### Validation

- Targeted build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_scalar_dispatch_test`
- Targeted verbose test succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_scalar_dispatch_test" --output-on-failure -V`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `25/25 tests passed`

### Current Limitations

- Scalar inference still uses local shader module and compute pipeline creation.
- Scalar inference and the sample still use local storage-image descriptor and image layout transition boilerplate.
- The sample path tracer still owns local shader module and compute pipeline creation for its compute passes.
- KHR cooperative-matrix training remains missing.

### Next Step

Adopt shader module/plain compute pipeline helpers in scalar inference, then add image descriptor and image layout transition helpers that can be shared by scalar inference and the sample path tracer.

## 2026-07-09 - Milestone 36 Scalar Inference Pipeline Helper Adoption

### Implemented

- Refactored `nrc_vulkan_inference_scalar_test` to create its shader module through `nrc::vulkan::create_shader_module`.
- Refactored the scalar inference compute pipeline creation path to use `nrc::vulkan::create_compute_pipeline`.
- Preserved the existing storage image creation, descriptor set layout, descriptor updates, image upload/readback, dispatch, and CPU/GPU validation logic.
- Added the same MSVC/Windows structured-exception guard used by the other plain compute pipeline tests.
- Updated `docs/vulkan_dispatch_harness.md` to document scalar inference helper usage.

### Behavior

The scalar inference test still validates the same screen-output image, training-record writes, and fp16-rounded CPU reference comparison. This milestone only moves shader module and pipeline creation behind the reusable Vulkan helper layer, which keeps the testing path aligned with scalar dispatch, queue ops, gradient, and full training-step dispatches.

### Validation

- Targeted build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_inference_scalar_test`
- Targeted verbose test succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_inference_scalar_test" --output-on-failure -V`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `25/25 tests passed`

### Current Limitations

- Test-side non-profile compute pipeline creation now uses helpers, but the sample path tracer still owns local shader module and compute pipeline creation.
- Scalar inference and the sample still use local storage-image descriptor and image layout transition boilerplate.
- Descriptor set layout helpers still cover storage buffers only.
- KHR cooperative-matrix training remains missing.

### Next Step

Add reusable storage-image descriptor helpers and image layout transition helpers, then adopt them in scalar inference before moving the same pattern into the sample path tracer.

## 2026-07-09 - Milestone 37 Storage Image Descriptor Helper Adoption

### Implemented

- Extended the Vulkan descriptor helper API with generic descriptor set layout creation:
  - `nrc::vulkan::VulkanDescriptorBinding`
  - `nrc::vulkan::VulkanDescriptorSetLayoutCreateInfo`
  - `nrc::vulkan::create_descriptor_set_layout`
- Kept `nrc::vulkan::create_storage_buffer_descriptor_set_layout` as a compatibility wrapper over the generic layout helper.
- Added storage-image descriptor update helpers:
  - `nrc::vulkan::VulkanStorageImageDescriptorWrite`
  - `nrc::vulkan::VulkanStorageImageDescriptorUpdate`
  - `nrc::vulkan::update_storage_image_descriptors`
- Extended `nrc_vulkan_capabilities_test` with invalid generic descriptor layout and invalid image descriptor update checks.
- Refactored `nrc_vulkan_inference_scalar_test` so its mixed buffer/image descriptor set layout uses `create_descriptor_set_layout`.
- Refactored scalar inference descriptor updates to use:
  - `nrc::vulkan::update_storage_buffer_descriptors`
  - `nrc::vulkan::update_storage_image_descriptors`
- Updated `README.md`, `docs/vulkan_capability_query.md`, and `docs/vulkan_dispatch_harness.md`.

### Behavior

This milestone does not change descriptor binding numbers, descriptor pool ownership, descriptor set allocation, image formats, image layouts, or shader ABI. Scalar inference still uses bindings 0/1/2/5 for storage buffers and bindings 3/4 for storage images; the difference is that layout/write boilerplate now goes through the reusable Vulkan descriptor layer.

The generic layout helper accepts explicit Vulkan descriptor types but still leaves descriptor pool allocation, descriptor set allocation, descriptor indexing policy, and descriptor binding to the host renderer. The update helpers intentionally cover only the storage-buffer and storage-image write patterns currently exercised by NRC tests.

### Validation

- Targeted build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_capabilities_test nrc_vulkan_inference_scalar_test`
- Targeted verbose tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_capabilities_test|nrc_vulkan_inference_scalar_test" --output-on-failure -V`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `25/25 tests passed`

### Current Limitations

- Image layout transitions in scalar inference and the sample remain local boilerplate.
- The sample path tracer still owns local descriptor layout/update boilerplate for its storage-image passes.
- Descriptor update helpers do not cover sampled images, samplers, acceleration structures, descriptor arrays, or bindless descriptors.
- KHR cooperative-matrix training remains missing.

### Next Step

Add reusable image layout transition helpers, adopt them in scalar inference, then migrate the sample path tracer descriptor and image-transition boilerplate onto the same helper layer.

## 2026-07-09 - Milestone 38 Image Pipeline Barrier Helper Adoption

### Implemented

- Extended the Vulkan barrier helper API with:
  - `nrc::vulkan::VulkanMemoryAccessBarrier`
  - `nrc::vulkan::VulkanImageBarrier`
  - `nrc::vulkan::VulkanPipelineBarrierInfo`
  - `nrc::vulkan::color_image_barrier`
  - `nrc::vulkan::cmd_pipeline_barrier`
- Kept the existing `nrc::vulkan::cmd_memory_barrier` API unchanged for simple memory-only barriers.
- Implemented `cmd_pipeline_barrier` so callers can record memory access barriers and image barriers in one `vkCmdPipelineBarrier` call.
- Extended `nrc_vulkan_capabilities_test` with:
  - color image barrier field checks;
  - invalid pipeline barrier request checks.
- Refactored `nrc_vulkan_inference_scalar_test` so storage-image upload, compute, and readback layout transitions use `color_image_barrier` plus `cmd_pipeline_barrier`.
- Refactored scalar inference host/transfer memory-only barriers to use `cmd_memory_barrier`.
- Updated `docs/vulkan_capability_query.md` and `docs/vulkan_dispatch_harness.md`.

### Behavior

This milestone preserves the scalar inference synchronization plan:

- host writes become visible to transfer reads;
- bias/factor images transition from `UNDEFINED` to `TRANSFER_DST_OPTIMAL`;
- copied images transition to `GENERAL` for compute;
- the bias image transitions to `TRANSFER_SRC_OPTIMAL` for readback;
- transfer writes become visible to host reads.

The new `cmd_pipeline_barrier` helper keeps the memory+image barrier cases combined into a single Vulkan pipeline barrier call, matching the previous local test code rather than splitting them into extra barriers.

### Validation

- Targeted build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_capabilities_test nrc_vulkan_inference_scalar_test`
- Targeted verbose tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_capabilities_test|nrc_vulkan_inference_scalar_test" --output-on-failure -V`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `25/25 tests passed`

### Current Limitations

- The helper records explicit barriers but does not track resource states or infer layout transitions.
- The sample path tracer still owns local descriptor, shader/pipeline, and image-transition boilerplate.
- Buffer image copy helpers are still local to scalar inference and the sample.
- KHR cooperative-matrix training remains missing.

### Next Step

Migrate the sample path tracer onto the reusable descriptor, shader module, compute pipeline, and barrier helpers, or add small buffer/image copy helpers before doing that migration.

## 2026-07-09 - Milestone 39 Sample Pipeline Helper Adoption

### Implemented

- Refactored `sample/path_tracer/sample.cpp` so the sample's shared `create_compute_pipeline` helper now uses:
  - `nrc::vulkan::create_shader_module`
  - `nrc::vulkan::create_compute_pipeline`
- This migrates all five sample compute passes onto the reusable shader/pipeline helper path:
  - producer;
  - scalar inference;
  - train prepare;
  - scalar gradient;
  - optimize.
- Added the same structured-exception guard used by the test dispatch paths, converting a driver-side `vkCreateComputePipelines` structured exception into a controlled `SkipTest`.
- Updated `docs/sample_path_tracer.md` with current helper usage and remaining sample-local descriptor/barrier limitations.

### Behavior

This milestone does not change the sample command graph, descriptors, shader ABI, resource layout, generated eval/train records, scalar inference output, gradient accumulation, optimizer update, or PPM output contract. It only replaces local shader module and compute pipeline boilerplate with the reusable Vulkan helper layer.

The sample still owns shader file loading, pipeline layouts, descriptor sets, command buffers, resources, and dispatch order. That keeps the helper boundary compatible with downstream engines such as MCVR, where renderer ownership should remain outside the NRC library.

### Validation

- Targeted sample build succeeded:
  - `cmake --build build --config Debug --target nrc_reference_path_tracer`
- Targeted verbose sample smoke test succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_sample_reference_path_tracer_smoke" --output-on-failure -V`
- Sample smoke output included:
  - `NRC reference path tracer minimal integration passed: 17 eval records, scalar inference resolve, and one scalar training update.`
  - `Wrote sample image: G:/cpp/radiance/NRC/build/sample/path_tracer/nrc_reference_path_tracer.ppm`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `25/25 tests passed`

### Current Limitations

- The sample path tracer still owns local descriptor layout/update boilerplate.
- The sample path tracer still owns local memory/image barrier boilerplate.
- Buffer/image copy helpers are still local to scalar inference and the sample.
- KHR cooperative-matrix training remains missing.

### Next Step

Migrate the sample path tracer descriptor layout/update paths onto `create_descriptor_set_layout`, `update_storage_buffer_descriptors`, and `update_storage_image_descriptors`, then migrate its image barrier recording onto `cmd_pipeline_barrier`.

## 2026-07-09 - Milestone 40 Sample Descriptor Helper Adoption

### Implemented

- Refactored `sample/path_tracer/sample.cpp` descriptor set layout creation to use `nrc::vulkan::create_descriptor_set_layout`.
- Added small sample-local wrappers around:
  - `nrc::vulkan::update_storage_buffer_descriptors`
  - `nrc::vulkan::update_storage_image_descriptors`
- Refactored all sample descriptor update paths onto those helpers:
  - producer set;
  - scalar inference set;
  - train prepare set;
  - scalar gradient set;
  - optimize set.
- Kept descriptor binding numbers and descriptor set ownership unchanged.
- Updated `docs/sample_path_tracer.md` to reflect current sample descriptor helper usage.

### Behavior

This milestone does not change descriptor pools, descriptor set allocation, binding numbers, shader ABI, dispatch order, resource ownership, sample output, or validation expectations.

The sample still owns descriptor pools and descriptor sets. The NRC helper layer now only removes repeated layout/write boilerplate and records invalid helper inputs consistently through `VulkanValidationStep`.

### Validation

- Targeted sample build succeeded:
  - `cmake --build build --config Debug --target nrc_reference_path_tracer`
- Targeted verbose sample smoke test succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_sample_reference_path_tracer_smoke" --output-on-failure -V`
- Sample smoke output included:
  - `NRC reference path tracer minimal integration passed: 17 eval records, scalar inference resolve, and one scalar training update.`
  - `Wrote sample image: G:/cpp/radiance/NRC/build/sample/path_tracer/nrc_reference_path_tracer.ppm`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `25/25 tests passed`

### Current Limitations

- The sample path tracer still owns local memory/image barrier boilerplate.
- Buffer/image copy helpers are still local to scalar inference and the sample.
- Descriptor pool allocation remains host-owned, by design.
- KHR cooperative-matrix training remains missing.

### Next Step

Migrate the sample path tracer memory/image barrier recording onto `cmd_memory_barrier`, `color_image_barrier`, and `cmd_pipeline_barrier`.

## 2026-07-09 - Milestone 41 Sample Barrier Helper Adoption

### Implemented

- Refactored `sample/path_tracer/sample.cpp` command recording so sample memory-only barriers use `nrc::vulkan::cmd_memory_barrier`.
- Refactored sample image layout transitions to use:
  - `nrc::vulkan::color_image_barrier`
  - `nrc::vulkan::cmd_pipeline_barrier`
- Kept memory+image synchronization points combined in one helper call through sample-local `record_memory_image_barriers`.
- Removed sample-local `VkImageMemoryBarrier` construction and direct `vkCmdPipelineBarrier` calls.
- Updated `docs/sample_path_tracer.md` to reflect current barrier helper usage.

### Behavior

This milestone preserves the sample synchronization plan:

- host writes become visible to transfer reads;
- screen images transition from `UNDEFINED` to `TRANSFER_DST_OPTIMAL`;
- uploaded images transition to `GENERAL` before producer/inference/training dispatches;
- compute pass writes become visible between reset, producer, inference, prepare, gradient, and optimize;
- the prepare-to-gradient barrier preserves indirect-dispatch visibility;
- the resolved bias image transitions to `TRANSFER_SRC_OPTIMAL` for readback;
- transfer writes become visible to host reads.

The number and placement of synchronization points remain aligned with the previous sample code. The change only routes barrier construction and recording through the reusable helper layer.

### Validation

- Targeted sample build succeeded:
  - `cmake --build build --config Debug --target nrc_reference_path_tracer`
- Targeted verbose sample smoke test succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_sample_reference_path_tracer_smoke" --output-on-failure -V`
- Sample smoke output included:
  - `NRC reference path tracer minimal integration passed: 17 eval records, scalar inference resolve, and one scalar training update.`
  - `Wrote sample image: G:/cpp/radiance/NRC/build/sample/path_tracer/nrc_reference_path_tracer.ppm`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `25/25 tests passed`

### Current Limitations

- Buffer/image copy helpers are still local to scalar inference and the sample.
- Resource allocation remains host-owned, by design.
- The sample is still a primary-ray triangle-loop integration skeleton, not a full path tracer.
- KHR cooperative-matrix training remains missing.

### Next Step

Decide whether to add small buffer/image copy helper wrappers, or shift from helper extraction back to missing NRC coverage: KHR cooperative-matrix training and richer iterative sample/path tracing behavior.

## 2026-07-09 - Milestone 42 KHR Backprop Shader Groundwork

### Implemented

- Added guarded KHR backpropagation declarations to `shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl` behind `NRC_NN_BACKPROPAGATION`:
  - `GL_EXT_shader_atomic_float` requirement;
  - configurable `NRC_NN_DWEIGHTS_BINDING`;
  - writable fp32 gradient buffer;
  - `NrcKhrGradientAtomicAdd`;
  - `NrcKhrOutputDerivative`, matching the scalar relative-L2 luminance derivative;
  - a larger shared-buffer allocation path reserved for per-subgroup gradient reductions.
- Added `tests/shader_compile/nrc_khr_backprop_smoke.comp`.
- Added compile smoke tests for all currently implemented KHR profile shader shapes:
  - `nrc_shader_khr_backprop_sg32_compile`
  - `nrc_shader_khr_backprop_sg16_compile`
  - `nrc_shader_khr_backprop_m8_sg32_compile`
- Updated `docs/khr_coopmat_backend.md` to distinguish backprop ABI groundwork from full KHR training support.

### Behavior

This milestone does not add a production KHR gradient pass yet. It establishes the first compile-checked shader-header boundary needed by a future KHR training implementation while preserving the existing forward inference entry point and profile shader variants.

The output derivative intentionally mirrors the scalar gradient pass and CPU reference so future KHR numerical tests can compare against the existing correctness path before optimizing the cooperative-matrix gradient accumulation.

### Validation

- Reconfigured CMake after adding CTest entries:
  - `cmake -S . -B build`
- Targeted KHR shader compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_khr" --output-on-failure -V`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `28/28 tests passed`

### Current Limitations

- Cooperative-matrix dA and dW propagation is still missing.
- KHR gradient reductions and gradient atomic accumulation are not wired through a full train-record pass.
- KHR runtime numerical validation still depends on a driver that can create and dispatch the selected cooperative-matrix pipeline.

### Next Step

Implement the output-layer dW path for the KHR backend first, validate its gradient layout against the scalar/CPU reference on a small batch, then extend backward propagation through hidden layers.

## 2026-07-09 - Milestone 43 KHR Output Gradient Layout Groundwork

### Implemented

- Added KHR backprop shared-buffer regions for:
  - final hidden-activation backup before output forward overwrites the shared activation tile;
  - future per-subgroup dW reductions.
- Added `NrcKhrLayerWeightFloatOffset` so KHR gradient writes use the same fp32 row-major weight index space as the scalar pass and CPU reference.
- Added KHR output-layer gradient helpers:
  - `NrcKhrStoreBackpropActivations`;
  - `NrcKhrReadBackpropActivation`;
  - `NrcKhrAccumulateOutputDWeights`.
- Updated `tests/shader_compile/nrc_khr_backprop_smoke.comp` so the smoke shader now explicitly:
  1. runs hidden-layer forward;
  2. stores final hidden activations for backpropagation;
  3. runs the output layer;
  4. computes the relative-L2 luminance output derivative;
  5. accumulates output-layer `dW` into the fp32 gradient buffer.
- Updated `docs/khr_coopmat_backend.md` to record the new output-gradient groundwork and remaining KHR training gaps.

### Behavior

The new output-layer `dW` helper intentionally writes the same gradient indices as the scalar gradient path:

```text
outputOffset + row * 64 + column
```

This keeps the first KHR training surface aligned with `NrcGradientScalar.glsl`, `nrc::reference_backward`, and the documented weight layout before replacing the helper with a cooperative-matrix reduction.

### Validation

- Targeted KHR shader compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_khr" --output-on-failure -V`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `28/28 tests passed`

### Current Limitations

- Output-layer `dW` accumulation is scalar per sample invocation and loops over all 64 input columns; it is currently compile-checked only and is not the final cooperative-matrix reduction path.
- Hidden-layer dA/dW propagation is still missing.
- There is still no full KHR train-record gradient pass or KHR runtime numerical comparison against the CPU/scalar reference.

### Next Step

Add a KHR train-record gradient pass skeleton that can dispatch the output-layer `dW` helper, then validate output-layer gradients against the scalar/CPU reference on devices that can create the selected KHR pipeline.

## 2026-07-09 - Milestone 44 KHR Gradient Pass Skeleton

### Implemented

- Added `shaders/nrc/passes/NrcGradientKhr.glsl`.
- Matched the scalar gradient pass ABI:
  - train records at binding 0;
  - queue state/train counts at binding 1;
  - packed fp16 weights at binding 2;
  - fp32 gradients at binding 3;
  - push constant `batch`;
  - integration-provided `NrcUnpackPackedInput`.
- The KHR pass now:
  1. clamps the requested batch and train count;
  2. encodes valid train records or zero-fills out-of-range workgroup lanes;
  3. runs KHR cooperative-matrix hidden forward;
  4. stores final hidden activations for output backpropagation;
  5. runs the output layer;
  6. computes the VkNRC-style relative-L2 luminance output derivative for valid records;
  7. accumulates output-layer `dW` into fp32 gradient storage.
- Added `tests/shader_compile/nrc_gradient_khr_dispatch.comp` with the same fake unpack hook shape as the scalar gradient compile test.
- Added CTest compile coverage for the KHR gradient pass profile variants:
  - `nrc_shader_gradient_khr_sg32_compile`
  - `nrc_shader_gradient_khr_sg16_compile`
  - `nrc_shader_gradient_khr_m8_sg32_compile`
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/training_optimizer.md`

### Behavior

The new pass is a real train-record shader-header entry point, but it intentionally covers only the first KHR training surface: output-layer `dW`. It preserves the scalar gradient pass resource layout so an integration can switch the gradient shader variant without changing descriptor binding order.

Out-of-range invocations cannot return early because KHR cooperative-matrix operations require workgroup/subgroup participation. They instead contribute zero input and zero gradients.

### Validation

- Reconfigured CMake after adding CTest entries:
  - `cmake -S . -B build`
- Targeted KHR gradient/profile shader compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr|nrc_shader_khr" --output-on-failure -V`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `31/31 tests passed`

### Current Limitations

- The KHR gradient pass accumulates only output-layer `dW`.
- Output-layer accumulation is still scalar per sample invocation, not a cooperative-matrix reduction.
- Hidden-layer dA/dW propagation is still missing.
- Runtime KHR gradient validation is not added yet because the current local KHR pipeline creation path remains driver-gated.

### Next Step

Add a KHR runtime gradient test that creates the KHR gradient pipeline when the driver allows it and compares output-layer gradients against the CPU/scalar reference, then continue with cooperative-matrix dA/dW propagation through hidden layers.

## 2026-07-09 - Milestone 45 KHR Gradient Runtime Harness

### Implemented

- Added `tests/vulkan_gradient_khr_test.cpp`.
- Added CMake SPIR-V generation for KHR gradient runtime variants:
  - `nrc_gradient_khr_dispatch_sg32.spv`
  - `nrc_gradient_khr_dispatch_sg16.spv`
  - `nrc_gradient_khr_dispatch_m8_sg32.spv`
  - `nrc_gradient_khr_dispatch_shaders`
- Added `nrc_vulkan_gradient_khr_test` and CTest registration.
- The runtime test combines the existing KHR inference and scalar gradient harness patterns:
  - selects a KHR profile with `ProfileRequirements::requireTraining = true`;
  - requires KHR cooperative matrix support and fp32 storage-buffer atomic add;
  - enables cooperative matrix, shader float16, storage buffer 16-bit access, subgroup size control, full compute subgroups, and shader atomic float features;
  - selects SG32, SG16, or M8/SG32 SPIR-V from `find_profile_shader_config`;
  - creates the shader module through `nrc::vulkan::create_shader_module`;
  - creates the pipeline through `nrc::vulkan::create_profile_compute_pipeline`;
  - records host/compute memory barriers through `nrc::vulkan::cmd_memory_barrier`;
  - uses the reusable storage-buffer descriptor layout/update helpers.
- If pipeline creation and dispatch succeed, the test compares:
  - output-layer gradients against an fp16-rounded CPU oracle that mirrors the current KHR forward/backprop groundwork;
  - hidden-layer gradient regions against zero, because hidden-layer dA/dW is not implemented yet.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/vulkan_dispatch_harness.md`

### Behavior

This milestone adds runtime validation scaffolding for the KHR gradient pass without claiming full KHR training correctness. The test proves that capability selection, descriptor ABI, profile-specific pipeline creation, dispatch recording, and output-layer gradient comparison are wired together whenever the driver can create the selected pipeline.

On the current local AMD driver, the KHR gradient pipeline follows the same protected path as KHR inference: `vkCreateComputePipelines` raises a structured exception and the test reports a controlled `SKIP:`.

### Validation

- Reconfigured CMake after adding CTest entries:
  - `cmake -S . -B build`
- Targeted KHR gradient test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime test succeeded with controlled skip:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr_test" --output-on-failure -V`
  - `SKIP: vkCreateComputePipelines raised a structured exception for the KHR gradient shader`
- Targeted KHR compile/runtime set succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr|nrc_vulkan_gradient_khr_test|nrc_vulkan_khr_dispatch_test" --output-on-failure -V`
  - KHR gradient and KHR inference runtime tests both reported controlled structured-exception skips.
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `32/32 tests passed`

### Current Limitations

- The KHR gradient runtime numerical comparison has not executed on this machine because KHR pipeline creation is still driver-gated.
- The KHR gradient pass still accumulates only output-layer `dW`.
- Output-layer accumulation is still scalar per sample invocation, not a cooperative-matrix reduction.
- Hidden-layer dA/dW propagation is still missing.

### Next Step

Implement KHR hidden-layer backpropagation helpers, starting with output derivative propagation to the final hidden layer (`dA5`) and a compile-checked hidden-layer `dW` layout path before replacing scalar accumulation with cooperative-matrix reductions.

## 2026-07-09 - Milestone 46 KHR Final Hidden Gradient Groundwork

### Implemented

- Extended `shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl` backpropagation helpers with:
  - one extra shared activation copy tile for source activations before the final hidden layer;
  - a generic activation-copy helper for storing multiple backprop snapshots;
  - fp16 packed weight loads for derivative propagation;
  - output-to-final-hidden derivative propagation;
  - scalar final hidden-layer `dW` accumulation in the shared row-major weight layout.
- Updated `shaders/nrc/passes/NrcGradientKhr.glsl` so the current KHR gradient path:
  1. runs hidden layers 0-3;
  2. stores `A4`, the activations feeding the final hidden layer;
  3. runs hidden layer 4;
  4. stores `A5`, the final hidden activations feeding the output layer;
  5. computes prediction and output derivative;
  6. accumulates output-layer `dW`;
  7. propagates output derivative to `dA5`;
  8. accumulates final hidden-layer `dW`.
- Updated `tests/shader_compile/nrc_khr_backprop_smoke.comp` to exercise the same split forward/backprop helper sequence.
- Extended `tests/vulkan_gradient_khr_test.cpp` so the CPU oracle and comparison cover:
  - output-layer `dW`;
  - final hidden-layer `dW`;
  - zero gradients for earlier hidden layers.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/training_optimizer.md`

### Behavior

This milestone keeps the KHR gradient path intentionally conservative. It does not attempt a new training algorithm; it follows the scalar/VkNRC-style chain rule over the existing 64-wide layout and only extends the validated surface from the output layer to the final hidden layer.

The final hidden-layer gradient writes:

```text
layer4Offset + row * 64 + column
```

with `row` selected by the final hidden pre-activation derivative and `column` selected by the saved source activation from the previous hidden layer.

### Validation

- Targeted KHR shader compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr|nrc_shader_khr_backprop" --output-on-failure -V`
- Targeted KHR gradient test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime test succeeded with controlled skip:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr_test" --output-on-failure -V`
  - `SKIP: vkCreateComputePipelines raised a structured exception for the KHR gradient shader`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `32/32 tests passed`

### Current Limitations

- Backpropagation through hidden layers 0-3 is still missing.
- The KHR `dW` helpers are still scalar per sample invocation, not cooperative-matrix reductions.
- Runtime numerical comparison has not executed on this machine because the current local AMD KHR pipeline creation path still reports a controlled structured-exception skip.

### Next Step

Continue hidden-layer backpropagation toward layers 3, 2, 1, and 0 while keeping the CPU oracle and shader pass in lockstep, then replace scalar per-sample accumulation with cooperative-matrix reduction once all layer regions are covered.

## 2026-07-09 - Milestone 47 KHR Hidden Layer 3 Gradient Groundwork

### Implemented

- Extended `shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl` backpropagation storage from two saved activation snapshots to three saved snapshots:
  - activations feeding hidden layer 3;
  - activations feeding hidden layer 4;
  - activations feeding the output layer.
- Added reusable scalar KHR backpropagation helpers:
  - generic hidden-layer derivative propagation through `W[layer]^T`;
  - generic hidden-layer `dW` accumulation from a saved destination activation and saved source activation;
  - wrappers for final hidden propagation and hidden layer 3 accumulation.
- Updated `shaders/nrc/passes/NrcGradientKhr.glsl` so the current KHR gradient path:
  1. runs hidden layers 0-2;
  2. stores the activations feeding hidden layer 3;
  3. runs hidden layer 3;
  4. stores the activations feeding hidden layer 4;
  5. runs hidden layer 4;
  6. stores the activations feeding the output layer;
  7. accumulates output-layer and hidden-layer 4 `dW`;
  8. propagates the hidden-layer 4 derivative to hidden-layer 3 activations;
  9. accumulates hidden-layer 3 `dW`.
- Updated `tests/shader_compile/nrc_khr_backprop_smoke.comp` to exercise the same split forward/backprop helper sequence.
- Extended `tests/vulkan_gradient_khr_test.cpp` so the fp16-rounded KHR CPU oracle and comparison cover:
  - output-layer `dW`;
  - hidden-layer 4 `dW`;
  - hidden-layer 3 `dW`;
  - zero gradients for hidden layers 0-2.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/training_optimizer.md`

### Behavior

This milestone moves the KHR gradient groundwork one hidden layer closer to full coverage while preserving the current correctness-first strategy. The new hidden-layer propagation uses the same chain-rule ordering as the scalar reference:

```text
dA3 = transpose(W4) * dZ4
dW3 = dZ3 * transpose(A2)
```

where `dZ4` and `dZ3` are gated by the saved ReLU outputs for hidden layers 4 and 3.

### Validation

- Targeted KHR shader compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr|nrc_shader_khr_backprop" --output-on-failure -V`
- Targeted KHR gradient test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `32/32 tests passed`
- Targeted KHR gradient runtime test succeeded with controlled skip:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr_test" --output-on-failure -V`
  - `SKIP: vkCreateComputePipelines raised a structured exception for the KHR gradient shader`

### Current Limitations

- Backpropagation through hidden layers 0-2 is still missing.
- The KHR `dW` helpers are still scalar per sample invocation, not cooperative-matrix reductions.
- More saved activation snapshots increase shared-memory pressure; this is acceptable for the correctness milestone but should be revisited before a production KHR training path.
- Runtime numerical comparison has not executed on this machine because the current local AMD KHR pipeline creation path still reports a controlled structured-exception skip.

### Next Step

Continue the same lockstep shader/oracle expansion through hidden layers 2, 1, and 0, then replace the scalar gradient accumulation helpers with subgroup/cooperative-matrix reduction once full-layer KHR gradient coverage exists.

## 2026-07-09 - Milestone 48 KHR Hidden Layer 2 Gradient Groundwork

### Implemented

- Extended `shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl` backpropagation storage from three saved activation snapshots to four saved snapshots:
  - activations feeding hidden layer 2;
  - activations feeding hidden layer 3;
  - activations feeding hidden layer 4;
  - activations feeding the output layer.
- Added the hidden-layer 2 activation snapshot base and store/read helpers.
- Removed the final-hidden/previous-hidden wrapper path from the KHR gradient pass usage and switched `shaders/nrc/passes/NrcGradientKhr.glsl` to call the generic hidden-layer propagation and `dW` accumulation helpers directly.
- Updated `shaders/nrc/passes/NrcGradientKhr.glsl` so the current KHR gradient path:
  1. runs hidden layers 0-1;
  2. stores the activations feeding hidden layer 2;
  3. runs hidden layer 2;
  4. stores the activations feeding hidden layer 3;
  5. runs hidden layer 3;
  6. stores the activations feeding hidden layer 4;
  7. runs hidden layer 4;
  8. stores the activations feeding the output layer;
  9. accumulates output-layer and hidden-layer 4/3/2 `dW`.
- Updated `tests/shader_compile/nrc_khr_backprop_smoke.comp` to exercise the same direct generic helper sequence.
- Refactored `tests/vulkan_gradient_khr_test.cpp` so the fp16-rounded KHR CPU oracle now stores every hidden-layer activation instead of only the currently covered suffix. The comparison currently covers:
  - output-layer `dW`;
  - hidden-layer 4 `dW`;
  - hidden-layer 3 `dW`;
  - hidden-layer 2 `dW`;
  - zero gradients for hidden layers 0-1.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/training_optimizer.md`

### Behavior

This milestone extends the correctness-first KHR gradient suffix from layers 4/3 to layers 4/3/2. The shader pass now computes:

```text
dA2 = transpose(W3) * dZ3
dW2 = dZ2 * transpose(A1)
```

while preserving the existing output-layer, hidden-layer 4, and hidden-layer 3 gradient behavior.

The CPU oracle now mirrors this as a layer-indexed hidden activation table, which should make the remaining layer 1/0 expansion less brittle than continuing to add one-off triplet/quartet helpers.

### Validation

- Targeted KHR shader compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr|nrc_shader_khr_backprop" --output-on-failure -V`
- Targeted KHR gradient test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `32/32 tests passed`
- Targeted KHR gradient runtime test succeeded with controlled skip:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr_test" --output-on-failure -V`
  - `SKIP: vkCreateComputePipelines raised a structured exception for the KHR gradient shader`

### Current Limitations

- Backpropagation through hidden layers 0-1 is still missing.
- The KHR `dW` helpers are still scalar per sample invocation, not cooperative-matrix reductions.
- Saved activation snapshots now cover four 128-sample activation tiles in shared memory. This remains a correctness-groundwork choice, not the final production-memory plan.
- Runtime numerical comparison has not executed on this machine because the current local AMD KHR pipeline creation path still reports a controlled structured-exception skip.

### Next Step

Continue the same shader/oracle expansion through hidden layers 1 and 0. Once all hidden layers are covered, compare the full KHR gradient layout against the scalar reference shape and then start replacing scalar per-sample accumulation with cooperative-matrix/subgroup reduction.

## 2026-07-09 - Milestone 49 KHR Hidden Layer 1 Gradient Groundwork

### Implemented

- Extended `shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl` backpropagation storage from four saved activation snapshots to five saved snapshots:
  - activations feeding hidden layer 1;
  - activations feeding hidden layer 2;
  - activations feeding hidden layer 3;
  - activations feeding hidden layer 4;
  - activations feeding the output layer.
- Added the hidden-layer 1 source activation snapshot base and store/read helpers.
- Updated `shaders/nrc/passes/NrcGradientKhr.glsl` so the current KHR gradient path:
  1. runs hidden layer 0;
  2. stores the activations feeding hidden layer 1;
  3. runs hidden layer 1;
  4. stores the activations feeding hidden layer 2;
  5. runs hidden layer 2;
  6. stores the activations feeding hidden layer 3;
  7. runs hidden layer 3;
  8. stores the activations feeding hidden layer 4;
  9. runs hidden layer 4;
  10. stores the activations feeding the output layer;
  11. accumulates output-layer and hidden-layer 4/3/2/1 `dW`.
- Updated `tests/shader_compile/nrc_khr_backprop_smoke.comp` to exercise the same layer-1 coverage path.
- Extended `tests/vulkan_gradient_khr_test.cpp` by moving the covered suffix start from hidden layer 2 to hidden layer 1. The comparison currently covers:
  - output-layer `dW`;
  - hidden-layer 4 `dW`;
  - hidden-layer 3 `dW`;
  - hidden-layer 2 `dW`;
  - hidden-layer 1 `dW`;
  - zero gradients for hidden layer 0.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/training_optimizer.md`

### Behavior

This milestone extends the correctness-first KHR gradient suffix from layers 4/3/2 to layers 4/3/2/1. The shader pass now computes:

```text
dA1 = transpose(W2) * dZ2
dW1 = dZ1 * transpose(A0)
```

while preserving the existing output-layer and hidden-layer 4/3/2 gradient behavior.

### Validation

- Targeted KHR shader compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr|nrc_shader_khr_backprop" --output-on-failure -V`
- Targeted KHR gradient test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `32/32 tests passed`
- Targeted KHR gradient runtime test succeeded with controlled skip:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr_test" --output-on-failure -V`
  - `SKIP: vkCreateComputePipelines raised a structured exception for the KHR gradient shader`

### Current Limitations

- Backpropagation through hidden layer 0 is still missing.
- The KHR `dW` helpers are still scalar per sample invocation, not cooperative-matrix reductions.
- Saved activation snapshots now cover five 128-sample activation tiles in shared memory. This remains a correctness-groundwork choice, not the final production-memory plan.
- Runtime numerical comparison has not executed on this machine because the current local AMD KHR pipeline creation path still reports a controlled structured-exception skip.

### Next Step

Add hidden-layer 0 coverage by saving the encoded input activation tile before the first hidden layer, then compare the full hidden/output KHR gradient layout against the scalar reference shape before moving to cooperative-matrix/subgroup reduction work.

## 2026-07-09 - Milestone 50 KHR Full Hidden/Output Gradient Groundwork

### Implemented

- Extended `shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl` with encoded-input source reads for hidden-layer 0 `dW`.
- Added `NrcKhrAccumulateInputHiddenDWeights`, which:
  - uses the saved hidden-layer 0 activation snapshot for the ReLU gate;
  - reads the encoded input from the invocation-local encoded array;
  - writes hidden-layer 0 `dW` into the same row-major fp32 gradient layout as the scalar path.
- Updated `shaders/nrc/passes/NrcGradientKhr.glsl` so the current KHR gradient path now:
  1. accumulates output-layer `dW`;
  2. backpropagates and accumulates hidden-layer 4 `dW`;
  3. backpropagates and accumulates hidden-layer 3 `dW`;
  4. backpropagates and accumulates hidden-layer 2 `dW`;
  5. backpropagates and accumulates hidden-layer 1 `dW`;
  6. backpropagates and accumulates hidden-layer 0 `dW`.
- Updated `tests/shader_compile/nrc_khr_backprop_smoke.comp` to exercise the hidden-layer 0 path.
- Extended `tests/vulkan_gradient_khr_test.cpp` so the fp16-rounded KHR CPU oracle and comparison cover every hidden/output weight region.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/training_optimizer.md`

### Behavior

This milestone completes the correctness-groundwork gradient coverage for the current 5-hidden-layer NRC network shape. The shader pass now computes hidden-layer 0 gradients as:

```text
dA0 = transpose(W1) * dZ1
dW0 = dZ0 * transpose(encodedInput)
```

The encoded input is kept in the invocation-local array instead of adding another full shared-memory activation snapshot. This keeps the current scalar layer-0 accumulation path aligned with the existing pass ABI while avoiding additional shared-memory pressure.

### Validation

- Targeted KHR shader compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr|nrc_shader_khr_backprop" --output-on-failure -V`
- Targeted KHR gradient test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `32/32 tests passed`
- Targeted KHR gradient runtime test succeeded with controlled skip:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr_test" --output-on-failure -V`
  - `SKIP: vkCreateComputePipelines raised a structured exception for the KHR gradient shader`

### Current Limitations

- Full hidden/output `dW` coverage exists, but the helpers are still scalar per sample invocation and not cooperative-matrix/subgroup reductions.
- The current correctness path still depends on fp32 storage-buffer atomic add.
- Runtime numerical comparison for the full KHR gradient did not execute on this machine because the current local AMD KHR pipeline creation path still reports a controlled structured-exception skip.
- Saved activation snapshots still cover five 128-sample activation tiles in shared memory. The layer-0 encoded input is kept local, but the broader saved-activation strategy remains a correctness-groundwork choice rather than the final production-memory plan.

### Next Step

Compare and document the full KHR-gradient groundwork against the scalar/reference gradient shape in more detail, then begin replacing scalar per-sample accumulation with cooperative-matrix/subgroup reduction while keeping the same full hidden/output coverage.

## 2026-07-09 - Milestone 51 Reusable KHR Gradient CPU Oracle

### Implemented

- Added `include/nrc/nrc_khr_reference.hpp`.
- Added `src/nrc_khr_reference.cpp` to `nrc_core`.
- Added `nrc::reference_backward_khr_fp16`, a reusable host-side oracle for the current KHR gradient path. It mirrors the KHR groundwork behavior:
  - consumes fp16-visible weights;
  - rounds hidden-layer ReLU outputs through fp16;
  - rounds output predictions through fp16;
  - computes the relative-L2 luminance output derivative;
  - accumulates every hidden/output `dW` region in the shared row-major fp32 layout;
  - supports `accumulate=true`;
  - validates weight and gradient buffer sizes.
- Refactored `tests/vulkan_gradient_khr_test.cpp` to use `nrc::reference_backward_khr_fp16` instead of carrying an anonymous copy of the KHR CPU oracle inside the Vulkan runtime test.
- Added `tests/khr_gradient_reference_test.cpp`.
- Registered `nrc_khr_gradient_reference_test` in CTest. The new test validates:
  - finite KHR fp16 prediction/gradients;
  - non-zero KHR gradients for every hidden layer and the output layer;
  - matching coverage shape against the scalar CPU reference path;
  - accumulate behavior;
  - invalid weight-count error handling.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/training_optimizer.md`

### Behavior

This milestone does not change the shader math. It moves the full hidden/output KHR fp16 gradient oracle into `nrc_core`, so the runtime test, future reduction tests, and integration-facing validation can use one authoritative host-side implementation.

This gives the next KHR reduction work a better anchor: shader changes can continue to compare against the same full-layer fp16 oracle instead of reimplementing expected gradients in each test harness.

### Validation

- Reconfigured CMake after adding source and test targets:
  - `cmake -S . -B build`
- Targeted KHR reference test build succeeded after rerunning sequentially:
  - `cmake --build build --config Debug --target nrc_khr_gradient_reference_test`
- Targeted KHR reference test succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_khr_gradient_reference_test" --output-on-failure -V`
- Targeted KHR gradient runtime test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime test succeeded with controlled skip:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr_test" --output-on-failure -V`
  - `SKIP: vkCreateComputePipelines raised a structured exception for the KHR gradient shader`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `33/33 tests passed`

### Current Limitations

- The reusable KHR oracle validates the current fp16 scalar-groundwork behavior; it is not a cooperative-matrix reduction implementation.
- Runtime numerical comparison for the KHR shader still did not execute on this machine because the local AMD KHR pipeline creation path reports a controlled structured-exception skip.
- The current shader path still uses scalar per-sample accumulation and fp32 storage-buffer atomics.

### Next Step

Use `nrc::reference_backward_khr_fp16` as the invariant oracle while introducing cooperative-matrix/subgroup gradient reductions. Start with one weight region, likely the output layer, and prove that the reduced shader path preserves the full hidden/output gradient layout before extending the reduction strategy through all hidden layers.

## 2026-07-09 - Milestone 52 KHR Output Workgroup Gradient Reduction

### Implemented

- Added `NrcKhrAccumulateOutputDWeightsWorkgroup` to `shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl`.
- Changed `shaders/nrc/passes/NrcGradientKhr.glsl` to use the workgroup output-layer `dW` helper.
- Updated `tests/shader_compile/nrc_khr_backprop_smoke.comp` so the backprop compile smoke test exercises the same helper as the pass.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/training_optimizer.md`

### Behavior

The output-layer `dW` path now stores each invocation's 3-value output derivative in shared memory, then lets the workgroup reduce every output weight across `NRC_KHR_WORKGROUP_SIZE` samples before issuing one fp32 atomic add per output weight per workgroup.

This preserves the current row-major fp32 gradient layout and keeps the hidden-layer `dW` path unchanged. Hidden layers still use scalar per-sample accumulation, so this is the first reduction step rather than the final cooperative-matrix/subgroup gradient path.

### Validation

- Targeted KHR shader compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr|nrc_shader_khr_backprop" --output-on-failure -V`
- Targeted KHR gradient test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime test succeeded with controlled skip:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr_test" --output-on-failure -V`
  - `SKIP: vkCreateComputePipelines raised a structured exception for the KHR gradient shader`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `33/33 tests passed`

### Current Limitations

- Output-layer `dW` reduction is a scalar workgroup loop, not a cooperative-matrix or subgroup reduction.
- Hidden-layer `dW` helpers still use scalar per-sample accumulation.
- The current correctness path still depends on fp32 storage-buffer atomic add.
- Runtime numerical comparison for the KHR shader still did not execute on this machine because the local AMD KHR pipeline creation path reports a controlled structured-exception skip.

### Next Step

Use the same `nrc::reference_backward_khr_fp16` oracle while replacing more scalar accumulation with subgroup or cooperative-matrix reduction, starting with a tighter output-layer reduction strategy or the first hidden-layer region.

## 2026-07-09 - Milestone 53 KHR Final Hidden Workgroup Gradient Reduction

### Implemented

- Added shared-output-derivative helpers to `shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl`:
  - `NrcKhrReadSharedOutputDerivative`
  - `NrcKhrPropagateSharedOutputDActivation`
- Added `NrcKhrAccumulateFinalHiddenDWeightsWorkgroup`, a scalar workgroup reduction helper for hidden-layer 4 `dW`.
- Changed `shaders/nrc/passes/NrcGradientKhr.glsl` so hidden-layer 4 uses the workgroup-reduced helper after output-to-hidden derivative propagation.
- Updated `tests/shader_compile/nrc_khr_backprop_smoke.comp` so the backprop compile smoke test exercises the same final-hidden helper as the pass.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/training_optimizer.md`

### Behavior

The output-layer workgroup reduction leaves each invocation's 3-value output derivative in the active shared-memory tile. The new final-hidden helper reuses that cached `dOutput` to recompute each sample's hidden-layer 4 derivative for the specific row being reduced:

```text
dA4[row] = transpose(Wout)[row] * dOutput
dZ4[row] = A4[row] > 0 ? dA4[row] : 0
dW4[row, col] = sum_samples(dZ4[row] * A3[col])
```

Each invocation owns a strided subset of the 64x64 hidden-layer 4 weight region, sums that weight across the full workgroup, and emits one fp32 atomic add per covered weight per workgroup. This preserves the existing row-major fp32 gradient layout and keeps `dHidden4` available locally for hidden-layer 3 propagation.

### Validation

- Targeted KHR shader compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr|nrc_shader_khr_backprop" --output-on-failure -V`
- Targeted KHR gradient test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime test succeeded with controlled skip:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr_test" --output-on-failure -V`
  - `SKIP: vkCreateComputePipelines raised a structured exception for the KHR gradient shader`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `33/33 tests passed`

### Current Limitations

- Output-layer and hidden-layer 4 `dW` reductions are scalar workgroup loops, not cooperative-matrix or subgroup reductions.
- Hidden-layer 3/2/1/0 `dW` helpers still use scalar per-sample accumulation.
- The current correctness path still depends on fp32 storage-buffer atomic add.
- Runtime numerical comparison for the KHR shader still did not execute on this machine because the local AMD KHR pipeline creation path reports a controlled structured-exception skip.
- The final-hidden helper deliberately recomputes hidden-layer 4 derivatives from cached output derivatives instead of storing a full fp32 `dA4` tile in shared memory. This avoids overwriting saved activation snapshots but is not yet the final performance plan.

### Next Step

Plan the next hidden-layer reduction step. Hidden-layer 3 and earlier cannot reuse only the cached 3-value output derivative; they need either a compact shared derivative snapshot, a recomputation strategy, or a subgroup/cooperative-matrix reduction design that preserves the saved activation snapshots and remains stable across SG16/SG32 and M8/M16 profiles.

## 2026-07-09 - Milestone 54 KHR Hidden Layer 3 Workgroup Gradient Reduction

### Implemented

- Added fp32 shared `dZ` scratch constants to `shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl`:
  - `NRC_KHR_FP32_PER_UVEC4`
  - `NRC_KHR_DPRE_UVEC4_PER_SAMPLE`
  - `NRC_KHR_DPRE_UVEC4_COUNT`
- Added compile-time checks that the 64-wide fp32 `dZ` scratch fits in the active tile plus the retired final-hidden activation snapshot.
- Added reusable shared-`dZ` helpers:
  - `NrcKhrStoreSharedDPreActivations`
  - `NrcKhrReadSharedDPreActivation`
  - `NrcKhrAccumulateSharedHiddenDWeightsWorkgroup`
- Changed `shaders/nrc/passes/NrcGradientKhr.glsl` so hidden-layer 3 stores its ReLU-gated derivative into shared scratch and accumulates `dW3` with one fp32 atomic add per weight per workgroup.
- Updated `tests/shader_compile/nrc_khr_backprop_smoke.comp` to exercise the shared-`dZ` helper path.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/training_optimizer.md`

### Behavior

After hidden-layer 4 derivative propagation computes each invocation's local `dHidden3`, the pass stores the ReLU-gated hidden-layer 3 derivative:

```text
dZ3[row] = A3[row] > 0 ? dA3[row] : 0
```

The scratch stores all 128 samples x 64 hidden features as fp32 values in the first 2048 `uvec4` entries of `nrcKhrSharedBuffer`. This intentionally overwrites the active tile and the hidden-layer 4 activation snapshot only after both are no longer needed. The still-needed snapshots for A3, A2, A1, and A0 start at later offsets and are preserved.

The new workgroup helper then computes:

```text
dW3[row, col] = sum_samples(dZ3[row] * A2[col])
```

and emits one fp32 atomic add per covered hidden-layer 3 weight per workgroup. The local `dHidden3` array remains available for propagating to hidden-layer 2, so this changes only the `dW3` accumulation strategy.

### Validation

- Targeted KHR shader compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr|nrc_shader_khr_backprop" --output-on-failure -V`
- Targeted KHR gradient test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime test succeeded with controlled skip:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr_test" --output-on-failure -V`
  - `SKIP: vkCreateComputePipelines raised a structured exception for the KHR gradient shader`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `33/33 tests passed`

### Current Limitations

- Output-layer, hidden-layer 4, and hidden-layer 3 `dW` reductions are scalar workgroup loops, not cooperative-matrix or subgroup reductions.
- Hidden-layer 2/1/0 `dW` helpers still use scalar per-sample accumulation.
- The current correctness path still depends on fp32 storage-buffer atomic add.
- Runtime numerical comparison for the KHR shader still did not execute on this machine because the local AMD KHR pipeline creation path reports a controlled structured-exception skip.
- Shared `dZ` scratch reuse depends on the current 5-hidden-layer backprop ordering and the current 64-wide/128-sample workgroup shape. It is guarded for this shape, but the final production plan should still revisit scratch lifetime and matrix-profile-specific memory pressure.

### Next Step

Reuse the shared-`dZ` helper for hidden-layer 2 and hidden-layer 1 once their local derivatives have been computed. Hidden-layer 0 still needs a separate source path because its source activation is the invocation-local encoded input rather than a saved shared activation snapshot.

## 2026-07-09 - Milestone 55 KHR Hidden Layer 2/1 Workgroup Gradient Reduction

### Implemented

- Reused the shared-`dZ` helper path from Milestone 54 for hidden-layer 2 and hidden-layer 1.
- Changed `shaders/nrc/passes/NrcGradientKhr.glsl` so:
  - hidden-layer 2 stores ReLU-gated `dZ2` into shared scratch and accumulates `dW2` with one fp32 atomic add per weight per workgroup;
  - hidden-layer 1 stores ReLU-gated `dZ1` into shared scratch and accumulates `dW1` with one fp32 atomic add per weight per workgroup.
- Updated `tests/shader_compile/nrc_khr_backprop_smoke.comp` so the compile smoke test exercises the hidden-layer 2/1 shared-`dZ` paths.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/training_optimizer.md`

### Behavior

The pass now uses the shared fp32 `dZ` scratch for hidden-layer 3, 2, and 1 `dW` accumulation. For hidden-layer 2:

```text
dZ2[row] = A2[row] > 0 ? dA2[row] : 0
dW2[row, col] = sum_samples(dZ2[row] * A1[col])
```

For hidden-layer 1:

```text
dZ1[row] = A1[row] > 0 ? dA1[row] : 0
dW1[row, col] = sum_samples(dZ1[row] * A0[col])
```

The scratch continues to reuse only the first 2048 `uvec4` entries of `nrcKhrSharedBuffer`, which overwrites the active tile and retired A4 snapshot. The saved A3/A2/A1/A0 snapshots used as source activations remain untouched at later offsets. The local `dHidden2` and `dHidden1` arrays are still used for derivative propagation to earlier layers.

### Validation

- Targeted KHR shader compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr|nrc_shader_khr_backprop" --output-on-failure -V`
- Targeted KHR gradient test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime test succeeded with controlled skip:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr_test" --output-on-failure -V`
  - `SKIP: vkCreateComputePipelines raised a structured exception for the KHR gradient shader`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `33/33 tests passed`

### Current Limitations

- Output-layer and hidden-layer 4/3/2/1 `dW` reductions are scalar workgroup loops, not cooperative-matrix or subgroup reductions.
- Hidden-layer 0 `dW` still uses scalar per-sample accumulation because its source activation is the invocation-local encoded input.
- The current correctness path still depends on fp32 storage-buffer atomic add.
- Runtime numerical comparison for the KHR shader still did not execute on this machine because the local AMD KHR pipeline creation path reports a controlled structured-exception skip.
- Shared `dZ` scratch reuse is still tied to the current fixed 64-wide/128-sample shape. It is guarded for this shape, but final production reduction should revisit memory lifetime and profile-specific pressure.

### Next Step

Add a workgroup-reduced hidden-layer 0 path by staging the encoded input source features, or design a source reader abstraction that lets the shared-`dZ` helper reduce against either saved shared activations or invocation-local encoded activations without duplicating the entire loop.

## 2026-07-09 - Milestone 56 KHR Hidden Layer 0 Workgroup Gradient Reduction

### Implemented

- Added `NrcKhrStoreSharedEncodedInput` to `shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl`.
- Changed `shaders/nrc/passes/NrcGradientKhr.glsl` so hidden-layer 0 now:
  1. stores ReLU-gated `dZ0` into shared fp32 scratch;
  2. stages the invocation-local encoded input into the retired A0 snapshot region;
  3. reuses `NrcKhrAccumulateSharedHiddenDWeightsWorkgroup` to reduce `dW0` with one fp32 atomic add per weight per workgroup.
- Updated `tests/shader_compile/nrc_khr_backprop_smoke.comp` so the compile smoke test exercises the hidden-layer 0 encoded-source staging path.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/training_optimizer.md`

### Behavior

Hidden-layer 0 is now reduced across the workgroup like the other hidden layers. The pass computes:

```text
dZ0[row] = A0[row] > 0 ? dA0[row] : 0
dW0[row, col] = sum_samples(dZ0[row] * encodedInput[col])
```

The source activation is different from the other hidden layers because it is the per-invocation encoded input rather than a saved hidden activation snapshot. The pass therefore stages `encoded` into `NRC_KHR_BACKPROP_OLDEST_SOURCE_ACT_COPY_UVEC4_BASE` after `NrcKhrStoreSharedDPreActivations` has already consumed the A0 snapshot for the ReLU gate. At that point A0 is no longer needed, so the staged encoded input does not destroy any later backpropagation dependency.

With this milestone, every hidden/output `dW` region in the KHR gradient groundwork path uses scalar workgroup reduction. The gradient layout remains unchanged and still matches `nrc::reference_backward_khr_fp16`.

### Validation

- Targeted KHR shader compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr|nrc_shader_khr_backprop" --output-on-failure -V`
- Targeted KHR gradient test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime test succeeded with controlled skip:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr_test" --output-on-failure -V`
  - `SKIP: vkCreateComputePipelines raised a structured exception for the KHR gradient shader`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `33/33 tests passed`

### Current Limitations

- All hidden/output `dW` regions now use scalar workgroup reductions, but they are still not cooperative-matrix or subgroup reductions.
- The current correctness path still depends on fp32 storage-buffer atomic add.
- Runtime numerical comparison for the KHR shader still did not execute on this machine because the local AMD KHR pipeline creation path reports a controlled structured-exception skip.
- Shared scratch reuse is tied to the current fixed 64-wide/128-sample shape and fixed 5-hidden-layer backprop ordering. It is guarded for this shape, but final production reduction should revisit memory lifetime and profile-specific pressure.

### Next Step

Start replacing scalar workgroup loops with subgroup or cooperative-matrix reduction while keeping the same full hidden/output coverage and `nrc::reference_backward_khr_fp16` as the invariant oracle. A practical next step is an output-layer subgroup reduction path because it has only three output rows and already has the smallest source-activation footprint.

## 2026-07-09 - Milestone 57 KHR Output Subgroup Gradient Reduction

### Implemented

- Enabled `GL_KHR_shader_subgroup_arithmetic` for KHR backpropagation in `shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl`.
- Added `NrcKhrAccumulateOutputDWeightsSubgroup`, a subgroup-arithmetic output-layer `dW` reduction helper.
- Changed `shaders/nrc/passes/NrcGradientKhr.glsl` so the KHR gradient pass uses the subgroup output helper.
- Updated `tests/shader_compile/nrc_khr_backprop_smoke.comp` so the backprop compile smoke test exercises the subgroup output helper.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/training_optimizer.md`

### Behavior

The output-layer helper still stages each invocation's 3-value `dOutput` in shared memory. Work assignment changed from one invocation looping over all 128 samples for a weight to one subgroup cooperating on a weight:

```text
for weight = gl_SubgroupID; weight < outputWeightCount; weight += subgroupCount:
    each lane accumulates samples lane, lane + subgroupSize, ...
    value = subgroupAdd(partial)
    subgroupElect lane emits one fp32 atomic add
```

This keeps the previous one-atomic-per-output-weight-per-workgroup behavior while replacing the serial per-weight sample loop with subgroup arithmetic. The helper remains compatible with the SG32, SG16, and M8/SG32 shader variants because the sample stride uses the selected `NRC_KHR_SUBGROUP_SIZE` profile macro.

### Validation

- Targeted KHR shader compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr|nrc_shader_khr_backprop" --output-on-failure -V`
- Targeted KHR gradient test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime test succeeded with controlled skip:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr_test" --output-on-failure -V`
  - `SKIP: vkCreateComputePipelines raised a structured exception for the KHR gradient shader`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `33/33 tests passed`

### Current Limitations

- Output-layer `dW` now uses subgroup arithmetic, but it is not a cooperative-matrix reduction.
- Hidden-layer 4/3/2/1/0 `dW` still use scalar workgroup reduction loops.
- The current correctness path still depends on fp32 storage-buffer atomic add.
- Runtime numerical comparison for the KHR shader still did not execute on this machine because the local AMD KHR pipeline creation path reports a controlled structured-exception skip.
- The helper assumes required subgroup size control keeps runtime subgroup size aligned with `NRC_KHR_SUBGROUP_SIZE`, as already required by the selected KHR profile and pipeline creation path.

### Next Step

Apply the same subgroup-cooperative idea to a hidden-layer region, likely hidden-layer 4 first. The hidden-layer case needs a source activation snapshot and a shared `dZ` source, so it should be introduced one layer at a time while keeping the full-layer `nrc::reference_backward_khr_fp16` oracle unchanged.

## 2026-07-09 - Milestone 58 KHR Final Hidden Subgroup Gradient Reduction

### Implemented

- Added `NrcKhrAccumulateFinalHiddenDWeightsSubgroup` to `shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl`.
- Changed `shaders/nrc/passes/NrcGradientKhr.glsl` so hidden-layer 4 `dW` uses the subgroup helper.
- Updated `tests/shader_compile/nrc_khr_backprop_smoke.comp` so the backprop compile smoke test exercises the hidden-layer 4 subgroup helper.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/training_optimizer.md`

### Behavior

The final-hidden helper mirrors the previous scalar workgroup math but changes the sample reduction to subgroup arithmetic. Each subgroup owns a strided set of hidden-layer 4 weights:

```text
for weight = gl_SubgroupID; weight < hiddenWeightCount; weight += subgroupCount:
    row = weight / hiddenWidth
    col = weight % hiddenWidth
    each lane accumulates samples lane, lane + subgroupSize, ...
    dA4[row] = transpose(Wout)[row] * dOutput
    dZ4[row] = A4[row] > 0 ? dA4[row] : 0
    partial += dZ4[row] * A3[col]
    value = subgroupAdd(partial)
    subgroupElect lane emits one fp32 atomic add
```

This keeps the same one-atomic-per-hidden4-weight-per-workgroup behavior as the scalar workgroup helper while replacing the serial per-weight sample loop with subgroup arithmetic. It still reuses the output helper's cached `dOutput` in shared memory and keeps the local `dHidden4` array for propagation to hidden-layer 3.

### Validation

- Targeted KHR shader compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr|nrc_shader_khr_backprop" --output-on-failure -V`
- Targeted KHR gradient test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime test succeeded with controlled skip:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr_test" --output-on-failure -V`
  - `SKIP: vkCreateComputePipelines raised a structured exception for the KHR gradient shader`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `33/33 tests passed`

### Current Limitations

- Output-layer and hidden-layer 4 `dW` now use subgroup arithmetic, but they are not cooperative-matrix reductions.
- Hidden-layer 3/2/1/0 `dW` still use scalar workgroup reduction loops.
- The current correctness path still depends on fp32 storage-buffer atomic add.
- Runtime numerical comparison for the KHR shader still did not execute on this machine because the local AMD KHR pipeline creation path reports a controlled structured-exception skip.
- The helper assumes required subgroup size control keeps runtime subgroup size aligned with `NRC_KHR_SUBGROUP_SIZE`, as already required by the selected KHR profile and pipeline creation path.

### Next Step

Move the shared-`dZ` hidden-layer helper to subgroup arithmetic for hidden-layer 3, then reuse the same helper shape for hidden-layer 2/1/0 if the scratch lifetime and source staging remain valid.

## 2026-07-09 - Milestone 59 KHR Shared Hidden Subgroup Gradient Reduction

### Implemented

- Added `NrcKhrAccumulateSharedHiddenDWeightsSubgroup` to `shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl`.
- Changed `shaders/nrc/passes/NrcGradientKhr.glsl` so hidden-layer 3/2/1/0 `dW` use the shared hidden subgroup helper.
- Updated `tests/shader_compile/nrc_khr_backprop_smoke.comp` so the backprop compile smoke test exercises hidden-layer 3/2/1/0 through the subgroup helper.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/training_optimizer.md`

### Behavior

The shared hidden helper keeps the existing shared fp32 `dZ` scratch and source activation staging strategy, but changes the sample reduction from one invocation looping over the whole workgroup to subgroup arithmetic:

```text
for weight = gl_SubgroupID; weight < hiddenWeightCount; weight += subgroupCount:
    row = weight / hiddenWidth
    col = weight % hiddenWidth
    each lane accumulates samples lane, lane + subgroupSize, ...
    partial += sharedDZ[row] * sourceActivation[col]
    value = subgroupAdd(partial)
    subgroupElect lane emits one fp32 atomic add
```

This helper covers hidden-layer 3/2/1 from saved source activation snapshots and hidden-layer 0 after the encoded input is staged into the retired A0 snapshot region. With this milestone, every hidden/output `dW` region in the KHR gradient groundwork path uses subgroup arithmetic while preserving the same row-major fp32 gradient layout.

### Validation

- Targeted KHR shader compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr|nrc_shader_khr_backprop" --output-on-failure -V`
- Targeted KHR gradient test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime test succeeded with controlled skip:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr_test" --output-on-failure -V`
  - `SKIP: vkCreateComputePipelines raised a structured exception for the KHR gradient shader`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `33/33 tests passed`

### Current Limitations

- All hidden/output `dW` regions now use subgroup arithmetic, but they are not cooperative-matrix reductions.
- The current correctness path still depends on fp32 storage-buffer atomic add.
- Runtime numerical comparison for the KHR shader still did not execute on this machine because the local AMD KHR pipeline creation path reports a controlled structured-exception skip.
- The subgroup helpers assume required subgroup size control keeps runtime subgroup size aligned with `NRC_KHR_SUBGROUP_SIZE`, as already required by the selected KHR profile and pipeline creation path.
- Shared scratch reuse is still tied to the current fixed 64-wide/128-sample shape and fixed 5-hidden-layer backprop ordering. It is guarded for this shape, but final production reduction should revisit memory lifetime and profile-specific pressure.

### Next Step

Start designing the cooperative-matrix reduction step or introduce a validation path that can run the KHR gradient numerically on a driver that creates the pipeline successfully. The shader now has full hidden/output subgroup-gradient coverage, so the next meaningful algorithmic improvement is cooperative-matrix accumulation rather than another scalar/subgroup loop reshuffle.

## 2026-07-09 - Milestone 60 KHR Subgroup Arithmetic Capability Gate

### Implemented

- Added renderer-neutral `subgroupArithmetic` capability reporting to `nrc::DeviceCapabilitiesView`.
- Added `nrc::ProfileSupportStatus::MissingSubgroupArithmetic` for precise profile diagnostics.
- Added `ProfileShaderConfig::requireSubgroupArithmeticForTraining` so integrations can see that KHR training shader variants depend on subgroup arithmetic.
- Updated Vulkan capability translation to set `subgroupArithmetic` from `VkPhysicalDeviceSubgroupProperties::supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT`.
- Updated profile selection so default KHR training profiles reject devices without subgroup arithmetic, while inference-only selection with `requireTraining = false` remains allowed.
- Updated:
  - `docs/profile_and_layout.md`
  - `docs/vulkan_capability_query.md`
  - `docs/khr_coopmat_backend.md`

### Behavior

The KHR gradient shader now has a matching host-side capability contract for its `subgroupAdd` reductions. This keeps shader compilation, profile selection, and Vulkan device capability reporting in sync:

```text
training KHR profile:
    requires fp32 buffer atomic add
    requires subgroup arithmetic

inference-only KHR profile:
    does not require fp32 buffer atomic add
    does not require subgroup arithmetic
```

The capability is treated as a subgroup operation property rather than a device feature. Device creation still enables the existing KHR cooperative matrix, float16, 16-bit storage, subgroup size control, and full subgroup features.

### Validation

- Targeted capability/profile build succeeded:
  - `cmake --build build --config Debug --target nrc_profile_layout_test nrc_vulkan_capabilities_test`
- Targeted capability/profile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_profile_layout_test|nrc_vulkan_capabilities_test" --output-on-failure -V`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Test result:
  - `33/33 tests passed`
- Targeted KHR runtime tests still pass as controlled skips on the local AMD driver:
  - `SKIP: vkCreateComputePipelines raised a structured exception for the KHR gradient shader`
  - `SKIP: vkCreateComputePipelines raised a structured exception for the KHR cooperative matrix shader`

### Current Limitations

- All hidden/output `dW` regions still use subgroup arithmetic rather than cooperative-matrix reductions.
- The current training correctness path still depends on fp32 storage-buffer atomic add.
- Runtime numerical comparison for the KHR shaders still did not execute on this machine because the local AMD KHR pipeline creation path reports controlled structured-exception skips.
- Capability/profile support still proves only that a device is a candidate. Integrations should keep strict shader module, pipeline, and numerical validation before enabling a production KHR NRC path.

### Next Step

Start designing the cooperative-matrix reduction step or add a validation route on hardware/driver combinations that can create the KHR pipelines successfully. The capability contract now reflects the subgroup-gradient implementation, so the next algorithmic improvement can focus on reducing atomic pressure and replacing subgroup reductions with cooperative-matrix accumulation where it is actually beneficial.

## 2026-07-09 - Milestone 61 Experimental KHR Output Cooperative-Matrix dW

### Implemented

- Added `NrcKhrAccumulateOutputDWeightsCoopmat` to `shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl`.
- Added output-derivative fp16 staging and output-dW subgroup partial layout constants for the current SG32, SG16, and M8/SG32 profile shapes.
- Added `NRC_KHR_USE_COOPMAT_OUTPUT_DW` to `shaders/nrc/passes/NrcGradientKhr.glsl`.
  - Default `0`: keep the existing subgroup-arithmetic output `dW` path.
  - `1`: use the new output-layer cooperative-matrix helper.
- Updated `tests/shader_compile/nrc_khr_backprop_smoke.comp` so the backprop smoke shader explicitly calls the new output cooperative-matrix helper.
- Added compile-only gradient dispatch tests for the optional output cooperative-matrix path:
  - `nrc_shader_gradient_khr_coopmat_output_sg32_compile`
  - `nrc_shader_gradient_khr_coopmat_output_sg16_compile`
  - `nrc_shader_gradient_khr_coopmat_output_m8_sg32_compile`
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/training_optimizer.md`

### Behavior

The helper mirrors the structure of VkNRC's NV output-layer `NNUpdateDW3` at a KHR-compatible level:

```text
stage dOutput as fp16 in the active shared tile
for each subgroup-owned sample block:
    load final hidden activations as an A cooperative matrix
    load staged dOutput as a B cooperative matrix
    accumulate a hidden-feature x output-row dW tile
store one fp16 partial dW tile per subgroup
one invocation per output weight:
    unpack fp16 partials from all subgroups
    reduce them in fp32
    atomicAdd into the existing fp32 row-major gradient buffer
```

The active shared tile is reused only after the saved final-hidden activation snapshot has been written outside that active tile. The helper therefore does not destroy the activation data needed for output `dW`.

This milestone deliberately keeps the default gradient pass on the previous subgroup-arithmetic path. The new helper stages derivatives and partial dW tiles as fp16, matching the broad shape of VkNRC's cooperative-matrix update path but differing from the current fp32 subgroup helper and `nrc::reference_backward_khr_fp16` oracle. It needs runtime numerical validation before becoming the default.

### Validation

- Targeted KHR shader compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr|nrc_shader_khr_backprop" --output-on-failure -V`
- The targeted test run covered:
  - existing KHR gradient SG32, SG16, and M8/SG32 compile tests;
  - new `NRC_KHR_USE_COOPMAT_OUTPUT_DW=1` SG32, SG16, and M8/SG32 compile tests;
  - KHR backprop smoke SG32, SG16, and M8/SG32 compile tests.
- Test result:
  - `9/9 tests passed`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Full test result:
  - `36/36 tests passed`
- Targeted KHR runtime tests still pass as controlled skips on the local AMD driver:
  - `SKIP: vkCreateComputePipelines raised a structured exception for the KHR gradient shader`
  - `SKIP: vkCreateComputePipelines raised a structured exception for the KHR cooperative matrix shader`

### Current Limitations

- The output cooperative-matrix `dW` helper is compile-validated only; it is not the default runtime path.
- The helper uses fp16 derivative/partial staging before fp32 cross-subgroup reduction and fp32 storage-buffer atomic add.
- Hidden-layer `dW` regions still use subgroup arithmetic by default.
- KHR `dA` propagation is still scalar-array based rather than VkNRC's cooperative-matrix backpropagation.
- Runtime numerical comparison for any KHR path still did not execute on this machine because the local AMD KHR pipeline creation path reports controlled structured-exception skips.

### Next Step

Either add a runtime A/B validation mode for `NRC_KHR_USE_COOPMAT_OUTPUT_DW=1` on a driver that can create the KHR pipeline, or continue porting the VkNRC fully fused shape by moving hidden-layer `dA` propagation and hidden-layer `dW` updates toward cooperative matrices under explicit compile-time/profile gates.

## 2026-07-09 - Milestone 62 KHR Output Coopmat Runtime Validation Entry

### Implemented

- Extended `tests/vulkan_gradient_khr_test.cpp` with a `--coopmat-output` mode.
- Added a test-side CPU oracle for the optional output-layer cooperative-matrix dW path:
  - hidden layers still use the existing `nrc::reference_backward_khr_fp16` shape;
  - output-layer dW is recomputed with fp16 output-derivative staging and fp16 output partial storage before fp32 comparison.
- Added runtime SPIR-V generation for `NRC_KHR_USE_COOPMAT_OUTPUT_DW=1`:
  - SG32;
  - SG16;
  - M8/SG32.
- Added `nrc_vulkan_gradient_khr_coopmat_output_test`, which runs the same Vulkan KHR gradient harness with the output-coopmat SPIR-V variants and the matching CPU oracle mode.
- Documented a hidden-layer cooperative-matrix dW design hazard:
  - the current shared dPre scratch lives in the active shared tile;
  - activation snapshots needed by later dA propagation live in the subsequent shared regions;
  - a naive hidden-dW partial-tile store would overwrite either dPre or snapshots before later propagation consumes them;
  - hidden-layer coopmat dW therefore needs pass reordering, smaller per-tile reduction storage, a different partial-reduction strategy, or a larger/profile-gated shared-memory layout.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/vulkan_dispatch_harness.md`
  - `docs/training_optimizer.md`

### Behavior

The default KHR gradient runtime test still receives the normal subgroup-output SPIR-V variants. The new output-coopmat test receives SPIR-V variants compiled with:

```text
NRC_KHR_USE_COOPMAT_OUTPUT_DW=1
```

When the test is run on a driver that can create the KHR gradient pipeline, it will compare:

```text
hidden-layer gradients: current KHR fp16 oracle
output-layer gradients: fp16 staged output-coopmat oracle
```

This gives the output cooperative-matrix dW helper a separate validation route without changing the default production candidate path.

### Validation

- Targeted KHR gradient test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- The build generated the new output-coopmat runtime SPIR-V variants:
  - `nrc_gradient_khr_coopmat_output_dispatch_sg32.spv`
  - `nrc_gradient_khr_coopmat_output_dispatch_sg16.spv`
  - `nrc_gradient_khr_coopmat_output_dispatch_m8_sg32.spv`
- Targeted KHR runtime tests succeeded with controlled skips:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr" --output-on-failure -V`
- Test result:
  - `2/2 tests passed`
- Local skip results:
  - default KHR gradient: `SKIP: vkCreateComputePipelines raised a structured exception for the KHR gradient shader`
  - output-coopmat KHR gradient: `SKIP: vkCreateComputePipelines raised a structured exception for the KHR gradient shader`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Full test result:
  - `37/37 tests passed`

### Current Limitations

- The local AMD driver still does not execute KHR numerical validation because pipeline creation raises a structured exception.
- The output-coopmat oracle approximates the intended fp16 derivative/partial staging path for the current small test batch; it should be revisited if larger batches span multiple subgroup ranges in a runtime-success environment.
- Hidden-layer cooperative-matrix dW is not implemented. The next version needs an explicit memory-lifetime/reordering plan rather than simply storing hidden partials over the existing shared snapshots.
- KHR dA propagation remains scalar-array based.

### Next Step

Either validate the output-coopmat runtime path on a driver that successfully creates the KHR gradient pipeline, or design the hidden-layer cooperative-matrix dW memory plan before adding code. The hidden plan should decide whether to reorder dA propagation before dW partial storage, reduce one tile at a time, accept more atomics, or introduce a separate profile with larger shared-memory requirements.

## 2026-07-09 - Milestone 63 Preserve Shared dOutput After Output Coopmat dW

### Implemented

- Fixed `NrcKhrAccumulateOutputDWeightsCoopmat` so it restores each invocation's fp32 output derivative into:

```text
nrcKhrSharedBuffer[gl_LocalInvocationID.x]
```

- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`

### Behavior

The optional output-coopmat dW helper temporarily uses shared memory for:

```text
fp16 staged dOutput
fp16 per-subgroup output dW partials
```

After those partials are reduced and atomically accumulated into the output gradient region, the helper rewrites the shared cache back to the same fp32 derivative layout used by the default subgroup-output helper:

```text
shared[sample].x = floatBitsToUint(dOutput.r)
shared[sample].y = floatBitsToUint(dOutput.g)
shared[sample].z = floatBitsToUint(dOutput.b)
```

This is required because `NrcKhrAccumulateFinalHiddenDWeightsSubgroup` reads output derivatives from shared memory for every sample while accumulating hidden-layer 4 `dW`. Without this restore step, a driver that successfully ran the optional output-coopmat path would compare output-layer gradients against the output-coopmat oracle but corrupt downstream hidden-layer gradients.

### Validation

- Targeted KHR shader compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr_coopmat_output|nrc_shader_khr_backprop" --output-on-failure -V`
- Test result:
  - `6/6 tests passed`
- Targeted KHR gradient test build succeeded and regenerated default plus output-coopmat runtime SPIR-V:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR runtime tests succeeded with controlled skips:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr" --output-on-failure -V`
- Test result:
  - `2/2 tests passed`
- Local skip results:
  - default KHR gradient: `SKIP: vkCreateComputePipelines raised a structured exception for the KHR gradient shader`
  - output-coopmat KHR gradient: `SKIP: vkCreateComputePipelines raised a structured exception for the KHR gradient shader`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Full test result:
  - `37/37 tests passed`

### Current Limitations

- This fix is still compile/skip validated locally because the local AMD driver does not create the KHR gradient pipeline.
- The output-coopmat path still needs real runtime numerical validation on a successful KHR pipeline creation path.
- Hidden-layer cooperative-matrix dW still needs an explicit scratch lifetime and pass-order plan.

### Next Step

Return to the host-side KHR shared-memory layout model so shader shared-memory assumptions, derivative cache lifetime, and future hidden-dW partial storage constraints are visible from tests and profile diagnostics before more hidden cooperative-matrix code is added.

## 2026-07-09 - Milestone 64 Host KHR Shared-Memory Layout Model

### Implemented

- Added `include/nrc/nrc_khr_shader_layout.hpp`.
- Added `src/nrc_khr_shader_layout.cpp`.
- Added `nrc::compute_khr_shader_shared_memory_layout`, which mirrors the KHR shader shared-memory constants from `NrcNNKhr.glsl` for a selected `ProfileShaderConfig`.
- Added the new source file to `nrc_core`.
- Extended `tests/profile_layout_test.cpp` to validate the shared-memory layout for all implemented KHR profiles.
- Added `docs/khr_shared_memory_layout.md`.
- Updated:
  - `docs/profile_and_layout.md`
  - `docs/khr_coopmat_backend.md`

### Behavior

The new host layout model reports:

```text
active tile
final hidden snapshot
output source snapshot
previous source snapshot
earlier source snapshot
oldest source snapshot
dPre scratch
output derivative fp16 scratch
output dW partial scratch
hidden dW partial scratch
total shared uvec4 count
total shared bytes
```

For the currently implemented KHR profiles, the model records:

```text
active tile:        1024 uvec4
activation regions: 6144 uvec4
shared allocation:  6144 uvec4 = 98304 bytes
dPre scratch:       2048 uvec4
```

The output coopmat dW scratch fits in the active tile for SG32, SG16, and M8/SG32.

The hidden dW partial scratch result is profile-dependent:

```text
SG32:    2048 uvec4, preserves the first shared hidden activation snapshot if placed at offset 0
SG16:    4096 uvec4, would overwrite shared hidden activation snapshots if placed at offset 0
M8/SG32: 2048 uvec4, preserves the first shared hidden activation snapshot if placed at offset 0
```

This makes the hidden-layer cooperative-matrix dW lifetime issue test-visible instead of relying on hand inspection of shader constants.

### Validation

- Targeted profile/layout build succeeded:
  - `cmake --build build --config Debug --target nrc_profile_layout_test`
- Targeted profile/layout test succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_profile_layout_test" --output-on-failure -V`
- Test result:
  - `1/1 tests passed`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Full test result:
  - `37/37 tests passed`

### Current Limitations

- The model mirrors the current fixed KHR shader shape. It is not yet used to reject devices whose `maxComputeSharedMemorySize` is below 98304 bytes.
- Vulkan capability query still does not report `maxComputeSharedMemorySize`.
- Hidden-layer cooperative-matrix dW is still not implemented; the model only exposes scratch sizes and the SG16 snapshot-overwrite hazard.

### Next Step

Promote shared-memory size into Vulkan capabilities/profile support so KHR training profiles can reject devices that cannot support the current 98304-byte backprop shader allocation. After that, use the layout model to design a hidden-layer cooperative-matrix dW path that is valid for SG32, SG16, and M8/SG32 rather than relying on a single offset-zero partial storage assumption.

## 2026-07-09 - Milestone 65 KHR Training Shared-Memory Capability Gate

### Implemented

- Added `DeviceCapabilitiesView::maxComputeSharedMemorySize`.
- Added `VulkanDeviceCapabilities::maxComputeSharedMemorySize`.
- Updated `VulkanDeviceCapabilities::view()` to carry the shared-memory limit into renderer-neutral profile selection.
- Updated `nrc::vulkan::query_device_capabilities` to read `VkPhysicalDeviceProperties::limits.maxComputeSharedMemorySize`.
- Added `ProfileSupportStatus::MissingSharedMemory`.
- Updated `nrc::evaluate_profile_support` so KHR training profile support compares `maxComputeSharedMemorySize` against `compute_khr_shader_shared_memory_layout(config).sharedBytes`.
- Kept the shared-memory gate training-specific:
  - default `ProfileRequirements::requireTraining = true` requires the current 98304-byte KHR backprop allocation;
  - `ProfileRequirements::requireTraining = false` skips this training-only allocation so KHR inference profile selection is not rejected by backprop shared-memory limits.
- Extended `tests/profile_layout_test.cpp` to validate:
  - supported KHR profiles provide the 98304-byte shared-memory capability;
  - insufficient shared memory rejects full training profile selection;
  - inference-only profile selection still works with the same low shared-memory value;
  - the support report records `MissingSharedMemory`.
- Extended `tests/vulkan_capabilities_test.cpp` to validate:
  - `VulkanDeviceCapabilities::view()` carries `maxComputeSharedMemorySize`;
  - Vulkan profile validation propagates `MissingSharedMemory` as `ProfileNotSupported`.
- Updated:
  - `docs/profile_and_layout.md`
  - `docs/vulkan_capability_query.md`
  - `docs/khr_shared_memory_layout.md`
  - `docs/khr_coopmat_backend.md`

### Behavior

The active KHR training profile contract now includes:

```text
maxComputeSharedMemorySize >= 98304 bytes
```

This value comes from the host-side KHR shader layout model rather than a duplicated literal in the profile logic. That keeps future shared-memory layout changes visible to capability selection and tests.

The check runs after required feature, training atomic, subgroup arithmetic, subgroup size, and cooperative-matrix shape checks. This preserves more specific diagnostics for devices that are missing an earlier prerequisite.

### Validation

- Targeted profile/layout build succeeded:
  - `cmake --build build --config Debug --target nrc_profile_layout_test`
- Targeted Vulkan capabilities build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_capabilities_test`
- Targeted tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_profile_layout_test|nrc_vulkan_capabilities_test" --output-on-failure -V`
- Targeted test result:
  - `2/2 tests passed`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Full test result:
  - `37/37 tests passed`

### Current Limitations

- The capability gate only checks the current fixed KHR backprop shared-memory allocation. Future profile variants with smaller/larger shared layouts need to flow through the same host layout model.
- Hidden-layer cooperative-matrix dW is still not implemented. The shared-memory gate prevents unsupported training profiles, but it does not solve the SG16 hidden dW partial-storage lifetime issue.
- Runtime numerical validation for KHR cooperative-matrix pipelines is still driver-gated on the current local AMD path.

### Next Step

Use the shared-memory layout model to design the hidden-layer cooperative-matrix dW memory plan. The design should choose a stable lifetime/order strategy for SG32, SG16, and M8/SG32 before adding shader code, because the naive offset-zero hidden partial storage would overwrite activation snapshots for SG16.

## 2026-07-09 - Milestone 66 Hidden dW Tile-Streaming Layout Plan

### Implemented

- Added `KhrHiddenDWeightTileStreamingLayout` to `include/nrc/nrc_khr_shader_layout.hpp`.
- Extended `KhrShaderSharedMemoryLayout` with `hiddenDWeightTileStreaming`.
- Updated `compute_khr_shader_shared_memory_layout` to model a hidden-layer cooperative-matrix `dW` tile-streaming plan.
- Tightened KHR layout validity checks so the layout model requires:
  - hidden width divisible by cooperative matrix `N`;
  - workgroup size divisible by cooperative matrix `N`.
- Extended `tests/profile_layout_test.cpp` to validate, for every implemented KHR profile:
  - hidden row-block count;
  - hidden column-block count;
  - tile count;
  - row-block `dPre` scratch offset and size;
  - per-tile partial size;
  - per-subgroup partial scratch size;
  - combined scratch size;
  - active-tile fit;
  - activation snapshot preservation.
- Added a scalar-profile negative test so non-KHR shader configs do not expose a hidden `dW` tile-streaming plan.
- Updated:
  - `docs/khr_shared_memory_layout.md`
  - `docs/profile_and_layout.md`
  - `docs/khr_coopmat_backend.md`

### Behavior

The plan avoids full-layer hidden `dW` partial storage. Full-layer partial storage would require:

```text
SG32:    2048 uvec4
SG16:    4096 uvec4
M8/SG32: 2048 uvec4
```

That is especially problematic for SG16 if placed at offset zero because it would overwrite saved activation snapshots still needed by dA propagation.

The new plan streams one hidden `dW` tile at a time. It stages one row block of fp16 `dPre` and one MxN partial tile per subgroup inside the active tile:

```text
M16/SG32: dPre row 256 uvec4 + partials 128 uvec4 = 384 uvec4
M16/SG16: dPre row 256 uvec4 + partials 256 uvec4 = 512 uvec4
M8/SG32:  dPre row 128 uvec4 + partials  64 uvec4 = 192 uvec4
```

All current profiles fit this combined scratch region inside the 1024-uvec4 active tile, ending before the first saved activation snapshot at offset 1024. This gives the shader implementation a stable cross-profile memory plan that does not require reordering all dA propagation or adding a larger shared-memory profile.

### Validation

- Targeted profile/layout build succeeded:
  - `cmake --build build --config Debug --target nrc_profile_layout_test`
- Targeted profile/layout test succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_profile_layout_test" --output-on-failure -V`
- Targeted test result:
  - `1/1 tests passed`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Full test result:
  - `37/37 tests passed`

### Current Limitations

- This milestone is a host-side layout and lifetime model. It does not yet add the GLSL hidden-layer cooperative-matrix `dW` helper.
- The planned helper will likely stage fp16 `dPre` row blocks, so it needs a dedicated oracle or tolerance path similar to the optional output-coopmat `dW` route.
- Runtime numerical validation for KHR cooperative-matrix pipelines is still driver-gated on the current local AMD path.

### Next Step

Implement an optional compile-validated hidden-layer cooperative-matrix `dW` helper using the tile-streaming plan. Start with one hidden-layer helper entry point and shader compile tests for SG32, SG16, and M8/SG32, then decide how to hook it into the full gradient pass without changing the default path before numerical validation exists.

## 2026-07-09 - Milestone 67 Experimental Final-Hidden Coopmat dW Helper

### Implemented

- Added KHR shader macros for the hidden-layer cooperative-matrix `dW` tile-streaming scratch layout:
  - row-block fp16 `dPre` scratch;
  - row-major source activation cooperative-matrix load offsets;
  - per-subgroup MxN partial tile scratch;
  - combined active-tile scratch size.
- Added a compile-time shader guard requiring hidden `dW` tile-streaming scratch to fit inside the active tile.
- Added `NrcKhrStageHiddenDPreRowFp16`.
- Added `NrcKhrAccumulateHiddenDWeightsCoopmatTileStreaming`.
- Updated `tests/shader_compile/nrc_khr_backprop_smoke.comp` to call the helper after the existing final-hidden subgroup helper has consumed the shared fp32 output-derivative cache.
- Added `NRC_KHR_USE_COOPMAT_FINAL_HIDDEN_DW` to `NrcGradientKhr.glsl`.
- When `NRC_KHR_USE_COOPMAT_FINAL_HIDDEN_DW=1`, the KHR gradient pass uses the new tile-streaming helper for hidden layer 4 `dW`; the default path remains the subgroup-arithmetic helper.
- Added shader compile tests for the full KHR gradient pass with `NRC_KHR_USE_COOPMAT_FINAL_HIDDEN_DW=1`:
  - SG32;
  - SG16;
  - M8/SG32.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/khr_shared_memory_layout.md`
  - `docs/training_optimizer.md`

### Behavior

The new helper implements the tile-streaming plan from Milestone 66:

```text
for each hidden row block:
  stage fp16 dPre for that row block in the active tile
  for each source column block:
    accumulate one MxN dW tile per subgroup with coopMatMulAdd
    store one fp16 partial tile per subgroup
    reduce subgroup partials into the fp32 row-major gradient buffer
```

The helper uses:

```text
A: dPre row block, MxK, column-major from compact active-tile staging
B: source activations, KxN, row-major from saved activation snapshots
C: dW tile, MxN, fp16 cooperative-matrix accumulator
```

This is the first hidden-layer cooperative-matrix `dW` implementation surface. It is deliberately limited to an experimental final-hidden switch in the full gradient pass. Earlier hidden layers still use the existing subgroup path.

### Validation

- Reconfigured CMake so the new shader compile tests are registered:
  - `cmake -S . -B build`
- Targeted shader compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr_coopmat_final_hidden|nrc_shader_khr_backprop" --output-on-failure -V`
- Targeted test result:
  - `6/6 tests passed`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Full test result:
  - `40/40 tests passed`

### Current Limitations

- The helper is compile-validated only. It is not yet numerically validated on a driver that can create the KHR gradient pipeline.
- No CPU oracle has been added yet for final-hidden coopmat `dW` fp16 staging and partial storage.
- The default KHR gradient path still uses subgroup arithmetic for final-hidden `dW`.
- Hidden layers 3, 2, 1, and 0 still use the subgroup `dW` path.
- KHR `dA` propagation still uses scalar arrays rather than cooperative matrices.

### Next Step

Add a CPU oracle path for the final-hidden cooperative-matrix `dW` helper, then add a runtime validation mode similar to `--coopmat-output` once the oracle is available. After that, extend the same tile-streaming helper to earlier hidden layers one layer at a time while keeping the default path stable until numerical validation succeeds.

## 2026-07-09 - Milestone 68 Final-Hidden Coopmat dW Runtime Oracle

### Implemented

- Added a test-side CPU oracle for the optional final-hidden cooperative-matrix `dW` path in `tests/vulkan_gradient_khr_test.cpp`.
- The oracle starts from the current KHR groundwork gradients and replaces hidden layer 4 `dW` with the tile-streaming coopmat expectation:
  - propagate `dOutput` to hidden layer 4 in fp32;
  - apply the hidden activation ReLU gate;
  - stage each row-block `dPre` value through fp16 rounding;
  - accumulate against the saved fp16 source activation for hidden layer 3;
  - round the final hidden-layer partials back through fp16 before writing the expected fp32 gradient buffer.
- Added expected-gradient sanity checks before Vulkan device selection:
  - all expected gradients must be finite;
  - output-coopmat expected output gradients must be non-zero;
  - final-hidden-coopmat expected hidden layer 4 gradients must be non-zero.
- Added `--coopmat-final-hidden` to `nrc_vulkan_gradient_khr_test`.
- Added runtime SPIR-V generation for `NRC_KHR_USE_COOPMAT_FINAL_HIDDEN_DW=1`:
  - SG32;
  - SG16;
  - M8/SG32.
- Added `nrc_gradient_khr_coopmat_final_hidden_dispatch_shaders`.
- Added `nrc_vulkan_gradient_khr_coopmat_final_hidden_test`.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/vulkan_dispatch_harness.md`
  - `docs/training_optimizer.md`
  - `docs/khr_shared_memory_layout.md`

### Behavior

The KHR gradient runtime test now has three modes:

```text
default:                 KHR groundwork oracle
--coopmat-output:        output-layer fp16 coopmat dW oracle
--coopmat-final-hidden:  hidden layer 4 fp16 coopmat dW oracle
```

The final-hidden oracle only changes hidden layer 4 `dW`. Other hidden layers and the output layer remain compared against the existing KHR groundwork oracle.

### Validation

- Reconfigured CMake so the new runtime SPIR-V target and test are registered:
  - `cmake -S . -B build`
- Targeted KHR gradient runtime test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr" --output-on-failure -V`
- Targeted test result:
  - `3/3 tests passed`
- Current local KHR training skip result for all three runtime modes:
  - `SKIP: no Vulkan device supports an implemented NRC KHR cooperative matrix training profile`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Full test result:
  - `41/41 tests passed`

### Current Limitations

- The final-hidden coopmat runtime validation path is wired, but the current local device does not satisfy the 98304-byte KHR training shared-memory profile gate. The numerical comparison therefore has not executed locally.
- The default KHR gradient path still uses subgroup arithmetic for final-hidden `dW`.
- Hidden layers 3, 2, 1, and 0 still use the subgroup `dW` path.
- KHR `dA` propagation still uses scalar arrays rather than cooperative matrices.
- The test-side final-hidden oracle currently mirrors the single-workgroup test batch. Broader batch/workgroup coverage should be added before making this helper a default path.

### Next Step

Extend the tile-streaming helper to an earlier hidden layer behind another explicit experimental switch, or broaden the final-hidden oracle/test batch coverage so multiple subgroups contribute non-zero partials before promoting the helper further.

## 2026-07-09 - Milestone 69 Multi-Subgroup KHR Coopmat Gradient Oracle Coverage

### Implemented

- Increased the KHR gradient runtime test batch from 3 train records to 67 train records.
- Kept the batch within one 128-wide KHR workgroup while making it span:
  - at least three SG32 subgroup sample ranges;
  - at least five SG16 subgroup sample ranges.
- Reworked the generated train records so packed barycentric and scattered-direction inputs stay inside the unorm range without saturating.
- Updated the output-coopmat CPU oracle to:
  - bucket samples by the selected subgroup size;
  - accumulate one output-layer partial per subgroup;
  - fp16-round each subgroup partial before summing subgroup partials into the expected fp32 gradient.
- Updated the final-hidden-coopmat CPU oracle to:
  - bucket samples by the selected subgroup size;
  - accumulate one hidden layer 4 partial per subgroup;
  - fp16-round each subgroup partial before summing subgroup partials into the expected fp32 gradient.
- Added SG16 and SG32 expected-gradient sanity generation before Vulkan profile selection, so oracle coverage is still checked on the local machine even when KHR training profile selection skips.
- Added `VulkanKhrGradientRunner::selected_subgroup_size()` so runtime comparison chooses the correct oracle for the selected profile.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/vulkan_dispatch_harness.md`

### Behavior

The optional coopmat `dW` oracles now mirror the shader's cross-subgroup reduction shape:

```text
for each subgroup:
  accumulate samples owned by that subgroup
  fp16-round the subgroup partial tile

expected gradient:
  fp32 sum of all rounded subgroup partials
```

This replaces the earlier small-batch oracle shape where only subgroup 0 had non-zero samples and a single fp16 rounding step was enough.

### Validation

- Targeted KHR gradient runtime test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr" --output-on-failure -V`
- Targeted test result:
  - `3/3 tests passed`
- Current local KHR training skip result for all three runtime modes remains:
  - `SKIP: no Vulkan device supports an implemented NRC KHR cooperative matrix training profile`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Full test result:
  - `41/41 tests passed`

### Current Limitations

- The KHR runtime numerical comparison still has not executed locally because the current device does not satisfy the 98304-byte KHR training shared-memory profile gate.
- The widened oracle still models one 128-wide workgroup. Multi-workgroup KHR gradient validation remains future work.
- The default KHR gradient path still uses subgroup arithmetic for output and final-hidden `dW`.
- Hidden layers 3, 2, 1, and 0 still use the subgroup `dW` path.

### Next Step

Either extend the tile-streaming helper to hidden layer 3 behind an explicit experimental switch or add multi-workgroup oracle coverage before broadening the coopmat path further.

## 2026-07-09 - Milestone 70 Hidden-Layer-3 Coopmat dW Runtime Oracle

### Implemented

- Added `NRC_KHR_USE_COOPMAT_HIDDEN3_DW=1` to `shaders/nrc/passes/NrcGradientKhr.glsl`.
- The new switch replaces only hidden layer 3 `dW` (`NRC_NN_HIDDEN_LAYER_COUNT - 2`) with `NrcKhrAccumulateHiddenDWeightsCoopmatTileStreaming`.
- Kept the default hidden-layer-3 path on the existing subgroup-arithmetic helper.
- Preserved downstream backpropagation by continuing to propagate `dHidden3` to hidden layer 2 from the local derivative array.
- Extended `tests/shader_compile/nrc_khr_backprop_smoke.comp` so the backend smoke translation unit also type-checks the hidden-layer-3 tile-streaming call shape.
- Added shader compile tests for `NRC_KHR_USE_COOPMAT_HIDDEN3_DW=1`:
  - SG32;
  - SG16;
  - M8/SG32.
- Added a test-side CPU oracle for hidden-layer-3 coopmat `dW` in `tests/vulkan_gradient_khr_test.cpp`.
- The oracle starts from current KHR groundwork gradients and replaces only hidden layer 3 `dW`:
  - propagate `dOutput` to hidden layer 4 in fp32;
  - propagate hidden layer 4 through its ReLU gate and weights to `dHidden3`;
  - apply the hidden-layer-3 ReLU gate;
  - fp16-round staged hidden-layer-3 `dPre`;
  - accumulate against saved hidden-layer-2 source activations;
  - fp16-round each subgroup partial before the final fp32 subgroup sum.
- Added `--coopmat-hidden3` to `nrc_vulkan_gradient_khr_test`.
- Added runtime SPIR-V generation for `NRC_KHR_USE_COOPMAT_HIDDEN3_DW=1`:
  - SG32;
  - SG16;
  - M8/SG32.
- Added `nrc_gradient_khr_coopmat_hidden3_dispatch_shaders`.
- Added `nrc_vulkan_gradient_khr_coopmat_hidden3_test`.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/vulkan_dispatch_harness.md`

### Behavior

The KHR gradient runtime test now has four modes:

```text
default:                 KHR groundwork oracle
--coopmat-output:        output-layer fp16 coopmat dW oracle
--coopmat-final-hidden:  hidden layer 4 fp16 coopmat dW oracle
--coopmat-hidden3:       hidden layer 3 fp16 coopmat dW oracle
```

The hidden-layer-3 oracle only changes hidden layer 3 `dW`. Other hidden layers and the output layer remain compared against the existing KHR groundwork oracle.

### Validation

- Targeted KHR gradient runtime test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr" --output-on-failure -V`
- Targeted KHR gradient runtime result:
  - `4/4 tests passed`
- Current local KHR training skip result for all four runtime modes remains:
  - `SKIP: no Vulkan device supports an implemented NRC KHR cooperative matrix training profile`
- Targeted hidden-layer-3 shader compile and KHR backprop smoke tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr_coopmat_hidden3|nrc_shader_khr_backprop" --output-on-failure -V`
- Targeted shader compile result:
  - `6/6 tests passed`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Full test result:
  - `45/45 tests passed`

### Current Limitations

- The hidden-layer-3 coopmat runtime validation path is wired, but the current local device does not satisfy the 98304-byte KHR training shared-memory profile gate. The numerical comparison therefore has not executed locally.
- The default KHR gradient path still uses subgroup arithmetic for output, final-hidden, and hidden-layer-3 `dW`.
- Hidden layers 2, 1, and 0 still use the subgroup `dW` path.
- KHR `dA` propagation still uses scalar arrays rather than cooperative matrices.
- The test-side coopmat oracles still model one 128-wide workgroup. Multi-workgroup KHR gradient validation remains future work.

### Next Step

Extend the same tile-streaming helper to hidden layer 2 behind a new explicit experimental switch, or add multi-workgroup oracle coverage before broadening the coopmat path further.

## 2026-07-09 - Milestone 71 Hidden-Layer-2 Coopmat dW Runtime Oracle

### Implemented

- Added `NRC_KHR_USE_COOPMAT_HIDDEN2_DW=1` to `shaders/nrc/passes/NrcGradientKhr.glsl`.
- The new switch replaces only hidden layer 2 `dW` (`NRC_NN_HIDDEN_LAYER_COUNT - 3`) with `NrcKhrAccumulateHiddenDWeightsCoopmatTileStreaming`.
- Kept the default hidden-layer-2 path on the existing subgroup-arithmetic helper.
- Preserved downstream backpropagation by continuing to propagate `dHidden2` to hidden layer 1 from the local derivative array.
- Extended `tests/shader_compile/nrc_khr_backprop_smoke.comp` so the backend smoke translation unit type-checks the hidden-layer-2 tile-streaming call shape.
- Added shader compile tests for `NRC_KHR_USE_COOPMAT_HIDDEN2_DW=1`:
  - SG32;
  - SG16;
  - M8/SG32.
- Added a test-side CPU oracle for hidden-layer-2 coopmat `dW` in `tests/vulkan_gradient_khr_test.cpp`.
- The oracle starts from current KHR groundwork gradients and replaces only hidden layer 2 `dW`:
  - propagate `dOutput` to hidden layer 4 in fp32;
  - propagate hidden layer 4 through its ReLU gate and weights to `dHidden3`;
  - propagate hidden layer 3 through its ReLU gate and weights to `dHidden2`;
  - apply the hidden-layer-2 ReLU gate;
  - fp16-round staged hidden-layer-2 `dPre`;
  - accumulate against saved hidden-layer-1 source activations;
  - fp16-round each subgroup partial before the final fp32 subgroup sum.
- Added `--coopmat-hidden2` to `nrc_vulkan_gradient_khr_test`.
- Added runtime SPIR-V generation for `NRC_KHR_USE_COOPMAT_HIDDEN2_DW=1`:
  - SG32;
  - SG16;
  - M8/SG32.
- Added `nrc_gradient_khr_coopmat_hidden2_dispatch_shaders`.
- Added `nrc_vulkan_gradient_khr_coopmat_hidden2_test`.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/vulkan_dispatch_harness.md`

### Behavior

The KHR gradient runtime test now has five modes:

```text
default:                 KHR groundwork oracle
--coopmat-output:        output-layer fp16 coopmat dW oracle
--coopmat-final-hidden:  hidden layer 4 fp16 coopmat dW oracle
--coopmat-hidden3:       hidden layer 3 fp16 coopmat dW oracle
--coopmat-hidden2:       hidden layer 2 fp16 coopmat dW oracle
```

The hidden-layer-2 oracle only changes hidden layer 2 `dW`. Other hidden layers and the output layer remain compared against the existing KHR groundwork oracle.

### Validation

- Targeted KHR gradient runtime test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr" --output-on-failure -V`
- Targeted KHR gradient runtime result:
  - `5/5 tests passed`
- Current local KHR training skip result for all five runtime modes remains:
  - `SKIP: no Vulkan device supports an implemented NRC KHR cooperative matrix training profile`
- Targeted hidden-layer-2 shader compile and KHR backprop smoke tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr_coopmat_hidden2|nrc_shader_khr_backprop" --output-on-failure -V`
- Targeted shader compile result:
  - `6/6 tests passed`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Full test result:
  - `49/49 tests passed`

### Current Limitations

- The hidden-layer-2 coopmat runtime validation path is wired, but the current local device does not satisfy the 98304-byte KHR training shared-memory profile gate. The numerical comparison therefore has not executed locally.
- The default KHR gradient path still uses subgroup arithmetic for output, final-hidden, hidden-layer-3, and hidden-layer-2 `dW`.
- Hidden layers 1 and 0 still use the subgroup `dW` path.
- KHR `dA` propagation still uses scalar arrays rather than cooperative matrices.
- The test-side coopmat oracles still model one 128-wide workgroup. Multi-workgroup KHR gradient validation remains future work.

### Next Step

Extend the same tile-streaming helper to hidden layer 1 behind a new explicit experimental switch, or add multi-workgroup oracle coverage before broadening the coopmat path further.

## 2026-07-09 - Milestone 72 Hidden-Layer-1 Coopmat dW Runtime Oracle

### Implemented

- Added `NRC_KHR_USE_COOPMAT_HIDDEN1_DW=1` to `shaders/nrc/passes/NrcGradientKhr.glsl`.
- The new switch replaces only hidden layer 1 `dW` (`NRC_NN_HIDDEN_LAYER_COUNT - 4`) with `NrcKhrAccumulateHiddenDWeightsCoopmatTileStreaming`.
- Kept the default hidden-layer-1 path on the existing subgroup-arithmetic helper.
- Preserved downstream backpropagation by continuing to propagate `dHidden1` to hidden layer 0 from the local derivative array.
- Extended `tests/shader_compile/nrc_khr_backprop_smoke.comp` so the backend smoke translation unit type-checks the hidden-layer-1 tile-streaming call shape.
- Added shader compile tests for `NRC_KHR_USE_COOPMAT_HIDDEN1_DW=1`:
  - SG32;
  - SG16;
  - M8/SG32.
- Added a test-side CPU oracle for hidden-layer-1 coopmat `dW` in `tests/vulkan_gradient_khr_test.cpp`.
- The oracle starts from current KHR groundwork gradients and replaces only hidden layer 1 `dW`:
  - propagate `dOutput` to hidden layer 4 in fp32;
  - propagate hidden layer 4 through its ReLU gate and weights to `dHidden3`;
  - propagate hidden layer 3 through its ReLU gate and weights to `dHidden2`;
  - propagate hidden layer 2 through its ReLU gate and weights to `dHidden1`;
  - apply the hidden-layer-1 ReLU gate;
  - fp16-round staged hidden-layer-1 `dPre`;
  - accumulate against saved hidden-layer-0 source activations;
  - fp16-round each subgroup partial before the final fp32 subgroup sum.
- Added `--coopmat-hidden1` to `nrc_vulkan_gradient_khr_test`.
- Added runtime SPIR-V generation for `NRC_KHR_USE_COOPMAT_HIDDEN1_DW=1`:
  - SG32;
  - SG16;
  - M8/SG32.
- Added `nrc_gradient_khr_coopmat_hidden1_dispatch_shaders`.
- Added `nrc_vulkan_gradient_khr_coopmat_hidden1_test`.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/vulkan_dispatch_harness.md`

### Behavior

The KHR gradient runtime test now has six modes:

```text
default:                 KHR groundwork oracle
--coopmat-output:        output-layer fp16 coopmat dW oracle
--coopmat-final-hidden:  hidden layer 4 fp16 coopmat dW oracle
--coopmat-hidden3:       hidden layer 3 fp16 coopmat dW oracle
--coopmat-hidden2:       hidden layer 2 fp16 coopmat dW oracle
--coopmat-hidden1:       hidden layer 1 fp16 coopmat dW oracle
```

The hidden-layer-1 oracle only changes hidden layer 1 `dW`. Other hidden layers and the output layer remain compared against the existing KHR groundwork oracle.

### Validation

- Targeted KHR gradient runtime test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr" --output-on-failure -V`
- Targeted KHR gradient runtime result:
  - `6/6 tests passed`
- Current local KHR training skip result for all six runtime modes remains:
  - `SKIP: no Vulkan device supports an implemented NRC KHR cooperative matrix training profile`
- Targeted hidden-layer-1 shader compile and KHR backprop smoke tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr_coopmat_hidden1|nrc_shader_khr_backprop" --output-on-failure -V`
- Targeted shader compile result:
  - `6/6 tests passed`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Full test result:
  - `53/53 tests passed`

### Current Limitations

- The hidden-layer-1 coopmat runtime validation path is wired, but the current local device does not satisfy the 98304-byte KHR training shared-memory profile gate. The numerical comparison therefore has not executed locally.
- The default KHR gradient path still uses subgroup arithmetic for output, final-hidden, hidden-layer-3, hidden-layer-2, and hidden-layer-1 `dW`.
- Hidden layer 0 still uses the subgroup `dW` path.
- Hidden layer 0 is not a direct use of the current hidden tile-streaming helper because its source activations are the encoded input. The current subgroup path stages encoded input into the oldest source snapshot after using that snapshot as the hidden-layer-0 ReLU mask, so a coopmat version needs a separate source-staging plan that preserves the hidden-layer-0 activation mask.
- KHR `dA` propagation still uses scalar arrays rather than cooperative matrices.
- The test-side coopmat oracles still model one 128-wide workgroup. Multi-workgroup KHR gradient validation remains future work.

### Next Step

Design the hidden-layer-0 cooperative-matrix `dW` path with separate encoded-input source staging, or add multi-workgroup oracle coverage before broadening the coopmat path further.

## 2026-07-09 - Milestone 73 Hidden-Layer-0 Coopmat dW Runtime Oracle

### Implemented

- Added `NRC_KHR_USE_COOPMAT_HIDDEN0_DW=1` to `shaders/nrc/passes/NrcGradientKhr.glsl`.
- The new switch replaces only hidden layer 0 `dW` with `NrcKhrAccumulateHiddenDWeightsCoopmatTileStreaming`.
- Kept the default hidden-layer-0 path on the existing subgroup-arithmetic helper.
- Added hidden-layer-0 encoded-source staging for the coopmat path:
  - preserves `NRC_KHR_BACKPROP_OLDEST_SOURCE_ACT_COPY_UVEC4_BASE` as the hidden-layer-0 activation/ReLU-mask snapshot;
  - stages encoded input into `NRC_KHR_BACKPROP_ACT_COPY_UVEC4_BASE`, which is no longer needed by this point in backpropagation;
  - reuses the existing 98304-byte shared-memory layout without adding another activation-sized snapshot.
- Extended `tests/shader_compile/nrc_khr_backprop_smoke.comp` so the backend smoke translation unit type-checks the hidden-layer-0 tile-streaming call shape.
- Added shader compile tests for `NRC_KHR_USE_COOPMAT_HIDDEN0_DW=1`:
  - SG32;
  - SG16;
  - M8/SG32.
- Added a test-side CPU oracle for hidden-layer-0 coopmat `dW` in `tests/vulkan_gradient_khr_test.cpp`.
- The oracle starts from current KHR groundwork gradients and replaces only hidden layer 0 `dW`:
  - propagate `dOutput` to hidden layer 4 in fp32;
  - propagate hidden layer 4 through its ReLU gate and weights to `dHidden3`;
  - propagate hidden layer 3 through its ReLU gate and weights to `dHidden2`;
  - propagate hidden layer 2 through its ReLU gate and weights to `dHidden1`;
  - propagate hidden layer 1 through its ReLU gate and weights to `dHidden0`;
  - apply the hidden-layer-0 ReLU gate;
  - fp16-round staged hidden-layer-0 `dPre`;
  - accumulate against encoded input source activations;
  - fp16-round each subgroup partial before the final fp32 subgroup sum.
- Added `--coopmat-hidden0` to `nrc_vulkan_gradient_khr_test`.
- Added runtime SPIR-V generation for `NRC_KHR_USE_COOPMAT_HIDDEN0_DW=1`:
  - SG32;
  - SG16;
  - M8/SG32.
- Added `nrc_gradient_khr_coopmat_hidden0_dispatch_shaders`.
- Added `nrc_vulkan_gradient_khr_coopmat_hidden0_test`.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/vulkan_dispatch_harness.md`

### Behavior

The KHR gradient runtime test now has seven modes:

```text
default:                 KHR groundwork oracle
--coopmat-output:        output-layer fp16 coopmat dW oracle
--coopmat-final-hidden:  hidden layer 4 fp16 coopmat dW oracle
--coopmat-hidden3:       hidden layer 3 fp16 coopmat dW oracle
--coopmat-hidden2:       hidden layer 2 fp16 coopmat dW oracle
--coopmat-hidden1:       hidden layer 1 fp16 coopmat dW oracle
--coopmat-hidden0:       hidden layer 0 fp16 coopmat dW oracle
```

Every hidden layer now has a single-region experimental coopmat `dW` mode. Each mode still replaces exactly one gradient region; combined-region coopmat variants remain future work.

### Validation

- Targeted KHR gradient runtime test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr" --output-on-failure -V`
- Targeted KHR gradient runtime result:
  - `7/7 tests passed`
- Current local KHR training skip result for all seven runtime modes remains:
  - `SKIP: no Vulkan device supports an implemented NRC KHR cooperative matrix training profile`
- Targeted hidden-layer-0 shader compile and KHR backprop smoke tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr_coopmat_hidden0|nrc_shader_khr_backprop" --output-on-failure -V`
- Targeted shader compile result:
  - `6/6 tests passed`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Full test result:
  - `57/57 tests passed`

### Current Limitations

- The hidden-layer-0 coopmat runtime validation path is wired, but the current local device does not satisfy the 98304-byte KHR training shared-memory profile gate. The numerical comparison therefore has not executed locally.
- The default KHR gradient path still uses subgroup arithmetic for output and every hidden-layer `dW`.
- The optional coopmat `dW` paths are validated one region at a time; combined-region shader variants and oracles are still future work.
- KHR `dA` propagation still uses scalar arrays rather than cooperative matrices.
- The test-side coopmat oracles still model one 128-wide workgroup. Multi-workgroup KHR gradient validation remains future work.

### Next Step

Add combined KHR coopmat `dW` variants so multiple regions can use the tile-streaming path in one shader, or add multi-workgroup oracle coverage before promoting these helpers toward the default path.

## 2026-07-09 - Milestone 74 Combined All-dW Coopmat Gradient Mode

### Implemented

- Added `NRC_KHR_USE_COOPMAT_ALL_DW=1` to `shaders/nrc/passes/NrcGradientKhr.glsl`.
- The new aggregate switch enables:
  - output-layer coopmat `dW`;
  - final-hidden coopmat `dW`;
  - hidden-layer-3 coopmat `dW`;
  - hidden-layer-2 coopmat `dW`;
  - hidden-layer-1 coopmat `dW`;
  - hidden-layer-0 coopmat `dW`.
- Kept individually defined region switches authoritative, so targeted A/B variants can still override or use one region at a time.
- Added a test-side CPU oracle for the all-`dW` mode in `tests/vulkan_gradient_khr_test.cpp`.
- The combined oracle starts from the output-coopmat expectation and copies each hidden-layer gradient region from the matching hidden-layer coopmat oracle. This mirrors the shader contract that each `dW` helper writes a disjoint gradient-buffer region.
- Added `--coopmat-all-dw` to `nrc_vulkan_gradient_khr_test`.
- Added expected-gradient sanity checks for all-`dW`:
  - all expected gradients must be finite;
  - output-layer coopmat gradients must be non-zero;
  - every hidden-layer coopmat gradient region must be non-zero for SG16 and SG32 oracle shapes.
- Added shader compile tests for `NRC_KHR_USE_COOPMAT_ALL_DW=1`:
  - SG32;
  - SG16;
  - M8/SG32.
- Added runtime SPIR-V generation for `NRC_KHR_USE_COOPMAT_ALL_DW=1`:
  - SG32;
  - SG16;
  - M8/SG32.
- Added `nrc_gradient_khr_coopmat_all_dw_dispatch_shaders`.
- Added `nrc_vulkan_gradient_khr_coopmat_all_dw_test`.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/vulkan_dispatch_harness.md`
  - `docs/training_optimizer.md`
  - `docs/khr_shared_memory_layout.md`

### Behavior

The KHR gradient runtime test now has eight modes:

```text
default:                 KHR groundwork oracle
--coopmat-output:        output-layer fp16 coopmat dW oracle
--coopmat-final-hidden:  hidden layer 4 fp16 coopmat dW oracle
--coopmat-hidden3:       hidden layer 3 fp16 coopmat dW oracle
--coopmat-hidden2:       hidden layer 2 fp16 coopmat dW oracle
--coopmat-hidden1:       hidden layer 1 fp16 coopmat dW oracle
--coopmat-hidden0:       hidden layer 0 fp16 coopmat dW oracle
--coopmat-all-dw:        output and all hidden dW regions use fp16 coopmat oracles
```

`NRC_KHR_USE_COOPMAT_ALL_DW=1` is the first full shader variant where all currently implemented output/hidden `dW` regions use the coopmat paths together.

### Validation

- Targeted KHR gradient runtime test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr" --output-on-failure -V`
- Targeted KHR gradient runtime result:
  - `8/8 tests passed`
- Current local KHR training skip result for all eight runtime modes remains:
  - `SKIP: no Vulkan device supports an implemented NRC KHR cooperative matrix training profile`
- Targeted all-`dW` shader compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr_coopmat_all_dw" --output-on-failure -V`
- Targeted shader compile result:
  - `3/3 tests passed`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Full test result:
  - `61/61 tests passed`

### Current Limitations

- The all-`dW` coopmat runtime validation path is wired, but the current local device does not satisfy the 98304-byte KHR training shared-memory profile gate. The numerical comparison therefore has not executed locally.
- The default KHR gradient path still uses subgroup arithmetic for output and every hidden-layer `dW`.
- The all-`dW` oracle still models one 128-wide workgroup. Multi-workgroup KHR gradient validation remains future work.
- KHR `dA` propagation still uses scalar arrays rather than cooperative matrices.
- The all-`dW` mode still uses one fp32 atomic add per output/hidden weight per workgroup after reducing coopmat subgroup partials.

### Next Step

Add multi-workgroup KHR gradient oracle coverage, or start porting KHR `dA` propagation toward cooperative-matrix form so the training path moves closer to VkNRC's fully fused shape.

## 2026-07-09 - Milestone 75 Multi-Workgroup KHR Coopmat Gradient Oracle

### Implemented

- Increased `tests/vulkan_gradient_khr_test.cpp` from a 67-record single-workgroup KHR gradient batch to a 197-record batch.
- The new batch shape spans:
  - one full 128-wide workgroup;
  - one partial final workgroup;
  - multiple SG32 subgroup ranges in both workgroups;
  - multiple SG16 subgroup ranges in both workgroups.
- Added static coverage guards so the KHR gradient test batch must:
  - exceed one KHR gradient workgroup;
  - include a partial final workgroup;
  - fit inside one NRC train batch.
- Added shared CPU-oracle helpers for:
  - KHR coopmat subgroup-count validation;
  - sample-to-workgroup mapping;
  - sample-to-local-subgroup mapping;
  - storing `[workgroup][subgroup][weight]` fp16-rounded partials back into the expected gradient region.
- Updated every optional KHR coopmat `dW` oracle to model per-workgroup subgroup partials:
  - output layer;
  - final hidden layer;
  - hidden layer 3;
  - hidden layer 2;
  - hidden layer 1;
  - hidden layer 0;
  - all-`dW`.
- The oracle now mirrors the shader's workgroup-local reduction shape more closely:
  - accumulate one fp32 partial per local subgroup;
  - fp16-round each subgroup partial;
  - sum subgroup partials inside the workgroup;
  - accumulate each workgroup's contribution into the global fp32 gradient region.
- Updated:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/vulkan_dispatch_harness.md`

### Behavior

The KHR gradient runtime test still has eight modes:

```text
default:                 KHR groundwork oracle
--coopmat-output:        output-layer fp16 coopmat dW oracle
--coopmat-final-hidden:  hidden layer 4 fp16 coopmat dW oracle
--coopmat-hidden3:       hidden layer 3 fp16 coopmat dW oracle
--coopmat-hidden2:       hidden layer 2 fp16 coopmat dW oracle
--coopmat-hidden1:       hidden layer 1 fp16 coopmat dW oracle
--coopmat-hidden0:       hidden layer 0 fp16 coopmat dW oracle
--coopmat-all-dw:        output and all hidden dW regions use fp16 coopmat oracles
```

The shader dispatch path already computed group count from `trainCount` and the selected profile workgroup size. This milestone changes the expected-value model so the runtime comparison is ready for more than one workgroup when a local device can create and run the KHR training pipeline.

### Validation

- Diff whitespace check succeeded:
  - `git diff --check`
- Targeted KHR gradient runtime test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr" --output-on-failure -V`
- Targeted KHR gradient runtime result:
  - `8/8 tests passed`
- Current local KHR training skip result for all eight runtime modes remains:
  - `SKIP: no Vulkan device supports an implemented NRC KHR cooperative matrix training profile`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Full test result:
  - `61/61 tests passed`

### Current Limitations

- The multi-workgroup KHR coopmat runtime comparison is wired, but the current local device still does not satisfy the 98304-byte KHR training shared-memory profile gate. The numerical GPU-vs-CPU comparison therefore has not executed locally.
- The default KHR gradient path still uses subgroup arithmetic for output and every hidden-layer `dW`.
- KHR `dA` propagation still uses scalar arrays rather than cooperative matrices.
- The coopmat `dW` modes still use one fp32 atomic add per output/hidden weight per workgroup after reducing fp16 subgroup partials.
- The CPU oracle code is intentionally explicit but now has repeated hidden-layer backpropagation blocks. It is a candidate for a future helper refactor once the shader behavior stabilizes.

### Next Step

Refactor the duplicated KHR coopmat oracle backpropagation code into shared helpers, or start porting KHR `dA` propagation toward cooperative-matrix form so the training path moves closer to VkNRC's fully fused shape.

## 2026-07-09 - Milestone 76 Shared KHR Coopmat Oracle Backprop State

### Implemented

- Refactored `tests/vulkan_gradient_khr_test.cpp` so the optional KHR coopmat `dW` oracles share one per-sample backpropagation state.
- Added `KhrBackpropSample`, which stores:
  - encoded NRC input;
  - fp16-rounded hidden activations for every hidden layer;
  - output derivative;
  - hidden-layer derivatives for every hidden layer.
- Added shared helper functions for:
  - propagating output derivatives into the final hidden layer;
  - propagating a hidden derivative through a hidden layer's ReLU gate and weights;
  - constructing a complete per-sample KHR backprop state;
  - reading the correct source activation for any hidden layer, including encoded input for hidden layer 0.
- Replaced the five duplicated hidden-layer coopmat oracle bodies with one layer-parameterized helper:
  - final hidden layer;
  - hidden layer 3;
  - hidden layer 2;
  - hidden layer 1;
  - hidden layer 0.
- Kept the existing public test helper names as thin wrappers so the runtime modes and all-`dW` composition did not need to change.
- Updated the output-layer coopmat oracle to reuse the same per-sample backprop state for fp16 forward activations and `dOutput`.
- Updated current documentation:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
- Fixed stale current documentation in `docs/khr_coopmat_backend.md` that still described the runtime KHR gradient batch as a 67-record single-workgroup batch. The current batch is 197 records and spans two 128-wide workgroups with a partial final workgroup.

### Behavior

This milestone is intended to preserve numerical behavior. It changes the expected-value implementation shape, not the shader contract:

```text
default:                 KHR groundwork oracle
--coopmat-output:        output-layer fp16 coopmat dW oracle
--coopmat-final-hidden:  hidden layer 4 fp16 coopmat dW oracle
--coopmat-hidden3:       hidden layer 3 fp16 coopmat dW oracle
--coopmat-hidden2:       hidden layer 2 fp16 coopmat dW oracle
--coopmat-hidden1:       hidden layer 1 fp16 coopmat dW oracle
--coopmat-hidden0:       hidden layer 0 fp16 coopmat dW oracle
--coopmat-all-dw:        output and all hidden dW regions use fp16 coopmat oracles
```

The hidden-layer coopmat oracle now has one derivative chain to audit before adding cooperative-matrix `dA` variants. That should make the next shader/oracle expansion much less brittle than updating five separate hidden-layer backpropagation copies.

### Validation

- Diff whitespace check succeeded:
  - `git diff --check`
- Targeted KHR gradient runtime test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr" --output-on-failure -V`
- Targeted KHR gradient runtime result:
  - `8/8 tests passed`
- Current local KHR training skip result for all eight runtime modes remains:
  - `SKIP: no Vulkan device supports an implemented NRC KHR cooperative matrix training profile`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Full test result:
  - `61/61 tests passed`

### Current Limitations

- This is a test-oracle refactor and does not add cooperative-matrix `dA` shader execution yet.
- The multi-workgroup KHR coopmat runtime comparison is still wired but not numerically executed locally because the current device does not satisfy the 98304-byte KHR training shared-memory profile gate.
- The default KHR gradient path still uses subgroup arithmetic for output and every hidden-layer `dW`.
- KHR `dA` propagation still uses scalar arrays rather than cooperative matrices.
- The coopmat `dW` modes still use one fp32 atomic add per output/hidden weight per workgroup after reducing fp16 subgroup partials.

### Next Step

Start the KHR cooperative-matrix `dA` port with a hidden-to-hidden derivative propagation helper and matching CPU oracle mode, using the shared backprop sample state as the expected-value foundation.

## 2026-07-09 - Milestone 77 KHR Final-Hidden Coopmat dA Mode

### Implemented

- Added the first optional KHR cooperative-matrix `dA` propagation path for the final hidden layer into hidden layer 3.
- Added `NRC_KHR_USE_COOPMAT_FINAL_HIDDEN_DA`, defaulting to disabled, so the baseline KHR gradient path remains unchanged unless the test/profile explicitly enables this mode.
- Added KHR backend helper code in `NrcNNKhr.glsl` to:
  - stage final-hidden `dPre` values as fp16 in the active tile;
  - load hidden-layer weights through a cooperative-matrix transpose shape;
  - compute the hidden-3 derivative contribution with cooperative-matrix multiply-add;
  - store the fp16 staged hidden-3 derivative back into the existing activation snapshot area before the scalar gradient code reads it.
- Added the `--coopmat-final-hidden-da` runtime test mode and a matching CPU expected-value path that models the fp16 staged `dPre` and fp16 stored hidden-3 derivative.
- Added compile coverage for the final-hidden `dA` mode across the current KHR shader profiles:
  - subgroup 32;
  - subgroup 16;
  - `m8` subgroup 32.
- Updated current documentation:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/khr_shared_memory_layout.md`
  - `docs/training_optimizer.md`

### Behavior

This milestone introduces a narrow cooperative-matrix `dA` option while keeping the existing `dW` modes intact:

```text
default:                       KHR groundwork oracle
--coopmat-output:              output-layer fp16 coopmat dW oracle
--coopmat-final-hidden:        hidden layer 4 fp16 coopmat dW oracle
--coopmat-final-hidden-da:     final-hidden -> hidden3 fp16 coopmat dA oracle
--coopmat-hidden3:             hidden layer 3 fp16 coopmat dW oracle
--coopmat-hidden2:             hidden layer 2 fp16 coopmat dW oracle
--coopmat-hidden1:             hidden layer 1 fp16 coopmat dW oracle
--coopmat-hidden0:             hidden layer 0 fp16 coopmat dW oracle
--coopmat-all-dw:              output and all hidden dW regions use fp16 coopmat oracles
```

The new mode is intentionally separate from `--coopmat-all-dw`. That keeps the first `dA` port independently auditable before chaining cooperative-matrix derivative propagation through the remaining hidden layers.

### Validation

- Diff whitespace check succeeded:
  - `git diff --check`
- Manual shader compilation succeeded for the three final-hidden `dA` profiles:
  - subgroup 32
  - subgroup 16
  - `m8` subgroup 32
- Targeted KHR gradient runtime test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr" --output-on-failure -V`
- Targeted KHR gradient runtime result:
  - `9/9 tests passed`
- Targeted final-hidden `dA` compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr_coopmat_final_hidden_da" --output-on-failure -V`
- Targeted final-hidden `dA` compile result:
  - `3/3 tests passed`
- Current local KHR training skip result for all nine runtime modes remains:
  - `SKIP: no Vulkan device supports an implemented NRC KHR cooperative matrix training profile`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Full test result:
  - `65/65 tests passed`

### Current Limitations

- Only the final-hidden-to-hidden3 `dA` propagation path has an optional cooperative-matrix implementation.
- The default KHR gradient path still uses scalar `dA` propagation.
- The remaining hidden derivative chain, hidden3-to-hidden2 through hidden1-to-hidden0, still uses the existing scalar path.
- The local device still does not satisfy the 98304-byte KHR training shared-memory profile gate, so the final-hidden `dA` mode has compile coverage and a wired runtime comparison but no local GPU numerical comparison yet.
- The CPU oracle intentionally models the new mode's fp16 staging behavior, but it should be revisited once a device can execute the KHR training profile numerically.

### Next Step

Extend cooperative-matrix `dA` one hidden layer at a time, starting with hidden3-to-hidden2, then hidden2-to-hidden1 and hidden1-to-hidden0. Keep each layer behind an explicit mode until the full derivative chain has compile coverage, CPU oracle coverage, and a real-device numerical pass.

## 2026-07-09 - Milestone 78 KHR Hidden3 Coopmat dA Mode

### Implemented

- Added the second optional KHR cooperative-matrix `dA` propagation path: hidden layer 3 into hidden layer 2.
- Added `NRC_KHR_USE_COOPMAT_HIDDEN3_DA`, defaulting to disabled, so the baseline KHR gradient path remains unchanged unless a test/profile explicitly enables this mode.
- Reused the existing `NrcKhrPropagateHiddenDActivationCoopmat` helper instead of adding another shader helper. The same helper now covers:
  - final-hidden to hidden3;
  - hidden3 to hidden2.
- Wired the hidden3 `dA` switch after hidden-layer-3 `dW` accumulation. At that point the hidden-layer-3 activation snapshot has already been consumed for the ReLU mask, so the helper can safely store fp16 `dHidden2` into `NRC_KHR_BACKPROP_SOURCE_ACT_COPY_UVEC4_BASE`.
- Generalized the KHR runtime test oracle:
  - replaced the single `coopmatDActivationLayer` selector with a `CoopmatDActivationLayers` mask;
  - kept the final-hidden `dA` oracle behavior unchanged;
  - added a hidden3 `dA` oracle that models fp16 hidden-layer-3 `dPre` staging and fp16 `dHidden2` storage before recomputing hidden-layer 2/1/0 gradients.
- Added the `--coopmat-hidden3-da` runtime test mode.
- Added compile coverage for the hidden3 `dA` mode across the current KHR shader profiles:
  - subgroup 32;
  - subgroup 16;
  - `m8` subgroup 32.
- Updated current documentation:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/khr_shared_memory_layout.md`
  - `docs/training_optimizer.md`

### Behavior

This milestone introduces another narrow cooperative-matrix `dA` option while keeping the existing `dW` modes intact:

```text
default:                       KHR groundwork oracle
--coopmat-output:              output-layer fp16 coopmat dW oracle
--coopmat-final-hidden:        hidden layer 4 fp16 coopmat dW oracle
--coopmat-final-hidden-da:     final-hidden -> hidden3 fp16 coopmat dA oracle
--coopmat-hidden3-da:          hidden3 -> hidden2 fp16 coopmat dA oracle
--coopmat-hidden3:             hidden layer 3 fp16 coopmat dW oracle
--coopmat-hidden2:             hidden layer 2 fp16 coopmat dW oracle
--coopmat-hidden1:             hidden layer 1 fp16 coopmat dW oracle
--coopmat-hidden0:             hidden layer 0 fp16 coopmat dW oracle
--coopmat-all-dw:              output and all hidden dW regions use fp16 coopmat oracles
```

The new hidden3 `dA` mode is independent from the final-hidden `dA` mode. That keeps each derivative propagation step separately auditable before a later milestone introduces cumulative/full-chain `dA` combinations.

### Validation

- Diff whitespace check succeeded:
  - `git diff --check`
- Targeted KHR gradient runtime test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr" --output-on-failure -V`
- Targeted KHR gradient runtime result:
  - `10/10 tests passed`
- Current local KHR training skip result for all ten runtime modes remains:
  - `SKIP: no Vulkan device supports an implemented NRC KHR cooperative matrix training profile`
- Targeted hidden3 `dA` compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr_coopmat_hidden3_da" --output-on-failure -V`
- Targeted hidden3 `dA` compile result:
  - `3/3 tests passed`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Full test result:
  - `69/69 tests passed`

### Current Limitations

- Only two `dA` propagation steps have optional cooperative-matrix implementations:
  - final-hidden to hidden3;
  - hidden3 to hidden2.
- The default KHR gradient path still uses scalar `dA` propagation.
- The remaining hidden derivative chain, hidden2-to-hidden1 and hidden1-to-hidden0, still uses the existing scalar path.
- The local device still does not satisfy the 98304-byte KHR training shared-memory profile gate, so the hidden3 `dA` mode has compile coverage and a wired runtime comparison but no local GPU numerical comparison yet.
- The current `dA` modes are independent. A future cumulative mode still needs to validate chained fp16 `dA` staging semantics before making any all-`dA` or all-coopmat training variant.

### Next Step

Continue the layer-by-layer KHR cooperative-matrix `dA` port with hidden2-to-hidden1, then hidden1-to-hidden0. After every individual `dA` step has compile coverage and oracle coverage, add a cumulative full-chain `dA` mode and only then consider combining it with all-`dW`.

## 2026-07-09 - Milestone 79 KHR Hidden2 Coopmat dA Mode

### Implemented

- Added the third optional KHR cooperative-matrix `dA` propagation path: hidden layer 2 into hidden layer 1.
- Added `NRC_KHR_USE_COOPMAT_HIDDEN2_DA`, defaulting to disabled, so the baseline KHR gradient path remains unchanged unless a test/profile explicitly enables this mode.
- Reused `NrcKhrPropagateHiddenDActivationCoopmat` again. The same helper now covers:
  - final-hidden to hidden3;
  - hidden3 to hidden2;
  - hidden2 to hidden1.
- Wired the hidden2 `dA` switch after hidden-layer-2 `dW` accumulation. At that point the hidden-layer-2 activation snapshot has already been consumed for the ReLU mask, so the helper can safely store fp16 `dHidden1` into `NRC_KHR_BACKPROP_PREVIOUS_SOURCE_ACT_COPY_UVEC4_BASE`.
- Added a hidden2 `dA` test oracle wrapper that models fp16 hidden-layer-2 `dPre` staging and fp16 `dHidden1` storage before recomputing hidden-layer 1 and hidden-layer 0 gradients.
- Added the `--coopmat-hidden2-da` runtime test mode.
- Added compile coverage for the hidden2 `dA` mode across the current KHR shader profiles:
  - subgroup 32;
  - subgroup 16;
  - `m8` subgroup 32.
- Updated current documentation:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/khr_shared_memory_layout.md`
  - `docs/training_optimizer.md`

### Behavior

This milestone introduces another independent cooperative-matrix `dA` option while keeping the existing `dW` modes intact:

```text
default:                       KHR groundwork oracle
--coopmat-output:              output-layer fp16 coopmat dW oracle
--coopmat-final-hidden:        hidden layer 4 fp16 coopmat dW oracle
--coopmat-final-hidden-da:     final-hidden -> hidden3 fp16 coopmat dA oracle
--coopmat-hidden3-da:          hidden3 -> hidden2 fp16 coopmat dA oracle
--coopmat-hidden2-da:          hidden2 -> hidden1 fp16 coopmat dA oracle
--coopmat-hidden3:             hidden layer 3 fp16 coopmat dW oracle
--coopmat-hidden2:             hidden layer 2 fp16 coopmat dW oracle
--coopmat-hidden1:             hidden layer 1 fp16 coopmat dW oracle
--coopmat-hidden0:             hidden layer 0 fp16 coopmat dW oracle
--coopmat-all-dw:              output and all hidden dW regions use fp16 coopmat oracles
```

The new hidden2 `dA` mode remains separate from the other `dA` modes. That keeps each derivative propagation step separately auditable before adding a cumulative/full-chain `dA` mode.

### Validation

- Diff whitespace check succeeded:
  - `git diff --check`
- Targeted KHR gradient runtime test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr" --output-on-failure -V`
- Targeted KHR gradient runtime result:
  - `11/11 tests passed`
- Current local KHR training skip result for all eleven runtime modes remains:
  - `SKIP: no Vulkan device supports an implemented NRC KHR cooperative matrix training profile`
- Targeted hidden2 `dA` compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr_coopmat_hidden2_da" --output-on-failure -V`
- Targeted hidden2 `dA` compile result:
  - `3/3 tests passed`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Full test result:
  - `73/73 tests passed`

### Current Limitations

- Three `dA` propagation steps have optional cooperative-matrix implementations:
  - final-hidden to hidden3;
  - hidden3 to hidden2;
  - hidden2 to hidden1.
- The default KHR gradient path still uses scalar `dA` propagation.
- The remaining hidden derivative step, hidden1-to-hidden0, still uses the existing scalar path.
- The local device still does not satisfy the 98304-byte KHR training shared-memory profile gate, so the hidden2 `dA` mode has compile coverage and a wired runtime comparison but no local GPU numerical comparison yet.
- The current `dA` modes are independent. A future cumulative mode still needs to validate chained fp16 `dA` staging semantics before making any all-`dA` or all-coopmat training variant.

### Next Step

Finish the individual KHR cooperative-matrix `dA` pass with hidden1-to-hidden0. Then add a cumulative full-chain `dA` mode and compare its oracle against the expected chained fp16 staging before combining it with all-`dW`.

## 2026-07-09 - Milestone 80 KHR Hidden1 Coopmat dA Mode

### Implemented

- Added the fourth optional KHR cooperative-matrix `dA` propagation path: hidden layer 1 into hidden layer 0.
- Added `NRC_KHR_USE_COOPMAT_HIDDEN1_DA`, defaulting to disabled, so the baseline KHR gradient path remains unchanged unless a test/profile explicitly enables this mode.
- Reused `NrcKhrPropagateHiddenDActivationCoopmat` again. The same helper now covers every individual hidden-to-hidden derivative propagation step:
  - final-hidden to hidden3;
  - hidden3 to hidden2;
  - hidden2 to hidden1;
  - hidden1 to hidden0.
- Wired the hidden1 `dA` switch after hidden-layer-1 `dW` accumulation. At that point the hidden-layer-1 activation snapshot has already been consumed for the ReLU mask, so the helper can safely store fp16 `dHidden0` into `NRC_KHR_BACKPROP_EARLIER_SOURCE_ACT_COPY_UVEC4_BASE`.
- Added a hidden1 `dA` test oracle wrapper that models fp16 hidden-layer-1 `dPre` staging and fp16 `dHidden0` storage before recomputing hidden-layer 0 gradients.
- Added the `--coopmat-hidden1-da` runtime test mode.
- Added compile coverage for the hidden1 `dA` mode across the current KHR shader profiles:
  - subgroup 32;
  - subgroup 16;
  - `m8` subgroup 32.
- Updated current documentation:
  - `docs/gradient_pass.md`
  - `docs/khr_coopmat_backend.md`
  - `docs/khr_shared_memory_layout.md`
  - `docs/training_optimizer.md`

### Behavior

This milestone completes the set of independent single-step KHR cooperative-matrix `dA` options while keeping the existing `dW` modes intact:

```text
default:                       KHR groundwork oracle
--coopmat-output:              output-layer fp16 coopmat dW oracle
--coopmat-final-hidden:        hidden layer 4 fp16 coopmat dW oracle
--coopmat-final-hidden-da:     final-hidden -> hidden3 fp16 coopmat dA oracle
--coopmat-hidden3-da:          hidden3 -> hidden2 fp16 coopmat dA oracle
--coopmat-hidden2-da:          hidden2 -> hidden1 fp16 coopmat dA oracle
--coopmat-hidden1-da:          hidden1 -> hidden0 fp16 coopmat dA oracle
--coopmat-hidden3:             hidden layer 3 fp16 coopmat dW oracle
--coopmat-hidden2:             hidden layer 2 fp16 coopmat dW oracle
--coopmat-hidden1:             hidden layer 1 fp16 coopmat dW oracle
--coopmat-hidden0:             hidden layer 0 fp16 coopmat dW oracle
--coopmat-all-dw:              output and all hidden dW regions use fp16 coopmat oracles
```

The new hidden1 `dA` mode remains separate from the other `dA` modes. The next behavior change should be cumulative: enabling the whole hidden-to-hidden `dA` chain together and using an oracle that models chained fp16 derivative staging.

### Validation

- Diff whitespace check succeeded:
  - `git diff --check`
- Targeted KHR gradient runtime test build succeeded:
  - `cmake --build build --config Debug --target nrc_vulkan_gradient_khr_test`
- Targeted KHR gradient runtime tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_vulkan_gradient_khr" --output-on-failure -V`
- Targeted KHR gradient runtime result:
  - `12/12 tests passed`
- Current local KHR training skip result for all twelve runtime modes remains:
  - `SKIP: no Vulkan device supports an implemented NRC KHR cooperative matrix training profile`
- Targeted hidden1 `dA` compile tests succeeded:
  - `ctest --test-dir build -C Debug -R "nrc_shader_gradient_khr_coopmat_hidden1_da" --output-on-failure -V`
- Targeted hidden1 `dA` compile result:
  - `3/3 tests passed`
- Debug build succeeded:
  - `cmake --build build --config Debug`
- Full test command succeeded:
  - `ctest --test-dir build -C Debug --output-on-failure`
- Full test result:
  - `77/77 tests passed`

### Current Limitations

- Every individual hidden-to-hidden `dA` propagation step now has an optional cooperative-matrix implementation, but the default KHR gradient path still uses scalar `dA` propagation.
- The current `dA` modes are independent. There is not yet a cumulative full-chain `dA` mode that enables final-hidden-to-hidden3, hidden3-to-hidden2, hidden2-to-hidden1, and hidden1-to-hidden0 together.
- The local device still does not satisfy the 98304-byte KHR training shared-memory profile gate, so the hidden1 `dA` mode has compile coverage and a wired runtime comparison but no local GPU numerical comparison yet.
- The cumulative mode must validate chained fp16 `dA` staging semantics carefully before any all-`dA` or all-coopmat training variant is considered correct.

### Next Step

Add a cumulative full-chain KHR cooperative-matrix `dA` mode that enables all four hidden-to-hidden derivative propagation steps together. Its CPU oracle should use the existing `CoopmatDActivationLayers` mask with all valid hidden derivative layers enabled, then the runtime mode can become the foundation for combining all `dW` and all `dA` cooperative-matrix paths.
