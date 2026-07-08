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
