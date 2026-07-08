# Profile and Resource Layout Contract

This document records the first host-side contract for the reusable NRC implementation.

## Profile Selection

The library selects from validated profiles instead of allowing arbitrary cooperative matrix and workgroup combinations.

Initial profiles:

- `KHR_FP16_M16N16K16_SG32_WG128`
- `KHR_FP16_M16N16K16_SG16_WG128`
- `KHR_FP16_M8N16K16_SG32_WG128`
- `SCALAR_REFERENCE_WG64`

The scalar profile is for correctness and CI. It is never selected unless explicitly allowed by the caller.

Profile selection proves that a backend is a valid candidate for the reported capabilities. It does not prove that the driver can compile and run the backend shader. Integrations should also validate pipeline creation and, where possible, run the backend numerical validation before enabling a production KHR profile.

## Support Diagnostics

`nrc::evaluate_profile_support(capabilities, requirements, profile)` returns a `ProfileSupportReport` for one profile. It uses the same checks as `supports_profile`, but records the first decisive support status:

- `Supported`
- `ProfileNotImplemented`
- `ReferenceBackendNotAllowed`
- `UnsupportedBackend`
- `MissingRequiredFeature`
- `MissingTrainingAtomicFloat`
- `MissingSubgroupArithmetic`
- `MissingSubgroupSize`
- `MissingCooperativeMatrixShape`
- `MissingSharedMemory`

The report also carries the profile id, profile name, and final `supported` bool. This is intended for integration logs and user-facing diagnostics when a renderer queries matrix support across different vendors.

These diagnostics are capability-level only. They do not prove that a Vulkan device can create the selected shader pipeline or that numerical validation has passed.

## Shader Configuration

`nrc::find_profile_shader_config(profileId)` maps a selected profile to the shader-side configuration needed by an integration:

- backend kind;
- shader header path;
- workgroup size;
- required subgroup size;
- cooperative matrix `M`, `N`, and `K`;
- whether full subgroups and subgroup-size control are required.
- whether the training shader path requires subgroup arithmetic.

For KHR cooperative matrix profiles, integrations should use these fields to select or compile shader variants with:

```text
NRC_KHR_WORKGROUP_SIZE
NRC_KHR_SUBGROUP_SIZE
NRC_KHR_COOPMAT_M
NRC_KHR_COOPMAT_N
NRC_KHR_COOPMAT_K
```

This keeps profile selection and shader compilation synchronized as more matrix sizes are added.

`nrc::compute_khr_shader_shared_memory_layout(config)` mirrors the KHR shader's shared-memory constants for a selected `ProfileShaderConfig`. It reports the current 98304-byte KHR backprop shared allocation, saved activation snapshot offsets, dPre/output scratch regions, and hidden dW partial-storage constraints. See `docs/khr_shared_memory_layout.md`.

The layout model also records a hidden-layer cooperative-matrix `dW` tile-streaming plan. The plan keeps one row block of fp16 `dPre` and one MxN partial tile per subgroup inside the active tile, preserving all saved activation snapshots for SG32, SG16, and M8/SG32 profiles.

## Capability Input

`nrc::DeviceCapabilitiesView` is renderer-neutral. Vulkan-specific code should query device features and cooperative matrix properties, then translate them into this view.

Production KHR profiles require:

- KHR cooperative matrix support.
- Shader float16 support.
- Storage buffer 16-bit access.
- Subgroup size control.
- Full compute subgroups.
- Subgroup arithmetic when training is required.
- Matching subgroup size.
- Matching FP16 subgroup cooperative matrix shape.
- Float32 buffer atomic add when training is required.
- `maxComputeSharedMemorySize` large enough for the KHR backprop shared-memory layout when training is required.

The current KHR training layout requires 98304 bytes of compute shared memory. This requirement is skipped when callers choose with `ProfileRequirements::requireTraining = false`, because the inference shader does not use the backprop shared allocation.

## VkNRC-Compatible Layout

The first implementation follows VkNRC's model:

- 64 encoded fp16 inputs.
- 5 hidden layers of width 64.
- 3 fp16 outputs.
- 20672 trainable weights.
- 4 train batches of 16384 records.

Record sizes:

```text
PackedNRCInput   16 bytes
NRCEvalRecord    20 bytes
NRCTrainRecord   40 bytes
OptimizerState   20 bytes
OptimizerEntry   16 bytes
NRCQueueState    20 bytes
```

For a `1280x720` target:

```text
evalRecordCount              987136
evalRecordsBytes             19742720
evalCountBytes               4
queueStateBytes              20
trainBatchCountBytes         16
weightsBytes                 41344
useWeightsBytes              41344
optimizerEntriesBytes        330752
gradientsBytes               82688
singleTrainBatchRecordBytes  655360
allTrainBatchRecordBytes     2621440
```

## Optimizer Constants

Training support follows VkNRC's optimizer defaults:

```text
lossScale             1.0
adamBeta1             0.9
adamBeta2             0.999
emaAlpha              0.99
optimizerLearningRate 0.002
optimizerEpsilon      1e-8
```

The reusable shader passes select the active train batch through push constants and read the count from `NRCQueueState.trainCounts[batch]`.
