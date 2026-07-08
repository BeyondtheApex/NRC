# Profile and Resource Layout Contract

This document records the first host-side contract for the reusable NRC implementation.

## Profile Selection

The library selects from validated profiles instead of allowing arbitrary cooperative matrix and workgroup combinations.

Initial profiles:

- `KHR_FP16_M16N16K16_SG32_WG128`
- `KHR_FP16_M16N16K16_SG16_WG128`
- `KHR_FP16_M8N16K16_SG32_WG128` (planned, not selected until the backend is implemented)
- `SCALAR_REFERENCE_WG64`

The scalar profile is for correctness and CI. It is never selected unless explicitly allowed by the caller.

Profile selection proves that a backend is a valid candidate for the reported capabilities. It does not prove that the driver can compile and run the backend shader. Integrations should also validate pipeline creation and, where possible, run the backend numerical validation before enabling a production KHR profile.

## Capability Input

`nrc::DeviceCapabilitiesView` is renderer-neutral. Vulkan-specific code should query device features and cooperative matrix properties, then translate them into this view.

Production KHR profiles require:

- KHR cooperative matrix support.
- Shader float16 support.
- Storage buffer 16-bit access.
- Subgroup size control.
- Full compute subgroups.
- Matching subgroup size.
- Matching FP16 subgroup cooperative matrix shape.
- Float32 buffer atomic add when training is required.

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
```

For a `1280x720` target:

```text
evalRecordCount              987136
evalRecordsBytes             19742720
weightsBytes                 41344
useWeightsBytes              41344
optimizerEntriesBytes        330752
gradientsBytes               82688
singleTrainBatchRecordBytes  655360
allTrainBatchRecordBytes     2621440
```
