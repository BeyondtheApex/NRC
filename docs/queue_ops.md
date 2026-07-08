# Queue Operations

Milestone 11 adds reusable queue state and reservation helpers:

- Host ABI:
  - `nrc::NrcQueueState`
- Shader pass/header:
  - `shaders/nrc/passes/NrcQueueOps.glsl`
- Runtime validation:
  - `tests/vulkan_queue_ops_test.cpp`
  - `tests/shader_compile/nrc_queue_ops_dispatch.comp`

## Queue State

The queue state is:

```text
uint evalCount
uint trainCounts[4]
```

The C++ ABI size is 20 bytes, matching the shader `std430` layout.

## Eval Queue

`NrcReserveEvalRecord(capacity, out index)` reserves one eval record slot.

Unlike VkNRC's direct `atomicAdd(uEvalCount, 1)`, this helper uses a compare-and-swap loop and clamps the count at `capacity`. This prevents later inference dispatches from reading past the eval-record buffer when more work is requested than the configured capacity can hold.

`NrcPushEvalRecord(record, capacity, out index)` reserves and writes a record.

## Train Queue

`NrcReserveTrainRecords(batch, requestedCount, out firstIndex, out reservedCount)` reserves a contiguous range in one train batch.

The helper clamps each batch count at `NRC_TRAIN_BATCH_SIZE`. If the requested range would cross the end of the batch, the caller receives the truncated `reservedCount`.

Train records are stored in this repository's flat layout:

```text
flatIndex = batch * NRC_TRAIN_BATCH_SIZE + indexInBatch
```

## Reset

`NrcQueueResetMain()` resets:

- `evalCount`
- all `trainCounts`

It is intended for a small compute dispatch at the start of a frame or sample iteration.

## Validation

The Vulkan test:

- seeds queue counts with non-zero values;
- dispatches reset;
- dispatches 64 queue producers;
- reserves 17 eval records from 64 contenders;
- reserves more train records than a batch can hold;
- verifies eval count is clamped at 17;
- verifies train batch 0 count is clamped at 16384;
- verifies batches 1-3 remain zero;
- verifies written eval/train records contain expected payloads.

## Current Limitations

- Queue ops are validated with a synthetic producer shader, not yet the reference path tracer.
- No indirect dispatch argument generation exists yet.
- The queue helpers do not implement compaction, sorting, or priority.
- Overflow is intentionally dropped by clamped reservation; callers that need diagnostics should compare requested work against returned counts or inspect final queue counts.
