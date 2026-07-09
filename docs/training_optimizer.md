# Training Prepare and Optimizer

This document covers the reusable training support added before the gradient pass.

## Files

- Host reference API:
  - `include/nrc/nrc_optimizer.hpp`
  - `src/nrc_optimizer.cpp`
- Shader-header passes:
  - `shaders/nrc/passes/NrcTrainPrepare.glsl`
  - `shaders/nrc/passes/NrcOptimize.glsl`
- Tests:
  - `tests/optimizer_test.cpp`
  - `tests/shader_compile/nrc_train_prepare_dispatch.comp`
  - `tests/shader_compile/nrc_optimize_dispatch.comp`
  - `tests/vulkan_training_step_test.cpp`
  - `sample/path_tracer`

## VkNRC Behavior Preserved

The optimizer follows VkNRC's `nrc_train_prepare.comp` and `nrc_optimize.comp` behavior:

- clamp train count to `NRC_TRAIN_BATCH_SIZE`;
- advance optimizer state only when the clamped train count is non-zero;
- use Adam beta values:
  - `beta1 = 0.9`;
  - `beta2 = 0.999`;
- use EMA alpha:
  - `0.99`;
- divide gradients by `trainCount * lossScale`;
- ignore non-finite gradients by treating them as zero;
- write fp16 weights after updating fp32 optimizer entries;
- optionally write `useWeights` from either the updated fp32 weight or EMA weight.

## Deliberate ABI Difference

VkNRC stores each train-batch count in a separate buffer. This reusable project stores all queue counters in:

```text
NrcQueueState {
  uint evalCount;
  uint trainCounts[4];
}
```

`NrcTrainPrepare.glsl` and `NrcOptimize.glsl` therefore select the batch through push constants and read `trainCounts[batch]`. This keeps the queue ABI compact for downstream engines while preserving the optimizer math.

## Current Validation

`nrc_optimizer_test` validates the CPU reference behavior:

- optimizer entries initialize from fp32 weights;
- zero-count batches do not advance state;
- oversized train counts clamp to `NRC_TRAIN_BATCH_SIZE`;
- non-zero batches advance `step`, `beta1T`, `beta2T`, `alphaT`, and `alphaTMinus1`;
- Adam updates match the expected first-step formula;
- non-finite gradients are ignored;
- first-step EMA use weights match updated weights.

`nrc_khr_gradient_reference_test` validates the reusable KHR fp16 gradient CPU oracle that the KHR runtime gradient test uses for full hidden/output gradient comparison.

Shader compile tests validate the descriptor and GLSL ABI for:

- train prepare;
- optimize without `useWeights`;
- optimize with `NRC_OPTIMIZE_WRITE_USE_WEIGHTS=1`.

`nrc_vulkan_training_step_test` validates the first full Vulkan training step over the reusable passes. It runs:

```text
NrcTrainPrepare -> NrcGradientScalar -> NrcOptimize
```

The test uses the prepare pass output as the indirect dispatch command for the scalar gradient pass, then compares:

- clamped train count;
- optimizer state advancement;
- indirect dispatch command;
- full fp32 gradient buffer;
- fp32 optimizer entries;
- packed fp16 weights;
- packed fp16 use-weights.

The CPU oracle uses `nrc::encode_input`, `nrc::reference_backward`, and `nrc::optimize_weights` over the same fp16-rounded weights consumed by the shaders.

`sample/path_tracer` also runs the scalar training sequence after producing scene-backed train records and resolving them through scalar inference. The sample compares gradients, optimizer entries, packed weights, and packed use-weights against a CPU oracle that mirrors the sample scene unpacking path.

## Current Limitations

- Full prepare -> scalar gradient -> optimize validation exists, but it is still the scalar correctness path rather than a production-performance training backend.
- The sample path tracer runs one validated training update, not an iterative per-frame training loop.
- KHR cooperative-matrix training now has a recompute all-coopmat branch that runs within a 32768-byte shared-memory contract and passes the local Vulkan gradient dispatch test. `nrc::choose_training_profile`, `nrc::make_training_shader_compile_config`, `nrc::vulkan::make_vulkan_training_profile_plan`, and `nrc::vulkan::make_vulkan_training_profile_plan_with_runtime_fallback` expose the integration-facing selection/plan contract. The sample path tracer records that plan, runs scalar training as the correctness anchor, and additionally creates/dispatches an optional KHR recompute gradient probe when supported. KHR optimizer integration and renderer-level performance validation are still pending.
- NV cooperative-matrix shader paths are intentionally not implemented in this KHR-focused reusable library.
