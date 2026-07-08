# Gradient Passes

`shaders/nrc/passes/NrcGradientScalar.glsl` is the first reusable gradient shader-header pass.

## Purpose

The pass is a correctness-oriented scalar backpropagation path. It is intended to validate the training ABI before KHR cooperative-matrix training is implemented.

## Integration Contract

Like inference, the gradient pass is scene-agnostic. The integration must provide:

```glsl
NrcUnpackedInput NrcUnpackPackedInput(PackedNrcInput packedInput);
```

The pass consumes:

- flat `NrcTrainRecord` storage;
- `NrcQueueState` for train batch counts;
- packed fp16 weights;
- fp32 gradient storage.

The active train batch is selected by push constant.

## Loss

The output derivative matches VkNRC's relative L2 luminance expression:

```text
2 * lossScale * (prediction - target)
/ (luminance(max(prediction, 0))^2 + 0.01)
```

The luminance denominator is treated as the current prediction scale, matching VkNRC's shader behavior.

## Current Validation

`tests/shader_compile/nrc_gradient_scalar_dispatch.comp` compile-validates the shader integration boundary with a fake unpack function.

`tests/reference_gradient_test.cpp` validates the CPU reference backpropagation against finite differences.

`tests/khr_gradient_reference_test.cpp` validates `nrc::reference_backward_khr_fp16`, the reusable CPU oracle for the current KHR fp16 gradient path. The test checks that every hidden/output gradient region is finite and non-zero for a positive test network, compares coverage shape against the scalar reference path, and validates accumulate/error behavior.

`tests/vulkan_gradient_scalar_test.cpp` dispatches the scalar gradient pass on Vulkan when `VK_EXT_shader_atomic_float` and `shaderBufferFloat32AtomicAdd` are available. The test uploads packed fp16 weights, train records, and queue state, then compares the full fp32 gradient buffer against `nrc::reference_backward`.

`tests/vulkan_gradient_khr_test.cpp` selects a KHR cooperative-matrix training profile when cooperative matrices, fp32 storage-buffer atomic add, subgroup arithmetic, and the KHR training shared-memory requirement are available. If pipeline creation succeeds, it dispatches `NrcGradientKhr.glsl` over a 67-record one-workgroup batch and compares all hidden-layer and output-layer gradient regions against the matching CPU oracle. On the current local AMD driver, KHR training profile selection reports a controlled `SKIP:` because no implemented training profile satisfies the current 98304-byte shared-memory requirement.

KHR shader compile tests also cover experimental cooperative-matrix `dW` variants for the output layer, final hidden layer, hidden layer 3, hidden layer 2, and hidden layer 1 across SG32, SG16, and M8/SG32 profiles.

`tests/vulkan_training_step_test.cpp` runs the gradient pass between `NrcTrainPrepare` and `NrcOptimize`, using prepare's indirect dispatch command and comparing the final optimizer state and packed weights against the CPU reference path.

`sample/path_tracer` dispatches the same scalar gradient pass with sample scene unpacking after scalar inference has resolved train-record targets.

`tests/encoding_test.cpp` validates the host-side `nrc::encode_input` mirror that the Vulkan gradient test uses to match the shader-side `NrcEncodeInput` path.

## Current Limitations

- The pass uses scalar loops and one invocation per train record. It is not a production-performance path.
- It uses fp32 storage-buffer atomic add and therefore requires `VK_EXT_shader_atomic_float` / `shaderBufferFloat32AtomicAdd`.
- The Vulkan gradient test is feature-gated and reports a controlled skip on devices without fp32 storage-buffer atomic add support.
- The sample path tracer currently runs one validated scalar training update, not an iterative training loop.
- The KHR cooperative-matrix pass now has full hidden/output `dW` training groundwork and subgroup reduction for every hidden/output `dW` region. Experimental cooperative-matrix `dW` helpers exist for the output layer, final hidden layer, hidden layer 3, hidden layer 2, and hidden layer 1, but the default path still waits on runtime numerical validation on a successful KHR pipeline creation path.

## KHR Gradient Pass

`shaders/nrc/passes/NrcGradientKhr.glsl` is the first KHR cooperative-matrix train-record gradient pass.

It uses the same external ABI as the scalar gradient pass:

- flat `NrcTrainRecord` storage;
- `NrcQueueState` train batch counts;
- packed fp16 weights;
- fp32 gradient storage;
- push-constant train batch selection;
- integration-provided `NrcUnpackPackedInput`.

Current behavior:

- all invocations in a workgroup participate in KHR cooperative-matrix forward work;
- train indices outside the selected batch count use zero input and zero output derivative;
- source activations before hidden layers 1, 2, 3, and 4, plus before the output layer, are saved for backpropagation;
- encoded input is kept as the source activation for hidden layer 0 without adding another shared-memory snapshot;
- output-layer `dW` is reduced across each subgroup with `subgroupAdd`, one subgroup cooperates on each output weight, and the result is accumulated into the same fp32 row-major gradient layout as the scalar pass;
- output derivative is propagated through the output weights to final hidden activations;
- hidden-layer 4 `dW` is reduced across each subgroup from cached output derivatives and accumulated into the same fp32 row-major gradient layout as the scalar pass;
- hidden-layer 4 derivative is propagated to hidden-layer 3 activations;
- hidden-layer 3 `dW` is reduced across each subgroup through shared fp32 `dZ` scratch and accumulated into the same fp32 row-major gradient layout as the scalar pass;
- hidden-layer 3 derivative is propagated to hidden-layer 2 activations;
- hidden-layer 2 `dW` is reduced across each subgroup through shared fp32 `dZ` scratch and accumulated into the same fp32 row-major gradient layout as the scalar pass;
- hidden-layer 2 derivative is propagated to hidden-layer 1 activations;
- hidden-layer 1 `dW` is reduced across each subgroup through shared fp32 `dZ` scratch and accumulated into the same fp32 row-major gradient layout as the scalar pass;
- hidden-layer 1 derivative is propagated to hidden-layer 0 activations;
- hidden-layer 0 `dW` stages encoded input into a retired shared snapshot region, then reduces across each subgroup into the same fp32 row-major gradient layout as the scalar pass.

Experimental KHR output-layer cooperative-matrix reduction:

- `NrcKhrAccumulateOutputDWeightsCoopmat` stages each invocation's output derivative as fp16, multiplies final hidden activations by the staged derivative through KHR cooperative matrices, stores one fp16 partial dW tile per subgroup, then unpacks and reduces those subgroup partials into the existing fp32 gradient buffer with one atomic add per output weight per workgroup.
- After reducing the output partials, the helper restores each invocation's fp32 `dOutput` into `nrcKhrSharedBuffer[gl_LocalInvocationID.x]`. The final-hidden `dW` helper depends on that shared fp32 derivative cache.
- `NRC_KHR_USE_COOPMAT_OUTPUT_DW=1` switches `NrcGradientKhr.glsl` to this helper for output-layer `dW`.
- `nrc_vulkan_gradient_khr_coopmat_output_test` builds matching SG32, SG16, and M8/SG32 SPIR-V variants and uses a CPU oracle that mirrors fp16 output-derivative staging and per-subgroup fp16 output partial storage.
- The default KHR gradient pass still uses the subgroup-arithmetic output helper until runtime numerical validation succeeds on a driver that can create the KHR gradient pipeline.

Experimental KHR final-hidden cooperative-matrix reduction:

- `NrcKhrAccumulateHiddenDWeightsCoopmatTileStreaming` stages one row block of fp16 `dPre`, loads source activations as row-major KxN cooperative matrices, stores one MxN fp16 partial tile per subgroup, then reduces those partials into the existing fp32 gradient buffer.
- `NRC_KHR_USE_COOPMAT_FINAL_HIDDEN_DW=1` switches only hidden layer 4 `dW` to this helper.
- The helper follows the host-side tile-streaming layout plan and keeps all temporary storage inside the active tile, preserving saved activation snapshots for SG32, SG16, and M8/SG32 profiles.
- `nrc_vulkan_gradient_khr_coopmat_final_hidden_test` builds matching SG32, SG16, and M8/SG32 SPIR-V variants and uses a CPU oracle that mirrors fp16 `dPre` row-block staging and per-subgroup fp16 final-hidden partial storage.
- The 67-record test batch spans multiple subgroup sample ranges for both SG16 and SG32, so the coopmat oracles now model the shader's cross-subgroup partial reduction instead of only subgroup 0.
- The local test currently reaches a controlled profile-selection skip before dispatch because the available device does not satisfy the 98304-byte KHR training shared-memory requirement.

Experimental KHR hidden-layer-3 cooperative-matrix reduction:

- `NRC_KHR_USE_COOPMAT_HIDDEN3_DW=1` switches only hidden layer 3 `dW` to `NrcKhrAccumulateHiddenDWeightsCoopmatTileStreaming`.
- The pass still propagates `dHidden3` to hidden layer 2 through the local derivative array, so the helper does not need to preserve the active-tile fp32 `dPre` scratch after hidden layer 3 `dW`.
- `nrc_vulkan_gradient_khr_coopmat_hidden3_test` builds matching SG32, SG16, and M8/SG32 SPIR-V variants and uses a CPU oracle that mirrors hidden layer 4 backpropagation, fp16 hidden-layer-3 `dPre` staging, and per-subgroup fp16 partial storage.
- The default KHR gradient pass still uses the subgroup hidden-layer-3 helper until runtime numerical validation succeeds on a driver that can create the KHR gradient pipeline.

Experimental KHR hidden-layer-2 cooperative-matrix reduction:

- `NRC_KHR_USE_COOPMAT_HIDDEN2_DW=1` switches only hidden layer 2 `dW` to `NrcKhrAccumulateHiddenDWeightsCoopmatTileStreaming`.
- The pass still propagates `dHidden2` to hidden layer 1 through the local derivative array, so the helper can reuse the active-tile scratch without preserving fp32 `dPre`.
- `nrc_vulkan_gradient_khr_coopmat_hidden2_test` builds matching SG32, SG16, and M8/SG32 SPIR-V variants and uses a CPU oracle that mirrors hidden layer 4 and hidden layer 3 backpropagation, fp16 hidden-layer-2 `dPre` staging, and per-subgroup fp16 partial storage.
- The default KHR gradient pass still uses the subgroup hidden-layer-2 helper until runtime numerical validation succeeds on a driver that can create the KHR gradient pipeline.

Experimental KHR hidden-layer-1 cooperative-matrix reduction:

- `NRC_KHR_USE_COOPMAT_HIDDEN1_DW=1` switches only hidden layer 1 `dW` to `NrcKhrAccumulateHiddenDWeightsCoopmatTileStreaming`.
- The pass still propagates `dHidden1` to hidden layer 0 through the local derivative array.
- `nrc_vulkan_gradient_khr_coopmat_hidden1_test` builds matching SG32, SG16, and M8/SG32 SPIR-V variants and uses a CPU oracle that mirrors hidden layer 4, hidden layer 3, and hidden layer 2 backpropagation, fp16 hidden-layer-1 `dPre` staging, and per-subgroup fp16 partial storage.
- The default KHR gradient pass still uses the subgroup hidden-layer-1 helper until runtime numerical validation succeeds on a driver that can create the KHR gradient pipeline.

Current limitations:

- the default hidden/output `dW` regions use subgroup arithmetic and one fp32 atomic add per covered weight per workgroup;
- the optional output-layer, final-hidden, hidden-layer-3, hidden-layer-2, and hidden-layer-1 cooperative-matrix helpers are not the default path;
- the optional output-layer, final-hidden, hidden-layer-3, hidden-layer-2, and hidden-layer-1 cooperative-matrix runtime tests are present but currently report the same controlled KHR training profile-selection skip as the default KHR gradient test on the local AMD driver;
- hidden layer 0 still needs a separate cooperative-matrix `dW` design because its source activations are the encoded input rather than another saved hidden activation snapshot;
- KHR backpropagation of `dA` still uses scalar arrays rather than VkNRC-style cooperative matrices;
- hidden-layer cooperative-matrix dW beyond the final hidden layer still needs staged integration and numerical validation;
- KHR runtime gradient comparison is present but remains driver-gated on the current local AMD pipeline creation path.
