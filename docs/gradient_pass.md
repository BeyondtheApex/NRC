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

`tests/vulkan_gradient_khr_test.cpp` selects a KHR cooperative-matrix training profile when both cooperative matrices and fp32 storage-buffer atomic add are available. If pipeline creation succeeds, it dispatches `NrcGradientKhr.glsl` and compares all hidden-layer and output-layer gradient regions against `nrc::reference_backward_khr_fp16`. On the current local AMD driver, KHR gradient pipeline creation raises the same structured exception as KHR inference and the test reports a controlled `SKIP:`.

`tests/vulkan_training_step_test.cpp` runs the gradient pass between `NrcTrainPrepare` and `NrcOptimize`, using prepare's indirect dispatch command and comparing the final optimizer state and packed weights against the CPU reference path.

`sample/path_tracer` dispatches the same scalar gradient pass with sample scene unpacking after scalar inference has resolved train-record targets.

`tests/encoding_test.cpp` validates the host-side `nrc::encode_input` mirror that the Vulkan gradient test uses to match the shader-side `NrcEncodeInput` path.

## Current Limitations

- The pass uses scalar loops and one invocation per train record. It is not a production-performance path.
- It uses fp32 storage-buffer atomic add and therefore requires `VK_EXT_shader_atomic_float` / `shaderBufferFloat32AtomicAdd`.
- The Vulkan gradient test is feature-gated and reports a controlled skip on devices without fp32 storage-buffer atomic add support.
- The sample path tracer currently runs one validated scalar training update, not an iterative training loop.
- The KHR cooperative-matrix pass now has full hidden/output `dW` training groundwork and scalar workgroup reduction for output-layer `dW`, but hidden-layer cooperative-matrix reduction and runtime numerical validation on a successful KHR pipeline creation path are still missing.

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
- output-layer `dW` is reduced across the workgroup and accumulated into the same fp32 row-major gradient layout as the scalar pass;
- output derivative is propagated through the output weights to final hidden activations;
- hidden-layer 4 `dW` is accumulated into the same fp32 row-major gradient layout as the scalar pass;
- hidden-layer 4 derivative is propagated to hidden-layer 3 activations;
- hidden-layer 3 `dW` is accumulated into the same fp32 row-major gradient layout as the scalar pass;
- hidden-layer 3 derivative is propagated to hidden-layer 2 activations;
- hidden-layer 2 `dW` is accumulated into the same fp32 row-major gradient layout as the scalar pass;
- hidden-layer 2 derivative is propagated to hidden-layer 1 activations;
- hidden-layer 1 `dW` is accumulated into the same fp32 row-major gradient layout as the scalar pass;
- hidden-layer 1 derivative is propagated to hidden-layer 0 activations;
- hidden-layer 0 `dW` is accumulated from the encoded input into the same fp32 row-major gradient layout as the scalar pass.

Current limitations:

- output-layer `dW` uses a scalar workgroup reduction with one fp32 atomic add per output weight per workgroup;
- hidden-layer `dW` helpers are still scalar per sample invocation and are not the final cooperative-matrix/subgroup reduction path;
- KHR runtime gradient comparison is present but remains driver-gated on the current local AMD pipeline creation path.
