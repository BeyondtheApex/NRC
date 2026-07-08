# Scalar Gradient Pass

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

`tests/reference_gradient_test.cpp` validates the CPU reference backpropagation against finite differences. A future Vulkan dispatch test should compare this shader's gradient buffer against the CPU reference.

## Current Limitations

- The pass uses scalar loops and one invocation per train record. It is not a production-performance path.
- It uses fp32 storage-buffer atomic add and therefore requires `VK_EXT_shader_atomic_float` / `shaderBufferFloat32AtomicAdd`.
- Runtime feature gating and Vulkan numerical validation are still pending.
- KHR cooperative-matrix gradient/backpropagation is not implemented yet.
