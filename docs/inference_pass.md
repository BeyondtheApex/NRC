# Inference Pass

Milestone 9 adds the first queue-consuming inference pass header:

- `shaders/nrc/passes/NrcInferenceScalar.glsl`

This is a reusable shader header, not a fixed application pipeline. Integrations provide the final `.comp` wrapper, local size declaration, descriptor set choices, and scene unpack function.

## Contract

The pass consumes:

- Packed fp16 NN weights through `NrcNNScalar.glsl`.
- `NrcEvalRecord` entries.
- An eval count buffer.
- A screen output pair:
  - `rgba32f` image storing `bias.rgb` and `factor.r`.
  - `rg32f` readonly image storing `factor.g` and `factor.b`.
- A flat `NrcTrainRecord` buffer containing `NRC_TRAIN_BATCH_COUNT * NRC_TRAIN_BATCH_SIZE` records.

The pass requires the integration shader to define:

```glsl
NrcUnpackedInput NrcUnpackPackedInput(PackedNrcInput packedInput);
```

This mirrors VkNRC's split: the NRC library owns record formats, input encoding, queue destination decoding, and NN evaluation, while the renderer owns scene/material unpacking.

## Behavior

For each valid `NrcEvalRecord`:

1. Decode the packed renderer input through `NrcUnpackPackedInput`.
2. Encode it with `NrcEncodeInput`.
3. Run `NrcScalarForward`.
4. Clamp the prediction to non-negative values, matching VkNRC inference behavior.
5. If the destination is screen:
   - Load `bias` and `factor`.
   - Store `bias + factor * prediction`.
6. If the destination is train:
   - Decode batch/left/right.
   - Apply `bias + factor * prediction` to each referenced train record's bias.

Train records are modeled as one flat buffer. VkNRC used an array of batch buffers; the flat layout matches this repository's resource layout and is easier for downstream engines to bind without descriptor arrays.

## Current Validation

CTest compiles:

- `tests/shader_compile/nrc_inference_scalar_smoke.comp`

The smoke shader supplies a deterministic fake `NrcUnpackPackedInput` implementation. This validates GLSL syntax, descriptor declarations, queue destination logic, and scalar backend inclusion.

CTest also dispatches the pass on Vulkan through:

- `tests/vulkan_inference_scalar_test.cpp`

The Vulkan test creates real storage images and buffers:

- `rgba32f` bias/factor-R image.
- `rg32f` factor-GB image.
- packed fp16 weights.
- eval records.
- eval count.
- flat train-record buffer.

It uses deterministic weights that make the NN prediction constant, then verifies:

- screen destinations update `bias.rgb` with `bias + factor * prediction`;
- written screen alpha becomes `0.0`, matching the pass;
- train destinations update only the referenced train-record bias values;
- invalid eval destinations do not write.

## Current Limitations

- The pass is scalar-only; a KHR inference pass should be added after KHR pipeline creation is stable on at least one driver.
- The screen output contract follows VkNRC's bias/factor image model but has not yet been exercised in the reference path tracer.
- Scene/material unpacking remains integration-owned and must be implemented by the sample path tracer and any downstream renderer.
