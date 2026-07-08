# Reference Path Tracer Sample

`sample/path_tracer` is the reference integration target for the reusable NRC library.

Milestone 12 replaces the placeholder executable with a minimal Vulkan integration skeleton. It is not a full path tracer yet, but it exercises the same seams the full sample will use:

1. Queue reset.
2. Eval-record production.
3. Scalar inference.
4. Screen image resolve.
5. Readback validation.

## Current Files

- Host app:
  - `sample/path_tracer/sample.cpp`
- Producer shader:
  - `sample/path_tracer/shaders/sample_producer.comp`
- Inference shader wrapper:
  - `sample/path_tracer/shaders/sample_inference_scalar.comp`
- Build config template:
  - `sample/path_tracer/sample_config.hpp.in`

## Current Runtime Flow

The sample creates:

- a queue state buffer;
- an eval-record buffer;
- a small train-record buffer placeholder;
- packed fp16 weights;
- a `4x4` `rgba32f` bias/factor-R image;
- a `4x4` `rg32f` factor-GB image;
- a readback buffer.

It records one command buffer:

1. Upload the initial screen images.
2. Dispatch `sample_producer.comp` in reset mode.
3. Dispatch `sample_producer.comp` in producer mode.
4. Dispatch `sample_inference_scalar.comp`.
5. Copy the resolved screen image back to the CPU.

The producer emits one screen eval record for every `4x4` pixel.

The inference wrapper provides a deterministic fake scene unpack function and uses `NrcInferenceScalar.glsl`.

Weights are configured so the scalar MLP predicts:

```text
vec3(0.25, 0.5, 0.75)
```

The host verifies:

- `evalCount == 16`;
- every pixel has exactly one eval record;
- train counts remain zero;
- every screen pixel was resolved as `bias + factor * prediction`;
- output alpha is zero, matching the inference pass contract.

## Build and Test

The sample builds when:

- `nrc_vulkan` is available;
- `glslc` is available.

CTest runs:

```text
nrc_sample_reference_path_tracer_smoke
```

## Current Limitations

- This is a synthetic integration skeleton, not a real path tracer.
- No acceleration structure, ray query, material system, BRDF sampling, or scene loading exists yet.
- The scene unpack function is deterministic fake data.
- Training records are bound only as a placeholder because the current sample emits screen eval records only.
- KHR cooperative matrix inference is not used because KHR pipeline creation is driver-gated on the current AMD machine.
- No gradient/backpropagation or optimizer pass is connected yet.

## Next Steps

The next sample milestones should add:

- a small scene representation and real `PackedNrcInput` generation;
- direct lighting or simple path tracing;
- train-record production;
- scalar gradient and optimizer once those passes exist;
- visual output artifacts for inspection.
