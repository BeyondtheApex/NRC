# Reference Path Tracer Sample

`sample/path_tracer` is the reference integration target for the reusable NRC library.

The current sample is a minimal Vulkan integration skeleton with a tiny primary-ray triangle tracer. It is not a full path tracer yet, but it exercises the same seams the full sample will use:

1. Queue reset.
2. Eval-record production.
3. Train-record production.
4. Scalar inference.
5. Screen and train-record resolve.
6. Readback validation.

## Current Files

- Host app:
  - `sample/path_tracer/sample.cpp`
- Producer shader:
  - `sample/path_tracer/shaders/sample_producer.comp`
- Inference shader wrapper:
  - `sample/path_tracer/shaders/sample_inference_scalar.comp`
- Shared sample scene shader:
  - `sample/path_tracer/shaders/sample_scene.glsl`
- Build config template:
  - `sample/path_tracer/sample_config.hpp.in`

## Current Runtime Flow

The sample creates:

- a queue state buffer;
- an eval-record buffer;
- a small train-record buffer;
- a small scene buffer containing two triangles with per-vertex position, normal, roughness, diffuse, and specular data;
- packed fp16 weights;
- a `4x4` `rgba32f` bias/factor-R image;
- a `4x4` `rg32f` factor-GB image;
- a readback buffer.

It records one command buffer:

1. Upload the initial screen images.
2. Dispatch `sample_producer.comp` in reset mode.
3. Dispatch `sample_producer.comp` in producer mode, emitting screen eval records plus one train eval record.
4. Dispatch `sample_inference_scalar.comp`.
5. Copy the resolved screen image back to the CPU and read back train records.

The producer traces one primary ray for every `4x4` pixel, emits a screen eval record for each hit, reserves two train records in batch 0, and emits one train eval record targeting those records. Train-record bias and factor values come from a deterministic direct-lighting payload over the traced sample scene hit.

`sample_scene.glsl` builds a camera ray, intersects it against the two sample triangles, packs `primitiveId`, instance id, barycentrics, and scattered direction into `PackedNrcInput`, then unpacks the same payload into `NrcUnpackedInput` for `NrcInferenceScalar.glsl`.

Weights are configured so the scalar MLP predicts:

```text
vec3(0.25, 0.5, 0.75)
```

The host verifies:

- `evalCount == 17`;
- every pixel has exactly one eval record;
- train batch 0 has two records and batches 1-3 remain zero;
- the train eval record targets batch 0 records `[0, 1]`;
- every screen and train eval payload matches the CPU reference `PackedNrcInput` generated from the same primary-ray triangle intersection rule;
- both train records preserve the expected packed scene input payload;
- every screen pixel was resolved as `bias + factor * prediction`;
- both train record bias values were resolved as `directLightingBias + directLightingFactor * prediction`;
- both train record factor values match the CPU reference direct-lighting payload;
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

- This is a primary-ray triangle-loop integration skeleton, not a full path tracer.
- No acceleration structure, Vulkan ray query, material system, BRDF sampling, secondary rays, or scene loading exists yet.
- Training records use deterministic direct lighting, not real multi-bounce path-traced samples.
- KHR cooperative matrix inference is not used because KHR pipeline creation is driver-gated on the current AMD machine.
- No gradient/backpropagation or optimizer pass is connected yet.

## Next Steps

The next sample milestones should add:

- replacing the sample-local triangle loop with Vulkan ray query or a fuller path tracing stage;
- train-record production from multi-bounce sampled paths;
- scalar gradient and optimizer once those passes exist;
- visual output artifacts for inspection.
