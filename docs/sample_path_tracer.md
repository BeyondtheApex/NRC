# Reference Path Tracer Sample

`sample/path_tracer` is the reference integration target for the reusable NRC library.

The current sample is a minimal Vulkan integration skeleton with a tiny primary-ray triangle tracer. It is not a full path tracer yet, but it exercises the same seams the full sample will use:

1. Queue reset.
2. Eval-record production.
3. Train-record production.
4. Scalar inference.
5. Screen and train-record resolve.
6. Host-side NRC training plan selection.
7. Scalar training update.
8. Visual output write.
9. Readback and artifact validation.

## Current Files

- Host app:
  - `sample/path_tracer/sample.cpp`
- Producer shader:
  - `sample/path_tracer/shaders/sample_producer.comp`
- Inference shader wrapper:
  - `sample/path_tracer/shaders/sample_inference_scalar.comp`
- Training shader wrappers:
  - `sample/path_tracer/shaders/sample_train_prepare.comp`
  - `sample/path_tracer/shaders/sample_gradient_scalar.comp`
  - `sample/path_tracer/shaders/sample_optimize.comp`
- Shared sample scene shader:
  - `sample/path_tracer/shaders/sample_scene.glsl`
- Build config template:
  - `sample/path_tracer/sample_config.hpp.in`

## Current Runtime Flow

The sample creates:

- a queue state buffer;
- an eval-record buffer;
- a train-record buffer using the full flat batch layout expected by reusable training passes;
- a small scene buffer containing two triangles with per-vertex position, normal, roughness, diffuse, and specular data;
- packed fp16 weights;
- packed fp16 use-weights;
- fp32 optimizer state and entries;
- fp32 gradients;
- a dispatch-indirect command buffer;
- a `4x4` `rgba32f` bias/factor-R image;
- a `4x4` `rg32f` factor-GB image;
- a readback buffer.
- a binary PPM output image:
  - `build/sample/path_tracer/nrc_reference_path_tracer.ppm`

Before recording commands, the sample builds a `nrc::vulkan::VulkanTrainingProfilePlan` from the selected physical device capabilities. It then records one command buffer:

1. Upload the initial screen images.
2. Dispatch `sample_producer.comp` in reset mode.
3. Dispatch `sample_producer.comp` in producer mode, emitting screen eval records plus one train eval record.
4. Dispatch `sample_inference_scalar.comp`.
5. Dispatch `sample_train_prepare.comp`.
6. Dispatch `sample_gradient_scalar.comp` through the prepare pass's indirect command.
7. Dispatch `sample_optimize.comp`.
8. Copy the resolved screen image back to the CPU and read back train records, gradients, optimizer entries, and packed weights.
9. Write a tone-mapped binary PPM image from the resolved screen readback.

The training plan is currently diagnostic/integration-facing. The sample explicitly allows scalar reference fallback, prints the selected plan, validates that a compile config exists, and still executes the scalar training shaders. On the current local machine this reports scalar reference fallback because KHR training is blocked by the cooperative-matrix training shared-memory requirement.

The producer traces one primary ray for every `4x4` pixel, emits a screen eval record for each hit, reserves two train records in batch 0, and emits one train eval record targeting those records. Train-record bias and factor values come from a deterministic direct-lighting payload over the traced sample scene hit.

`sample_scene.glsl` builds a camera ray, intersects it against the two sample triangles, packs `primitiveId`, instance id, barycentrics, and scattered direction into `PackedNrcInput`, then unpacks the same payload into `NrcUnpackedInput` for scalar inference and scalar gradient training.

Weights are configured so the scalar MLP predicts:

```text
vec3(0.25, 0.5, 0.75)
```

The host verifies:

- a host-side training plan was selected and produced a valid shader compile config;
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
- training prepare advances optimizer state and writes the expected indirect dispatch command;
- scalar gradient matches the CPU reference over the sample scene unpacking path;
- optimizer entries match the CPU Adam/EMA reference;
- packed weights and packed use-weights match the CPU reference and at least one packed weight changes.
- the PPM output exists, matches the readback-derived bytes, has the expected `P6 4 4 255` header, and contains visible non-uniform pixels.

## Build and Test

The sample builds when:

- `nrc_vulkan` is available;
- `glslc` is available;
- `VK_EXT_shader_atomic_float` / `shaderBufferFloat32AtomicAdd` is available for the scalar gradient pass.

CTest runs:

```text
nrc_sample_reference_path_tracer_smoke
```

The host sample queries device capabilities through `nrc::vulkan::query_device_capabilities`, builds a training plan through `nrc::vulkan::make_vulkan_training_profile_plan`, creates shader modules through `nrc::vulkan::create_shader_module`, and creates all five active compute pipelines through plain `nrc::vulkan::create_compute_pipeline`.

It also creates descriptor set layouts through `nrc::vulkan::create_descriptor_set_layout` and writes storage-buffer/storage-image descriptors through `nrc::vulkan::update_storage_buffer_descriptors` and `nrc::vulkan::update_storage_image_descriptors`.

Memory and image barriers are recorded through `nrc::vulkan::cmd_memory_barrier`, `nrc::vulkan::color_image_barrier`, and `nrc::vulkan::cmd_pipeline_barrier`.

## Current Limitations

- This is a primary-ray triangle-loop integration skeleton, not a full path tracer.
- No acceleration structure, Vulkan ray query, material system, BRDF sampling, secondary rays, or scene loading exists yet.
- Training records use deterministic direct lighting, not real multi-bounce path-traced samples.
- Buffer/image copy helpers and resource allocation remain sample-local.
- KHR cooperative matrix inference is not used because KHR pipeline creation is driver-gated on the current AMD machine.
- Training plan selection is wired for integration diagnostics, but execution still uses the scalar correctness path rather than a production-performance KHR cooperative-matrix backend.

## Next Steps

The next sample milestones should add:

- replacing the sample-local triangle loop with Vulkan ray query or a fuller path tracing stage;
- train-record production from multi-bounce sampled paths;
- iterative per-frame training instead of a single validated update;
- a larger inspection output once the sample moves beyond the `4x4` smoke image.
