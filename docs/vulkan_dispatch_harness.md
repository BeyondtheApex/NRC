# Vulkan Dispatch Harness

Milestone 7 adds the first GPU runtime validation path. The harness is intentionally test-scoped: it proves the shader ABI, packed buffers, descriptors, push constants, command recording, and CPU/GPU numerical comparison before the reusable runtime wrapper is designed.

## Current Scope

- Scalar test file:
  - `tests/vulkan_scalar_dispatch_test.cpp`
- Scalar dispatch shader:
  - `tests/shader_compile/nrc_scalar_dispatch.comp`
- Scalar backend under test:
  - `shaders/nrc/backend/scalar/NrcNNScalar.glsl`
- KHR test file:
  - `tests/vulkan_khr_dispatch_test.cpp`
- KHR dispatch shader:
  - `tests/shader_compile/nrc_khr_dispatch.comp`
- KHR backend under test:
  - `shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl`
- KHR gradient test file:
  - `tests/vulkan_gradient_khr_test.cpp`
- KHR gradient shader:
  - `tests/shader_compile/nrc_gradient_khr_dispatch.comp`
- Optional KHR output-coopmat gradient mode:
  - `nrc_vulkan_gradient_khr_coopmat_output_test`
  - SPIR-V variants compiled with `NRC_KHR_USE_COOPMAT_OUTPUT_DW=1`
- Optional KHR final-hidden-coopmat gradient mode:
  - `nrc_vulkan_gradient_khr_coopmat_final_hidden_test`
  - SPIR-V variants compiled with `NRC_KHR_USE_COOPMAT_FINAL_HIDDEN_DW=1`
- Optional KHR hidden-layer-3-coopmat gradient mode:
  - `nrc_vulkan_gradient_khr_coopmat_hidden3_test`
  - SPIR-V variants compiled with `NRC_KHR_USE_COOPMAT_HIDDEN3_DW=1`
- Optional KHR hidden-layer-2-coopmat gradient mode:
  - `nrc_vulkan_gradient_khr_coopmat_hidden2_test`
  - SPIR-V variants compiled with `NRC_KHR_USE_COOPMAT_HIDDEN2_DW=1`
- Scalar inference test file:
  - `tests/vulkan_inference_scalar_test.cpp`
- Scalar inference shader wrapper:
  - `tests/shader_compile/nrc_inference_scalar_smoke.comp`
- Scalar inference pass under test:
  - `shaders/nrc/passes/NrcInferenceScalar.glsl`

The test uploads:

- Packed fp16 weights in the library weight layout.
- Packed fp16 64-wide encoded inputs.
- A storage output buffer containing one `vec4` per sample.

The scalar test dispatches one invocation per sample and compares the GPU result against `nrc::reference_forward`.

The KHR test dispatches one 128-wide workgroup tile per sample block. It first validates the selected KHR profile at device and pipeline creation time, then compares against an fp16-rounded CPU reference if the driver successfully creates and runs the pipeline.

The KHR gradient test selects a KHR training profile, which additionally requires fp32 storage-buffer atomic add, subgroup arithmetic, and enough compute shared memory for the current KHR backprop layout. If the driver successfully creates the profile pipeline, it dispatches the KHR train-record gradient pass over a 67-record one-workgroup batch and compares the gradient buffer against the selected CPU oracle. The optional output-coopmat, final-hidden-coopmat, hidden-layer-3-coopmat, and hidden-layer-2-coopmat modes replace only the corresponding region in the expected gradient buffer, with expected partials rounded per subgroup before cross-subgroup reduction.

The scalar inference test dispatches `NrcInferenceScalar.glsl` against real storage images and a flat train-record buffer. It verifies queue destination decoding and `bias + factor * prediction` writes for both screen and train destinations.

## Comparison Policy

CPU comparison uses the fp16-rounded view of the same data that the shader sees:

- CPU weights are packed to fp16 and unpacked back to float before `reference_forward`.
- CPU inputs are fp16-rounded before both packing and reference evaluation.

This keeps the test focused on shader/runtime correctness rather than expected float32-vs-fp16 differences.

## Runtime Behavior

- The dispatch shader uses `local_size_x = 64`.
- The test dispatches 73 samples to verify non-multiple-of-workgroup guards.
- The test uses host-visible coherent storage buffers to avoid adding staging/copy complexity at this stage.
- If no Vulkan 1.1 compute device or compatible host-visible coherent memory type is available, the test prints `SKIP:` and returns success.
- The scalar dispatch shader module and non-profile compute pipeline use `nrc::vulkan::create_shader_module` plus plain `nrc::vulkan::create_compute_pipeline`.
- The scalar dispatch descriptor set layout/update path uses `nrc::vulkan::create_storage_buffer_descriptor_set_layout` and `nrc::vulkan::update_storage_buffer_descriptors`.
- The scalar dispatch host-to-shader and shader-to-host memory barriers use `nrc::vulkan::cmd_memory_barrier`.

KHR-specific behavior:

- The KHR dispatch shader uses `local_size_x = NRC_KHR_WORKGROUP_SIZE`, currently 128.
- The test dispatches 149 samples to verify padded final workgroups.
- The test selects SG32, SG16, or M8/SG32 SPIR-V through `nrc::find_profile_shader_config` after choosing an implemented profile from queried capabilities.
- The KHR dispatch descriptor set layout/update path uses the same reusable storage-buffer descriptor helpers as the scalar dispatch test.
- The KHR dispatch host-to-shader and shader-to-host memory barriers use the same reusable memory barrier helper as the scalar dispatch test.
- The KHR shader module is created through `nrc::vulkan::create_shader_module` and fed into `nrc::vulkan::evaluate_vulkan_profile_validation`.
- The KHR pipeline is created through `nrc::vulkan::create_profile_compute_pipeline`, which applies the selected subgroup size and full compute subgroup requirements from `ProfileShaderConfig`.
- On MSVC/Windows, the test catches structured exceptions from `vkCreateComputePipelines` and reports a controlled `SKIP:`. This is needed because the current local AMD driver advertises the profile but raises an exception while compiling the VkNRC-style KHR cooperative matrix shader.
- The KHR gradient runtime tests use the same profile pipeline path with `ProfileRequirements::requireTraining = true`; on the current local AMD driver the default gradient path, optional output-coopmat path, optional final-hidden-coopmat path, optional hidden-layer-3-coopmat path, and optional hidden-layer-2-coopmat path report a controlled profile-selection `SKIP:` because no implemented KHR training profile satisfies the current 98304-byte shared-memory requirement.
- Reusable runtime diagnostics now live in `nrc::vulkan::evaluate_vulkan_profile_validation`, shader module creation lives in `nrc::vulkan::create_shader_module`, descriptor helpers live in `nrc::vulkan::create_descriptor_set_layout`, `create_storage_buffer_descriptor_set_layout`, `update_storage_buffer_descriptors`, and `update_storage_image_descriptors`, memory barriers live in `nrc::vulkan::cmd_memory_barrier`, plain compute pipeline creation lives in `nrc::vulkan::create_compute_pipeline`, and profile-aware compute pipeline creation lives in `nrc::vulkan::create_profile_compute_pipeline`. The dispatch harness still owns shader file loading, descriptor pools/sets, dispatch, and numerical comparison.

Inference-specific behavior:

- The scalar inference shader module and non-profile compute pipeline use `nrc::vulkan::create_shader_module` plus plain `nrc::vulkan::create_compute_pipeline`.
- The scalar inference descriptor set layout/update path uses `nrc::vulkan::create_descriptor_set_layout`, `nrc::vulkan::update_storage_buffer_descriptors`, and `nrc::vulkan::update_storage_image_descriptors`.
- The scalar inference image layout transitions use `nrc::vulkan::color_image_barrier` and `nrc::vulkan::cmd_pipeline_barrier`.
- The scalar inference test creates optimal-tiled `rgba32f` and `rg32f` storage images.
- It uploads images through staging buffers, dispatches compute in `VK_IMAGE_LAYOUT_GENERAL`, then copies the screen output image back for verification.
- It also verifies shader writes to the host-visible train-record buffer.

Training/queue-specific behavior:

- Queue ops, scalar dispatch, scalar gradient, and scalar full-training-step tests now use `nrc::vulkan::create_shader_module` plus plain `nrc::vulkan::create_compute_pipeline` for their non-profile compute pipelines. KHR gradient uses `nrc::vulkan::create_shader_module` plus `nrc::vulkan::create_profile_compute_pipeline`.
- Queue ops, scalar gradient, and scalar full-training-step tests now use `nrc::vulkan::create_storage_buffer_descriptor_set_layout` and `nrc::vulkan::update_storage_buffer_descriptors` for their storage-buffer descriptor set layout/update paths.
- Queue ops, scalar gradient, KHR gradient, and scalar full-training-step tests now route their host/compute memory barriers through `nrc::vulkan::cmd_memory_barrier`.
- The full-training-step test also uses the helper for its compute-to-indirect-dispatch dependency.

## Next Uses

The same pattern should be reused for:

- KHR cooperative matrix numerical validation on devices that can create and run the selected implemented profile pipelines.
- Reference path tracer integration using the validated scalar inference pass.
- Regression tests for future backend layout changes.

The final runtime helper should not copy this test harness wholesale. It should continue promoting only reusable pieces: sample descriptor/barrier migration and dispatch/numerical validation scaffolding.
