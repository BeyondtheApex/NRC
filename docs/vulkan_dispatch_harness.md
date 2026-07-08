# Vulkan Dispatch Harness

Milestone 7 adds the first GPU runtime validation path. The harness is intentionally test-scoped: it proves the shader ABI, packed buffers, descriptors, push constants, command recording, and CPU/GPU numerical comparison before the reusable runtime wrapper is designed.

## Current Scope

- Test file:
  - `tests/vulkan_scalar_dispatch_test.cpp`
- Dispatch shader:
  - `tests/shader_compile/nrc_scalar_dispatch.comp`
- Backend under test:
  - `shaders/nrc/backend/scalar/NrcNNScalar.glsl`

The test uploads:

- Packed fp16 weights in the library weight layout.
- Packed fp16 64-wide encoded inputs.
- A storage output buffer containing one `vec4` per sample.

It then dispatches one invocation per sample and compares the GPU result against `nrc::reference_forward`.

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

## Next Uses

The same pattern should be reused for:

- KHR cooperative matrix numerical validation on devices that support a selected implemented profile.
- Inference pass validation once `NrcEvalRecord` queue consumption exists.
- Regression tests for future backend layout changes.

The final runtime helper should not copy this test harness wholesale. It should promote only the reusable parts: shader module loading, descriptor contract creation, resource barriers, and profile-specific pipeline creation.
