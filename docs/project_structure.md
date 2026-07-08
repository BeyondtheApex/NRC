# Project Structure

This repository separates reusable NRC code from the reference renderer.

```text
include/nrc/
  Public host-side ABI, profile descriptions, constants, encoding mirror, and resource layouts.

shaders/nrc/common/
  Shader-side ABI and algorithm helpers shared by integration shaders and library passes.

shaders/nrc/backend/
  Neural-network execution backends. KHR cooperative matrix is the first production target.

shaders/nrc/passes/
  Standard shader-header passes: indirect dispatch generation, inference, train preparation, gradient, and optimizer.

runtime/vulkan/
  Optional Vulkan helper implementation for applications that want a ready-made runtime wrapper.

sample/path_tracer/
  Reference integration sample used to validate NRC behavior. It currently uses a tiny primary-ray Vulkan pipeline with scalar inference, one scalar training update, and a validated PPM output artifact, and will grow into a full path tracer.

tests/
  ABI, profile selection, shader compilation, encoding, queue behavior, NN reference tests, GPU dispatch validation, and full scalar training-step validation.

docs/
  Implementation log, design notes, integration guide, training notes, and milestone results.
```

Downstream projects should be able to use only `include/nrc`, `shaders/nrc`, and optionally `runtime/vulkan`.
