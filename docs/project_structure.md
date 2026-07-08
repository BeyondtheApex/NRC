# Project Structure

This repository separates reusable NRC code from the reference renderer.

```text
include/nrc/
  Public host-side ABI, profile descriptions, constants, and resource layouts.

shaders/nrc/common/
  Shader-side ABI and algorithm helpers shared by integration shaders and library passes.

shaders/nrc/backend/
  Neural-network execution backends. KHR cooperative matrix is the first production target.

shaders/nrc/passes/
  Standard compute passes: indirect dispatch generation, inference, train preparation, gradient, and optimizer.

runtime/vulkan/
  Optional Vulkan helper implementation for applications that want a ready-made runtime wrapper.

sample/path_tracer/
  Reference path tracing pipeline used to validate NRC behavior. It is not part of the reusable core.

tests/
  ABI, profile selection, shader compilation, queue behavior, NN reference tests, and GPU dispatch validation.

docs/
  Implementation log, design notes, integration guide, and milestone results.
```

Downstream projects should be able to use only `include/nrc`, `shaders/nrc`, and optionally `runtime/vulkan`.
