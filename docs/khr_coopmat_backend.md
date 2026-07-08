# KHR Cooperative Matrix Backend

The first KHR backend is `shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl`.

## Current Scope

Implemented:

- Forward inference only.
- 64 input values.
- 5 hidden ReLU layers of width 64.
- 3 output values.
- FP16 cooperative matrices.
- 16x16x16 subgroup matrices.
- Workgroup size 128.
- Subgroup size 16 or 32, selected by `NRC_KHR_SUBGROUP_SIZE`.

Not implemented yet:

- 8x16x16 profile.
- Backpropagation.
- Gradient accumulation.
- Optimizer.
- Runtime numerical validation.

## Weight Layout

The backend consumes the same packed fp16 `uvec4` layout documented in `weight_layout.md`.

Hidden layers are loaded as full 64x64 matrices.

The output layer uses a 16x64 cooperative matrix tile. Only the first 3 rows are populated from the packed 3x64 output weights; remaining rows are zero-filled.

## Compile Tests

CTest currently compiles the backend twice:

```text
NRC_KHR_SUBGROUP_SIZE=32
NRC_KHR_SUBGROUP_SIZE=16
```

These tests prove that the GLSL syntax, cooperative matrix types, and include structure are valid for the local Vulkan SDK. They do not prove runtime support or numerical correctness.

## Profile Selection

Profiles now carry an `implemented` flag. The 16x16x16 KHR SG16/SG32 profiles are implemented. The M8 profile remains in the profile list as a planned target but is not selected by `choose_profile` yet.
