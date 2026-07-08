# KHR Shared Memory Layout

`nrc::compute_khr_shader_shared_memory_layout` mirrors the compile-time shared-memory constants used by `shaders/nrc/backend/khr_coopmat/NrcNNKhr.glsl`.

Header:

```cpp
#include "nrc/nrc_khr_shader_layout.hpp"
```

The helper consumes a `ProfileShaderConfig` and returns:

- active activation tile size;
- saved activation snapshot regions;
- dPre scratch region;
- output derivative and output dW partial regions;
- hidden dW partial region size;
- total shared-memory requirement in `uvec4` units and bytes;
- boolean lifetime checks used by tests and future profile diagnostics.

## Current Profile Shape

All implemented KHR profiles use:

```text
workgroup size: 128
hidden width:   64
N:              16
K:              16
```

The active activation tile is:

```text
64 values/sample * 128 samples / 8 fp16-per-uvec4 = 1024 uvec4
1024 uvec4 * 16 bytes = 16384 bytes
```

Backpropagation keeps six activation-sized regions:

```text
active tile                              [0,    1024)
final hidden snapshot                    [1024, 2048)
output source snapshot                   [2048, 3072)
previous source snapshot                 [3072, 4096)
earlier source snapshot                  [4096, 5120)
oldest source snapshot                   [5120, 6144)
```

The current KHR backprop shader therefore allocates:

```text
6144 uvec4 = 98304 bytes
```

Vulkan capability query reports `maxComputeSharedMemorySize`, and KHR training profile support checks this value against the layout's `sharedBytes`. Inference-only profile selection skips this training-only backprop allocation.

## Scratch Regions

The fp32 dPre scratch is:

```text
128 samples * (64 fp32 / 4 fp32-per-uvec4) = 2048 uvec4
```

It intentionally occupies the active tile plus the retired final-hidden snapshot region.

The optional output-layer cooperative-matrix dW path uses:

```text
fp16 dOutput staging:
128 samples * (16 fp16 / 8 fp16-per-uvec4) = 256 uvec4

output dW partials:
SG32:  512 uvec4
SG16: 1024 uvec4
M8/SG32: 512 uvec4
```

These output regions fit inside the active tile. After reducing output partials, the helper restores the fp32 shared dOutput cache because final-hidden dW reads it.

## Hidden dW Constraint

Hidden dW cooperative-matrix partial storage would require:

```text
64 * 64 fp16 values / 8 fp16-per-uvec4 * subgroupCount
```

For current profiles:

```text
SG32:    subgroupCount 4 -> 2048 uvec4
SG16:    subgroupCount 8 -> 4096 uvec4
M8/SG32: subgroupCount 4 -> 2048 uvec4
```

All fit inside the current 6144-uvec4 allocation, but not all are safe to place at offset zero:

- SG32 and M8/SG32 end at `2048`, preserving the first shared hidden activation snapshot at `2048`.
- SG16 ends at `4096`, which would overwrite activation snapshots still needed by later dA propagation.

Hidden-layer cooperative-matrix dW therefore needs a deliberate memory-lifetime strategy before implementation. Options include:

- reorder dA propagation before hidden dW partial storage;
- reduce one tile at a time with less partial storage;
- use a different cross-subgroup reduction strategy;
- add a profile-gated larger shared-memory layout.

## Hidden dW Tile-Streaming Plan

`KhrShaderSharedMemoryLayout::hiddenDWeightTileStreaming` records the conservative plan for a future hidden-layer cooperative-matrix `dW` helper.

Instead of staging a full 64x64 hidden-layer partial per subgroup, the plan stages one row block of fp16 `dPre` plus one MxN `dW` tile partial per subgroup inside the active tile:

```text
dPre row-block scratch: M * 128 fp16 values
partial tile scratch:   M * N fp16 values * subgroupCount
```

For the current profiles:

```text
M16/SG32: dPre row 256 uvec4 + partials 128 uvec4 = 384 uvec4
M16/SG16: dPre row 256 uvec4 + partials 256 uvec4 = 512 uvec4
M8/SG32:  dPre row 128 uvec4 + partials  64 uvec4 = 192 uvec4
```

All of these fit within the 1024-uvec4 active tile and end before the first activation snapshot at offset 1024. That makes the plan stable across SG32, SG16, and M8/SG32 without overwriting the saved hidden activations needed by later dA propagation.

The tradeoff is that the shader will need to stream/reduce tiles rather than keep full-layer partials resident in shared memory. This is the intended next implementation shape unless runtime profiling proves a larger profile-gated shared-memory layout is worthwhile.

`NrcKhrAccumulateHiddenDWeightsCoopmatTileStreaming` is the first shader helper using this plan. It is compile-validated for every hidden-layer `dW` region and for the all-`dW` combined mode through explicit KHR gradient switches, with matching runtime validation entries and test-side CPU oracles; it is not yet the default gradient path.

## Hidden dA Cooperative-Matrix Scratch

`NrcKhrPropagateHiddenDActivationCoopmat` is the first experimental cooperative-matrix `dA` helper. The currently wired switch is:

```text
NRC_KHR_USE_COOPMAT_FINAL_HIDDEN_DA=1
NRC_KHR_USE_COOPMAT_HIDDEN3_DA=1
NRC_KHR_USE_COOPMAT_HIDDEN2_DA=1
```

The helper uses existing shared-memory regions:

```text
full dPre staging:           64 * 128 fp16 values = 1024 uvec4
source-derivative output:    64 * 128 fp16 values = 1024 uvec4
```

The full `dPre` staging occupies the active tile at offset 0. The fp16 source-derivative output is stored into the retired activation snapshot for the hidden layer whose `dA` is being propagated:

```text
final-hidden -> hidden3: NRC_KHR_BACKPROP_ACT_COPY_UVEC4_BASE
hidden3 -> hidden2:      NRC_KHR_BACKPROP_SOURCE_ACT_COPY_UVEC4_BASE
hidden2 -> hidden1:      NRC_KHR_BACKPROP_PREVIOUS_SOURCE_ACT_COPY_UVEC4_BASE
```

This is safe for the currently wired propagation points because the corresponding hidden-layer `dW` accumulation has already consumed that layer's activation snapshot before the source derivative is produced.

This path does not increase the 6144-uvec4 / 98304-byte KHR training shared-memory requirement. Extending the same approach to hidden1-to-hidden0 should similarly reuse the activation snapshot whose ReLU mask has just been consumed, but the remaining step still needs explicit lifetime validation before being enabled.
