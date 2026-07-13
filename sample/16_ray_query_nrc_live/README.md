# 16 Ray Query NRC Live

This sample is a clean NRC integration target based on `raytrace_tutorial/16_ray_query`, with `G:/cpp/radiance/NRC` linked as a reusable library. It is intended as a downstream-renderer reference, not as a synthetic NRC unit test.

The active NRC path requires KHR cooperative matrix. Scalar fallback is disabled on purpose.

## Execution Flow

The sample follows the same high-level shape as `G:/cpp/radiance/VkNRC`:

1. The ray-query path tracer traces ordinary camera paths until `Handoff Depth`.
2. At the handoff point, screen paths write:
   - `bias`: radiance accumulated before the handoff;
   - `factor`: path throughput at the handoff;
   - one screen `NrcEvalRecord`;
   - one encoded NRC input.
3. A probabilistic subset of handoff paths also continues tracing real suffix paths for training.
4. Training paths write multiple suffix `NrcTrainRecord` entries into one of four batches.
5. If a training suffix is not fully terminated, it emits an `NRC_EVAL_DST_TRAIN` record so inference can bootstrap the train target.
6. KHR cooperative-matrix inference reads `useWeights`, consumes actual eval records, and writes:
   - screen color as `bias + factor * prediction`;
   - bootstrapped train targets.
7. Four train batches run:
   - train prepare clamps the real batch count and writes an indirect dispatch command;
   - KHR recompute gradient dispatches indirectly from that count;
   - optimizer updates `weights`;
   - the final batch writes stable `useWeights`.

## Modes

The UI exposes one mode selector:

- `Baseline`: full path tracing, no NRC record production.
- `Train Only`: collect real training records and update cooperative-matrix weights, but do not use NRC for the displayed image. This is the default so the first frame does not show random untrained network output.
- `Frozen NRC`: apply current `useWeights` to the displayed image without training.
- `Live NRC`: collect training records, apply NRC to the displayed image, and update weights in the same frame.

For visual comparison, train first, then switch between `Baseline`, `Frozen NRC`, and `Live NRC`.

When `Frozen NRC` or `Live NRC` applies predictions to the screen, the tutorial shader disables its built-in temporal accumulation and writes the current-frame low-bounce estimate before the NRC inference result is copied over it. This matches the important separation in `VkNRC`: the path tracer produces a current-frame `bias/factor` color image, and accumulation, if enabled, happens later in `ScreenPass` from the final color image. The final NRC image must not be read back as the next frame's path-tracing history.

## Dynamic Parameters

- `Handoff Depth`: the bounce where NRC predicts the remaining indirect radiance. Smaller values increase NRC coverage and reduce ray depth, but demand a better-trained network.
- `Train Sample Probability`: probability that an eligible handoff path also traces a real suffix for training. Higher values train faster and cost more ray work.
- Eval capacity is not a user quality knob. It is derived from resolution and train capacity through `nrc::compute_resource_layout()`.
- Train batch size/count come from the NRC library constants: `kTrainBatchSize = 16384`, `kTrainBatchCount = 4`.
- `useWeights` is separate from training `weights`. Optimizer writes EMA/raw inference weights after the last batch, so the next frame can infer from a stable buffer.

## Buffer Layout Convention

The current NRC record ABI stores a compact `PackedNrcInput` in records. This sample uses `PackedNrcInput::primitiveId` as an index into a separate encoded-input buffer:

- eval feature region: `[0, evalCapacity)`;
- train feature region: `[trainFeatureBase, trainFeatureBase + trainCapacity)`;
- `trainFeatureBase == evalCapacity`.

Keeping those regions separate is required. Earlier queue-polluting prototypes allocated train features from `evalCount`, which caused inference to scan stale records and produced unstable wrong-color speckle.

## Cooperative Matrix Policy

The sample hard-requires cooperative matrix:

- `TrainingProfileSelectionOptions::allowScalarReferenceFallback = false`;
- device creation enables `VK_KHR_cooperative_matrix` plus the selected profile's float16/storage/subgroup requirements;
- inference and gradient use KHR cooperative-matrix SPIR-V compiled with `glslc -O`;
- KHR shader-module or compute-pipeline creation failure aborts NRC initialization.

The known stable local profile is `KHR_FP16_M16N16K16_SG32_WG128` using KHR recompute backprop with cooperative-matrix `dW` and `dA`.

## Validation

Minimum validation for this sample:

```powershell
& cmd.exe /d /c '"D:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 -no_logo && set "VULKAN_SDK=D:\VulkanSDK\1.4.350.0" && "D:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build_vs --config Release --target 16_ray_query_nrc_live -j 8'

& cmd.exe /d /c 'set "VULKAN_SDK=D:\VulkanSDK\1.4.350.0" && set "PATH=D:\VulkanSDK\1.4.350.0\Bin;%PATH%" && _bin\Release\16_ray_query_nrc_live.exe --headless'
```

Expected evidence:

- KHR inference shader module: `Passed`;
- KHR inference pipeline: `Passed`;
- KHR gradient shader module: `Passed`;
- KHR gradient pipeline: `Passed`;
- headless stats show non-zero train counts in `Train Only` mode;
- optimizer `step` advances;
- `weightDelta` changes from zero.
- `NRC GPU ms` prints per-pass GPU timestamp measurements.

Current local Release headless evidence on the AMD test machine:

```text
NRC resources: 2048x1152 evalCapacity=2424832 trainCapacity=65536 trainFeatureBase=2424832
NRC KhrRecompute/KHR_FP16_M16N16K16_SG32_WG128: eval=8317/2424832 train0=4576/16384 step=4 firstWeight=0.189026 delta=0.006892
NRC GPU ms: total=35.428 reset=0.041 path=2.022 inferPrep=0.002 infer=0.185 trainPrep=0.029 gradient=32.896 optimize=0.253 copy=0.000
```

This is not a final performance claim. It shows where the current sample spends time: KHR recompute gradient dominates the default `Train Only` frame; inference itself is small in this scene/run.

## Performance Comparison Plan

Do not compare NRC with CPU frame wall time alone. This sample now reports GPU timestamps around:

- ray-query path tracing and record production;
- KHR inference;
- train prepare;
- KHR gradient;
- optimizer;
- final copy/resolve.

The useful question is quality per millisecond:

- high-sample/deep baseline as reference;
- realtime baseline with the same per-frame sample budget;
- NRC with the same direct/low-bounce budget and predicted high-bounce tail.

Useful image metrics are RMSE/MAE against the high-sample reference, plus visual side-by-side.

## Current Limits

- The copy pass named `visualize` only transfers the NRC inference image into the tutorial output image. It is not a production screen pass.
- `VkNRC` owns a separate screen accumulation image and resets its accumulation count on mode/camera/NRC-state changes. This sample currently avoids feedback pollution by disabling the tutorial accumulation while NRC is applied, rather than adding a full `VkNRC`-style screen pass.
- Numeric image-error comparison is still pending.
- The sample's material/lighting model is the tutorial ray-query model, not the full VkNRC renderer.
