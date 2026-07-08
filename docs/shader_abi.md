# Shader ABI

The shader ABI is intentionally split into small include files.

```text
shaders/nrc/common/NrcConstants.glsl
shaders/nrc/common/NrcTypes.glsl
shaders/nrc/common/NrcEncoding.glsl
shaders/nrc/common/NrcQueue.glsl
shaders/nrc/passes/NrcQueueOps.glsl
shaders/nrc/passes/NrcTrainPrepare.glsl
shaders/nrc/passes/NrcOptimize.glsl
```

## Records

The first record layout mirrors VkNRC.

```glsl
struct PackedNrcInput {
    uint primitiveId;
    uint flipBitInstanceId;
    uint barycentric2x16U;
    uint scatteredDir2x16U;
};

struct NrcEvalRecord {
    uint dst;
    PackedNrcInput packedInput;
};

struct NrcTrainRecord {
    float biasR;
    float biasG;
    float biasB;
    float factorR;
    float factorG;
    float factorB;
    PackedNrcInput packedInput;
};

struct NrcOptimizerState {
    uint step;
    float beta1T;
    float beta2T;
    float alphaT;
    float alphaTMinus1;
};

struct NrcOptimizerEntry {
    float moment1;
    float moment2;
    float weight;
    float emaWeight;
};
```

Expected sizes under the matching C++ ABI:

```text
PackedNrcInput  16 bytes
NrcEvalRecord   20 bytes
NrcTrainRecord  40 bytes
NrcOptimizerState 20 bytes
NrcOptimizerEntry 16 bytes
NrcQueueState   20 bytes
```

## Destination Encoding

Screen records and train records share a packed `dst` value:

```text
screen: (x | (y << 15)) << 1
train:  (batch | (left << 2) | (right << 16)) << 1 | 1
```

The low bit selects destination type.

## Input Encoding

`NrcEncodeInput` converts renderer-provided `NrcUnpackedInput` into eight `uvec4` values, matching the 64 fp16 input representation used by VkNRC's network.

Renderers remain responsible for constructing `NrcUnpackedInput` from their own scene/hit/material representation.

## Compile Smoke Test

`tests/shader_compile/nrc_common_smoke.comp` includes the common headers and compiles them with `glslc`. Queue, train-prepare, optimizer, scalar inference, and backend wrappers compile the shader-header passes. These catch GLSL syntax and portability issues before real integrations depend on the headers.
