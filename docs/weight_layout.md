# Weight Layout

NRC weights follow the VkNRC-compatible MLP topology.

```text
hidden layers: 5 * (64 x 64)
output layer:  3 x 64
total:         20672 fp16 values
```

## Float Reference Layout

The CPU reference uses float values with row-major matrices:

```text
index = layerOffset + row * 64 + column
```

Layer offsets:

```text
layer 0: 0
layer 1: 4096
layer 2: 8192
layer 3: 12288
layer 4: 16384
output: 20480
```

## Packed GPU Layout

GPU weights are packed as `uvec4` blocks.

```text
1 uvec4 = 4 uint values = 8 fp16 values
```

Within each `uint`:

```text
low 16 bits:  even lane
high 16 bits: odd lane
```

Block count:

```text
20672 / 8 = 2584 uvec4 blocks
```

Byte count:

```text
2584 * 16 = 41344 bytes
```

This layout is consumed by `NrcScalarLoadWeight` and should also be used by the KHR cooperative matrix backend unless a future profile explicitly defines a different layout.
