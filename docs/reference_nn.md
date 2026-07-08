# Reference Neural Network

Milestone 4 adds a reference implementation for the NRC MLP.

## Topology

```text
input:   64 fp16-derived values
hidden:  5 layers, width 64, ReLU after each hidden layer
output:  3 raw radiance-cache values
```

## Weight Layout

Weights are stored by layer.

```text
hidden layer 0: 64 x 64
hidden layer 1: 64 x 64
hidden layer 2: 64 x 64
hidden layer 3: 64 x 64
hidden layer 4: 64 x 64
output layer:   3 x 64
```

The CPU reference treats each matrix as row-major:

```text
weightIndex = layerOffset + row * 64 + column
```

Total weight count:

```text
64 * 64 * 5 + 64 * 3 = 20672
```

## Current Backends

- CPU reference: `nrc::reference_forward`, using float weights.
- GLSL scalar backend: `NrcScalarForward`, reading packed fp16 weights from `uvec4`.
- Vulkan scalar dispatch test: uploads packed fp16 weights/inputs, dispatches `NrcScalarForward`, and compares the result against `nrc::reference_forward` using the same fp16-rounded data.

The scalar shader backend is intended for correctness and debugging. It is not a production fallback for high-performance integrations.
