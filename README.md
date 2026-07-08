# NRC

Reusable Vulkan Neural Radiance Cache implementation.

This repository is intended to provide three layers:

1. A reusable NRC core with stable host/shader ABI.
2. A Vulkan runtime contract that external renderers can integrate without adopting the sample renderer.
3. A small reference path tracing sample used to validate quality, performance, and integration behavior.

The first implementation will follow the existing VkNRC implementation and the NRC paper closely, while restructuring the code so projects such as MCVR-custom can integrate it cleanly.

Current status:

- Repository bootstrap is complete.
- Public constants, profile/capability selection, support diagnostics, shader config mapping, and VkNRC-compatible resource layout calculations are implemented.
- Optional Vulkan capability query, profile validation report, shader module creation, storage-buffer descriptor, and profile-aware compute pipeline creation helpers are implemented.
- CPU reference forward/backpropagation, scalar Vulkan inference/training passes, KHR cooperative-matrix inference profiles, CTest coverage, and a small reference path tracing sample are in progress.
