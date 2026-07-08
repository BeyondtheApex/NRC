# NRC

Reusable Vulkan Neural Radiance Cache implementation.

This repository is intended to provide three layers:

1. A reusable NRC core with stable host/shader ABI.
2. A Vulkan runtime contract that external renderers can integrate without adopting the sample renderer.
3. A small reference path tracing sample used to validate quality, performance, and integration behavior.

The first implementation will follow the existing VkNRC implementation and the NRC paper closely, while restructuring the code so projects such as MCVR-custom can integrate it cleanly.

Current status:

- Repository bootstrap is complete.
- Public constants, profile/capability selection, and VkNRC-compatible resource layout calculations are implemented.
- A first profile/layout test is available through CTest.
