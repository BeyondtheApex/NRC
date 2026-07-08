# Vulkan Capability Query

The optional `nrc_vulkan` target translates Vulkan device capabilities into the renderer-neutral `nrc::DeviceCapabilitiesView`.

Header:

```cpp
#include "nrc/vulkan/nrc_vulkan_capabilities.hpp"
```

Basic use:

```cpp
nrc::vulkan::VulkanDeviceCapabilities caps =
    nrc::vulkan::query_device_capabilities(physicalDevice, getCoopMatrixPropertiesKHR);

nrc::ProfileId profile = nrc::choose_profile(caps.view());
```

`getCoopMatrixPropertiesKHR` should usually come from the application's Vulkan loader, for example volk or `vkGetInstanceProcAddr`.

The function pointer is optional, but passing it explicitly is recommended because extension commands are not guaranteed to link from `vulkan-1.lib`.

## Features Queried

- `shaderFloat16`
- `storageBuffer16BitAccess`
- `shaderBufferFloat32AtomicAdd`
- `subgroupSizeControl`
- `computeFullSubgroups`
- `VK_KHR_cooperative_matrix`

## Cooperative Matrix Translation

Only compute-stage, subgroup-scope matrix properties are translated into NRC capabilities.

The first production profiles expect FP16 matrices such as:

```text
M=16 N=16 K=16 scope=subgroup
M=8  N=16 K=16 scope=subgroup
```

Unsupported component types are mapped to `ComponentType::Unknown` and therefore will not match production profiles.

## Current Limit

This module only reports capabilities. It does not create devices, enable features, compile shaders, or dispatch NRC workloads.
