/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Ray Tracing Tutorial - 16 Ray Query NRC Live
//
// This sample demonstrates the use of ray queries in compute shaders.
// Ray queries allow you to perform ray tracing operations directly within compute shaders
// without needing a full ray tracing pipeline. This is useful for techniques like
// screen-space reflections, ambient occlusion, or other effects that need to trace rays
// from within compute shaders.
//


// Enable the use of Nsight Aftermath for crash tracking and shader debugging
// #define USE_NSIGHT_AFTERMATH  // (not always on, as it slows down the application)


#define TINYGLTF_IMPLEMENTATION         // Implementation of the GLTF loader library
#define STB_IMAGE_IMPLEMENTATION        // Implementation of the image loading library
#define STB_IMAGE_WRITE_IMPLEMENTATION  // Implementation of the image writing library
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1  // Use dynamic Vulkan functions for VMA (Vulkan Memory Allocator)
#define VMA_IMPLEMENTATION              // Implementation of the Vulkan Memory Allocator
#define VMA_LEAK_LOG_FORMAT(format, ...)                                                                               \
  {                                                                                                                    \
    printf((format), __VA_ARGS__);                                                                                     \
    printf("\n");                                                                                                      \
  }

#include "shaders/shaderio.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#if defined(_MSC_VER) && defined(_WIN32)
#include <excpt.h>
#endif

// Pre-compiled shaders
#include "_autogen/sky_simple.slang.h"  // from nvpro_core2
#include "_autogen/tonemapper.slang.h"  //   "    "
#include "_autogen/ray_query.slang.h"   // Local shader

#if defined(NRC_AVAILABLE)
#include "_autogen/nrc_demo_inference_prepare.comp.h"
#include "_autogen/nrc_demo_optimize.comp.h"
#include "_autogen/nrc_demo_producer.comp.h"
#include "_autogen/nrc_demo_train_prepare.comp.h"
#include "_autogen/nrc_demo_visualize.comp.h"

#include "nrc/nrc_optimizer.hpp"
#include "nrc/nrc_records.hpp"
#include "nrc/nrc_resource_layout.hpp"
#include "nrc/nrc_weights.hpp"
#include "nrc/vulkan/nrc_vulkan_capabilities.hpp"
#include "nrc/vulkan/nrc_vulkan_pipeline.hpp"
#include "nrc/vulkan/nrc_vulkan_shader.hpp"
#endif

// Common base class (see 02_basic)
#include "common/rt_base.hpp"

namespace {

void printVkFailure(VkResult result)
{
  std::printf("[16_ray_query_nrc_live] fatal Vulkan result: %d\n", static_cast<int>(result));
  std::fflush(stdout);
}

bool envFlagEnabled(const char* name)
{
#if defined(_MSC_VER)
  char*  value = nullptr;
  size_t length = 0;
  if(_dupenv_s(&value, &length, name) != 0 || value == nullptr)
    return false;

  const bool enabled = std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0
                       || std::strcmp(value, "TRUE") == 0 || std::strcmp(value, "on") == 0
                       || std::strcmp(value, "ON") == 0;
  std::free(value);
  return enabled;
#else
  const char* value = std::getenv(name);
  if(value == nullptr)
    return false;
  return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 || std::strcmp(value, "TRUE") == 0
         || std::strcmp(value, "on") == 0 || std::strcmp(value, "ON") == 0;
#endif
}

#if defined(NRC_AVAILABLE)
struct NrcProducerConstants
{
  uint32_t mode             = 0;
  uint32_t evalCapacity     = 0;
  uint32_t pixelCount       = 0;
  uint32_t width            = 0;
  uint32_t height           = 0;
  uint32_t trainRecordCount = 0;
  uint32_t frame            = 0;
};

struct NrcBatchConstants
{
  uint32_t batch = 0;
};

struct NrcOptimizeConstants
{
  uint32_t batch         = 0;
  uint32_t useEmaWeights = 1;
};

struct NrcVisualizeConstants
{
  float blend = 0.0f;
};

enum class NrcViewMode : uint32_t
{
  Baseline = 0,
  Live     = 1,
  Frozen   = 2,
  TrainOnly = 3,
};

struct NrcStats
{
  uint32_t evalCount     = 0;
  uint32_t trainCounts[4] = {};
  uint32_t optimizerStep = 0;
  uint32_t readbacks     = 0;
  float    firstWeight   = 0.0f;
  float    weightDelta   = 0.0f;
};

struct NrcGpuTimings
{
  bool  valid              = false;
  float nrcResetMs         = 0.0f;
  float pathTraceMs        = 0.0f;
  float inferencePrepareMs = 0.0f;
  float inferenceMs        = 0.0f;
  float trainPrepareMs     = 0.0f;
  float gradientMs         = 0.0f;
  float optimizeMs         = 0.0f;
  float copyMs             = 0.0f;
  float totalMs            = 0.0f;
};

constexpr uint32_t kNrcQueueWorkgroup     = 64;
constexpr uint32_t kNrcOptimizeWorkgroup  = 64;
constexpr uint32_t kNrcVisualWorkgroup    = 16;
constexpr uint32_t kNrcTimestampStart     = 0;
constexpr uint32_t kNrcTimestampAfterReset = 1;
constexpr uint32_t kNrcTimestampAfterPathTrace = 2;
constexpr uint32_t kNrcTimestampAfterInferencePrepare = 3;
constexpr uint32_t kNrcTimestampAfterInference = 4;
constexpr uint32_t kNrcTimestampTrainBase = 5;
constexpr uint32_t kNrcTimestampPerTrainBatch = 3;
constexpr uint32_t kNrcTimestampAfterCopy =
    kNrcTimestampTrainBase + nrc::kTrainBatchCount * kNrcTimestampPerTrainBatch;
constexpr uint32_t kNrcTimestampCount = kNrcTimestampAfterCopy + 1;

uint32_t divUp(uint32_t value, uint32_t divisor)
{
  return (value + divisor - 1) / divisor;
}

std::vector<uint32_t> readSpirvFile(const char* path)
{
  if(path == nullptr || path[0] == '\0')
    return {};

  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if(!file)
    return {};

  const std::streamsize size = file.tellg();
  if(size <= 0 || (size % 4) != 0)
    return {};

  std::vector<uint32_t> data(static_cast<size_t>(size) / sizeof(uint32_t));
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char*>(data.data()), size);
  return file ? data : std::vector<uint32_t>{};
}

bool hasDeviceExtension(VkPhysicalDevice physicalDevice, const char* name)
{
  uint32_t extensionCount = 0;
  if(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr) != VK_SUCCESS)
    return false;

  std::vector<VkExtensionProperties> extensions(extensionCount);
  if(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, extensions.data()) != VK_SUCCESS)
    return false;

  return std::any_of(extensions.begin(), extensions.end(), [name](const VkExtensionProperties& ext) {
    return std::strcmp(ext.extensionName, name) == 0;
  });
}

bool guardedGetPhysicalDeviceFeatures2(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2* features)
{
#if defined(_MSC_VER) && defined(_WIN32)
  __try
  {
    vkGetPhysicalDeviceFeatures2(physicalDevice, features);
    return true;
  }
  __except(EXCEPTION_EXECUTE_HANDLER)
  {
    return false;
  }
#else
  vkGetPhysicalDeviceFeatures2(physicalDevice, features);
  return true;
#endif
}

bool guardedGetPhysicalDeviceProperties2(VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2* properties)
{
#if defined(_MSC_VER) && defined(_WIN32)
  __try
  {
    vkGetPhysicalDeviceProperties2(physicalDevice, properties);
    return true;
  }
  __except(EXCEPTION_EXECUTE_HANDLER)
  {
    return false;
  }
#else
  vkGetPhysicalDeviceProperties2(physicalDevice, properties);
  return true;
#endif
}

VkResult guardedGetCooperativeMatrixProperties(PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR getProperties,
                                               VkPhysicalDevice physicalDevice,
                                               uint32_t* propertyCount,
                                               VkCooperativeMatrixPropertiesKHR* properties,
                                               bool& structuredException)
{
  structuredException = false;
  if(getProperties == nullptr)
    return VK_ERROR_EXTENSION_NOT_PRESENT;
#if defined(_MSC_VER) && defined(_WIN32)
  __try
  {
    return getProperties(physicalDevice, propertyCount, properties);
  }
  __except(EXCEPTION_EXECUTE_HANDLER)
  {
    structuredException = true;
    return VK_ERROR_UNKNOWN;
  }
#else
  return getProperties(physicalDevice, propertyCount, properties);
#endif
}

struct NrcCapabilityProbeResult
{
  nrc::vulkan::VulkanDeviceCapabilities capabilities;
  std::string                           status;
};

NrcCapabilityProbeResult queryNrcVulkanCapabilitiesGuarded(
    VkInstance instance, VkPhysicalDevice physicalDevice)
{
  NrcCapabilityProbeResult result{};
  nrc::vulkan::VulkanDeviceCapabilities& out = result.capabilities;

  out.khrCooperativeMatrix = hasDeviceExtension(physicalDevice, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
  const bool hasAtomicFloat = hasDeviceExtension(physicalDevice, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);

  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperativeMatrixFeatures{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, .pNext = &atomicFloatFeatures};
  VkPhysicalDeviceVulkan13Features vulkan13Features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .pNext = &cooperativeMatrixFeatures};
  VkPhysicalDeviceVulkan12Features vulkan12Features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &vulkan13Features};
  VkPhysicalDeviceVulkan11Features vulkan11Features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &vulkan12Features};
  VkPhysicalDeviceFeatures2 features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &vulkan11Features};

  if(!guardedGetPhysicalDeviceFeatures2(physicalDevice, &features))
  {
    out.khrCooperativeMatrix = false;
    result.status            = "capability query trapped in vkGetPhysicalDeviceFeatures2";
    return result;
  }

  out.shaderFloat16                  = vulkan12Features.shaderFloat16 == VK_TRUE;
  out.storageBuffer16BitAccess       = vulkan11Features.storageBuffer16BitAccess == VK_TRUE;
  out.shaderBufferFloat32AtomicAdd   = hasAtomicFloat && atomicFloatFeatures.shaderBufferFloat32AtomicAdd == VK_TRUE;
  out.subgroupSizeControl            = vulkan13Features.subgroupSizeControl == VK_TRUE;
  out.computeFullSubgroups           = vulkan13Features.computeFullSubgroups == VK_TRUE;
  out.khrCooperativeMatrix           = out.khrCooperativeMatrix && cooperativeMatrixFeatures.cooperativeMatrix == VK_TRUE;

  VkPhysicalDeviceSubgroupSizeControlProperties subgroupSizeControlProperties{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
  VkPhysicalDeviceSubgroupProperties subgroupProperties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
                                                       .pNext = &subgroupSizeControlProperties};
  VkPhysicalDeviceCooperativeMatrixPropertiesKHR cooperativeMatrixDeviceProperties{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR, .pNext = &subgroupProperties};
  VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                        .pNext = &cooperativeMatrixDeviceProperties};

  if(!guardedGetPhysicalDeviceProperties2(physicalDevice, &properties))
  {
    out.khrCooperativeMatrix = false;
    result.status            = "capability query trapped in vkGetPhysicalDeviceProperties2 with "
                               "VkPhysicalDeviceCooperativeMatrixPropertiesKHR";
    return result;
  }

  out.maxComputeSharedMemorySize = properties.properties.limits.maxComputeSharedMemorySize;
  out.supportedSubgroupSizes = nrc::vulkan::enumerate_subgroup_sizes(subgroupSizeControlProperties.minSubgroupSize,
                                                                     subgroupSizeControlProperties.maxSubgroupSize,
                                                                     subgroupProperties.subgroupSize);
  out.subgroupArithmetic =
      (subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;

  const auto getCooperativeMatrixProperties =
      reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
          vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
  if(out.khrCooperativeMatrix && getCooperativeMatrixProperties != nullptr)
  {
    uint32_t propertyCount       = 0;
    bool     structuredException = false;
    VkResult countResult = guardedGetCooperativeMatrixProperties(getCooperativeMatrixProperties, physicalDevice,
                                                                 &propertyCount, nullptr, structuredException);
    if(structuredException)
    {
      out.khrCooperativeMatrix = false;
      result.status            = "capability query trapped in vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(count)";
      return result;
    }
    if(countResult == VK_SUCCESS && propertyCount > 0)
    {
      std::vector<VkCooperativeMatrixPropertiesKHR> propertiesList(propertyCount);
      for(auto& property : propertiesList)
        property.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;

      VkResult listResult = guardedGetCooperativeMatrixProperties(getCooperativeMatrixProperties, physicalDevice,
                                                                  &propertyCount, propertiesList.data(),
                                                                  structuredException);
      if(structuredException)
      {
        out.khrCooperativeMatrix = false;
        result.status = "capability query trapped in vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(list)";
        return result;
      }
      if(listResult == VK_SUCCESS)
      {
        for(const auto& property : propertiesList)
        {
          auto translated = nrc::vulkan::translate_cooperative_matrix_property(
              property, cooperativeMatrixDeviceProperties.cooperativeMatrixSupportedStages);
          if(translated.has_value())
            out.cooperativeMatrices.push_back(*translated);
        }
      }
    }
  }

  result.status = "capability query passed";
  return result;
}
#endif

}  // namespace


//---------------------------------------------------------------------------------------
// Ray Tracing Tutorial - 16 Ray Query NRC Live
//
// This class demonstrates the use of ray queries in compute shaders.
// Unlike traditional ray tracing pipelines, ray queries allow you to perform
// ray tracing operations directly within compute shaders, providing more flexibility
// for implementing various rendering techniques.
//
class Rt16RayQueryNrcLive : public RtBase
{

public:
  Rt16RayQueryNrcLive() { m_slangCapabilities = {"spvRayQueryKHR"}; }
  ~Rt16RayQueryNrcLive() override = default;

  void onAttach(nvapp::Application* app) override
  {
    std::printf("[16_ray_query_nrc_live] onAttach begin\n");
    std::fflush(stdout);
    RtBase::onAttach(app);
    std::printf("[16_ray_query_nrc_live] RtBase attached\n");
    std::fflush(stdout);
#if defined(NRC_AVAILABLE)
    if(!initNrc())
    {
      std::printf("[16_ray_query_nrc_live] fatal: NRC KHR cooperative-matrix initialization failed\n");
      std::fflush(stdout);
      std::exit(1);
    }
#endif
    std::printf("[16_ray_query_nrc_live] onAttach end\n");
    std::fflush(stdout);
  }

  //-------------------------------------------------------------------------------
  // Override virtual methods from RtBase
  //-------------------------------------------------------------------------------

  //---------------------------------------------------------------------------------------------------------------
  // Rendering all UI elements, this includes the image of the GBuffer, the camera controls, and the sky parameters.
  // - Called every frame
  void onUIRender() override
  {
    namespace PE = nvgui::PropertyEditor;
#if defined(NRC_AVAILABLE)
    refreshNrcStatsForUi();
#endif
    bool changed = false;
    if(ImGui::Begin("Settings"))
    {
      ImGui::SeparatorText("16 Ray Query");
      if(PE::begin())
      {
        PE::DragInt("Max Frames", (int*)&m_maxFrames, 1, 1, 1000);
        ImGui::TextDisabled("Max Frames: %d", m_pushValues.frame);
        changed = PE::SliderInt("Max Depth", (int*)&m_pushValues.maxDepth, 1, 10);
        PE::end();
      }
#if defined(NRC_AVAILABLE)
      renderNrcUI(changed);
#else
      ImGui::SeparatorText("NRC");
      ImGui::TextDisabled("NRC_AVAILABLE is not defined for this target.");
#endif
    }
    ImGui::End();
    changed |= RtBase::renderUI();
    if(changed)
    {
      resetFrame();
    }
  }

  //---------------------------------------------------------------------------------------------------------------
  // Create the scene for this sample
  // - Create primitive meshes (cube and sphere) and load a GLTF model
  // - Create instances for them, assign materials and transformations
  // - Set up lighting and camera for the ray query demonstration
  void createScene() override
  {
    SCOPED_TIMER(__FUNCTION__);

    // Create primitives
    nvutils::PrimitiveMesh cube   = nvutils::createCube();
    nvutils::PrimitiveMesh sphere = nvutils::createSphereUv(10, 10);
    nvsamples::primitiveMeshToResource(m_sceneResource, m_stagingUploader, cube);
    nvsamples::primitiveMeshToResource(m_sceneResource, m_stagingUploader, sphere);
    m_sceneResource.instances = {{
                                     .transform     = glm::scale(glm::mat4(1), glm::vec3(10.0f, 0.01f, 10.0f)),
                                     .materialIndex = 1,
                                     .meshIndex     = 0,
                                 },
                                 {
                                     .transform = glm::scale(glm::translate(glm::mat4(1), glm::vec3(1.5f, 0.5f, 0.0f)),
                                                             glm::vec3(0.1f, 0.1f, 0.1f)),
                                     .materialIndex = 2,
                                     .meshIndex     = 1,
                                 }};

    // Load the GLTF resources
    {
      tinygltf::Model meetMat =
          nvsamples::loadGltfResources(nvutils::findFile("meet_mat.glb", nvsamples::getResourcesDirs()));  // Load the GLTF resources from the file

      const bool importAllInstances = true;
      nvsamples::importGltfData(m_sceneResource, meetMat, m_stagingUploader, importAllInstances);  // Import the GLTF resources
    }

    m_sceneResource.materials = {
        {.baseColorFactor = glm::vec4(0.8f, 1.0f, 0.6f, 1.0f), .metallicFactor = 0.5f, .roughnessFactor = 0.5f},
        {.baseColorFactor = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f), .metallicFactor = 0.5f, .roughnessFactor = 0.05f},
        {.baseColorFactor = glm::vec4(0.8f, 0.2f, 0.2f, 1.0f), .metallicFactor = 0.05f, .roughnessFactor = 0.95f},
    };

    nvsamples::createGltfSceneInfoBuffer(m_sceneResource, m_stagingUploader);

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_stagingUploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);  // Submit the command buffer to upload the resources

    // Set the camera
    m_cameraManip->setCamera(
        {{-1.8179423, 1.0988777, 2.4955902}, {0.15707326, 0.9963042, 0.089965165}, {0, 1, 0}, {60}, {0.01, 100}});
    m_sceneResource.sceneInfo.useSky = true;

    m_sceneResource.sceneInfo.punctualLights[0].position  = {2.0f, 4.0f, 4.0f};
    m_sceneResource.sceneInfo.punctualLights[0].direction = {.7f, 1.0f, 1.0f};
    m_sceneResource.sceneInfo.punctualLights[0].color     = {1.0f, 1.0f, 1.0f};
    m_sceneResource.sceneInfo.punctualLights[0].intensity = 50.0f;
    m_sceneResource.sceneInfo.punctualLights[0].type      = shaderio::GltfLightType::eSpot;
    m_sceneResource.sceneInfo.backgroundColor             = {0.03f, 0.03f, 0.03f};
  }

  //--------------------------------------------------------------------------------------------------
  // Create compute pipeline for ray query
  // Unlike traditional ray tracing pipelines, ray queries use compute shaders
  // This creates a compute pipeline that will execute the ray query shader
  void createRayTracingPipeline() override
  {
    // For re-creation
    destroyRayTracingPipeline();

    // Compile shader, and if failed, use pre-compiled shaders
    VkShaderModuleCreateInfo shaderCode = compileSlangShader("ray_query.slang", ray_query_slang);

    // Push constant: we want to be able to update constants used by the shaders
    const VkPushConstantRange pushConstant{VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::TutoPushConstant)};

    // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
    std::array<VkDescriptorSetLayout, 2> layouts = {{m_descPack.getLayout(), m_rtDescPack.getLayout()}};
    VkPipelineLayoutCreateInfo           pipelineLayoutCreateInfo{
                  .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                  .setLayoutCount         = uint32_t(layouts.size()),
                  .pSetLayouts            = layouts.data(),
                  .pushConstantRangeCount = 1,
                  .pPushConstantRanges    = &pushConstant,
    };
    vkCreatePipelineLayout(m_app->getDevice(), &pipelineLayoutCreateInfo, nullptr, &m_rtPipelineLayout);
    NVVK_DBG_NAME(m_rtPipelineLayout);

    VkPipelineShaderStageCreateInfo shaderStage{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = &shaderCode,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .pName = "main",
    };

    VkComputePipelineCreateInfo cpCreateInfo{
        .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage  = shaderStage,
        .layout = m_rtPipelineLayout,
    };

    NVVK_CHECK(vkCreateComputePipelines(m_app->getDevice(), {}, 1, &cpCreateInfo, nullptr, &m_rtPipeline));
    NVVK_DBG_NAME(m_rtPipeline);
  }


  //---------------------------------------------------------------------------------------------------------------
  // Ray query rendering method
  // This method executes the compute shader that performs ray queries
  // Unlike traditional ray tracing, this uses a compute pipeline with ray queries
  void raytraceScene(VkCommandBuffer cmd) override
  {
    NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight

    updateFrame();
    if(m_pushValues.frame >= m_maxFrames && !isNrcTrainingActive())
      return;

#if defined(NRC_AVAILABLE)
    const VkExtent2D& size = m_app->getViewportSize();
    if(m_nrc.enabled && m_nrc.initialized && m_nrc.evalCapacity != 0)
      beginNrcTimings(cmd);
    resetNrcFrame(cmd, size);
    if(m_nrc.enabled && m_nrc.initialized && m_nrc.evalCapacity != 0)
      writeNrcTimestamp(cmd, kNrcTimestampAfterReset);
#endif

    // Bind the descriptor sets for the graphics pipeline (making textures available to the shaders)
    const VkBindDescriptorSetsInfo bindDescriptorSetsInfo{.sType      = VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO,
                                                          .stageFlags = VK_SHADER_STAGE_ALL,
                                                          .layout     = m_rtPipelineLayout,
                                                          .firstSet   = 0,
                                                          .descriptorSetCount = 1,
                                                          .pDescriptorSets    = m_descPack.getSetPtr()};
    vkCmdBindDescriptorSets2(cmd, &bindDescriptorSetsInfo);

    // Push descriptor sets for ray tracing
    nvvk::WriteSetContainer write{};
    write.append(m_rtDescPack.makeWrite(shaderio::BindingPoints::eTlas), m_asBuilder.tlas);
    write.append(m_rtDescPack.makeWrite(shaderio::BindingPoints::eOutImage), m_gBuffers.getColorImageView(eImgRendered),
                 VK_IMAGE_LAYOUT_GENERAL);
#if defined(NRC_AVAILABLE)
    if(m_nrc.initialized && m_nrc.evalCapacity != 0)
    {
      write.append(m_rtDescPack.makeWrite(shaderio::BindingPoints::eNrcQueueState), m_nrc.buffers.queueState);
      write.append(m_rtDescPack.makeWrite(shaderio::BindingPoints::eNrcEvalRecords), m_nrc.buffers.evalRecords);
      write.append(m_rtDescPack.makeWrite(shaderio::BindingPoints::eNrcTrainRecords), m_nrc.buffers.trainRecords);
      write.append(m_rtDescPack.makeWrite(shaderio::BindingPoints::eNrcEncodedInputs), m_nrc.buffers.encodedInputs);
      write.append(m_rtDescPack.makeWrite(shaderio::BindingPoints::eNrcBiasFactorR), m_nrc.biasFactorR.descriptor.imageView,
                   VK_IMAGE_LAYOUT_GENERAL);
      write.append(m_rtDescPack.makeWrite(shaderio::BindingPoints::eNrcFactorGB), m_nrc.factorGB.descriptor.imageView,
                   VK_IMAGE_LAYOUT_GENERAL);
    }
#endif
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipelineLayout, 1, write.size(), write.data());

    // Push constant information, see usage later
    m_pushValues.sceneInfoAddress = (shaderio::GltfSceneInfo*)m_sceneResource.bSceneInfo.address;  // Pass the address of the scene information buffer to the shader
    m_pushValues.metallicRoughnessOverride = m_metallicRoughnessOverride;  // Override the metallic and roughness values
#if defined(NRC_AVAILABLE)
    const bool nrcReady = m_nrc.enabled && m_nrc.initialized && m_nrc.evalCapacity != 0;
    const bool nrcCollect = nrcReady && shouldTrainNrc();
    const bool nrcApply = nrcReady && shouldApplyNrc();
    const uint32_t nrcCollectMode = nrcCollect ? 1u : 0u;
    const uint32_t nrcApplyMode = nrcApply ? 2u : 0u;
    m_pushValues.nrcMode = nrcCollectMode | nrcApplyMode;
    m_pushValues.nrcEvalCapacity = m_nrc.evalCapacity;
    m_pushValues.nrcTrainFeatureBase = m_nrc.trainFeatureBase;
    m_pushValues.nrcHandoffDepth = m_nrc.handoffDepth;
    m_pushValues.nrcTrainProbability = nrcCollect ? m_nrc.trainProbability : 0.0f;
#else
    m_pushValues.nrcMode = 0u;
    m_pushValues.nrcEvalCapacity = 0u;
    m_pushValues.nrcTrainFeatureBase = 0u;
    m_pushValues.nrcHandoffDepth = 1u;
    m_pushValues.nrcTrainProbability = 0.0f;
#endif

    const VkPushConstantsInfo pushInfo{.sType      = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
                                       .layout     = m_rtPipelineLayout,
                                       .stageFlags = VK_SHADER_STAGE_ALL,
                                       .size       = sizeof(shaderio::TutoPushConstant),
                                       .pValues    = &m_pushValues};
    vkCmdPushConstants2(cmd, &pushInfo);


    // Execute the compute shader with ray queries
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipeline);
    vkCmdDispatch(cmd, (size.width + (WORKGROUP_SIZE - 1)) / WORKGROUP_SIZE,
                  (size.height + (WORKGROUP_SIZE - 1)) / WORKGROUP_SIZE, 1);

#if defined(NRC_AVAILABLE)
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    if(m_nrc.enabled && m_nrc.initialized && m_nrc.evalCapacity != 0)
      writeNrcTimestamp(cmd, kNrcTimestampAfterPathTrace);
    consumeNrc(cmd, size);
#endif

    // Making sure the rendered image is ready to be used by tonemapper
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  }

  //---------------------------------------------------------------------------------------------------------------
  // Frame management functions for progressive rendering
  // These functions handle frame accumulation for progressive ray query rendering
  //---------------------------------------------------------------------------------------------------------------

  // Reset the frame counter to restart progressive rendering
  void resetFrame() { m_pushValues.frame = -1; }

  // Update the frame counter and reset if camera changes
  // This enables progressive rendering where each frame accumulates more samples
  void updateFrame()
  {
    static glm::dmat4 refCamMatrix;
    static double     refFov{m_cameraManip->getFov()};

    const auto& m   = m_cameraManip->getViewMatrix();
    const auto  fov = m_cameraManip->getFov();

    if(refCamMatrix != m || refFov != fov)
    {
      resetFrame();
      refCamMatrix = m;
      refFov       = fov;
    }
    ++m_pushValues.frame;
    if(!isNrcTrainingActive())
      m_pushValues.frame = std::min(m_pushValues.frame, m_maxFrames);  // Limit only ordinary progressive accumulation
  }

  //---------------------------------------------------------------------------------------------------------------
  // When the viewport is resized, reset the frame counter for progressive rendering
  // - Called when the Window viewport is resized
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override
  {
    resetFrame();
    RtBase::onResize(cmd, size);
#if defined(NRC_AVAILABLE)
    resizeNrc(size);
#endif
  }

  void createRaytraceDescriptorLayout() override
  {
    SCOPED_TIMER(__FUNCTION__);
    m_rtBindings.addBinding({.binding         = shaderio::BindingPoints::eTlas,
                             .descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                             .descriptorCount = 1,
                             .stageFlags      = VK_SHADER_STAGE_ALL});
    m_rtBindings.addBinding({.binding         = shaderio::BindingPoints::eOutImage,
                             .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                             .descriptorCount = 1,
                             .stageFlags      = VK_SHADER_STAGE_ALL});
#if defined(NRC_AVAILABLE)
    m_rtBindings.addBinding({.binding         = shaderio::BindingPoints::eNrcQueueState,
                             .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             .descriptorCount = 1,
                             .stageFlags      = VK_SHADER_STAGE_ALL});
    m_rtBindings.addBinding({.binding         = shaderio::BindingPoints::eNrcEvalRecords,
                             .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             .descriptorCount = 1,
                             .stageFlags      = VK_SHADER_STAGE_ALL});
    m_rtBindings.addBinding({.binding         = shaderio::BindingPoints::eNrcTrainRecords,
                             .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             .descriptorCount = 1,
                             .stageFlags      = VK_SHADER_STAGE_ALL});
    m_rtBindings.addBinding({.binding         = shaderio::BindingPoints::eNrcEncodedInputs,
                             .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             .descriptorCount = 1,
                             .stageFlags      = VK_SHADER_STAGE_ALL});
    m_rtBindings.addBinding({.binding         = shaderio::BindingPoints::eNrcBiasFactorR,
                             .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                             .descriptorCount = 1,
                             .stageFlags      = VK_SHADER_STAGE_ALL});
    m_rtBindings.addBinding({.binding         = shaderio::BindingPoints::eNrcFactorGB,
                             .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                             .descriptorCount = 1,
                             .stageFlags      = VK_SHADER_STAGE_ALL});
#endif
    m_rtDescPack.init(m_rtBindings, m_app->getDevice(), 0, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);
  }

  void sampleDestroy() override
  {
#if defined(NRC_AVAILABLE)
    destroyNrc();
#endif
  }

  void onLastHeadlessFrame() override
  {
#if defined(NRC_AVAILABLE)
    readNrcStats();
    std::printf("[16_ray_query_nrc_live] NRC %s/%s: eval=%u/%u train0=%u/%u step=%u firstWeight=%.6f delta=%.6f\n",
                m_nrc.backendName.c_str(), m_nrc.profileName.c_str(), m_nrc.stats.evalCount, m_nrc.evalCapacity,
                m_nrc.stats.trainCounts[0], nrc::kTrainBatchSize, m_nrc.stats.optimizerStep,
                m_nrc.stats.firstWeight, m_nrc.stats.weightDelta);
    if(m_nrc.timings.valid)
    {
      std::printf("[16_ray_query_nrc_live] NRC GPU ms: total=%.3f reset=%.3f path=%.3f inferPrep=%.3f infer=%.3f trainPrep=%.3f gradient=%.3f optimize=%.3f copy=%.3f\n",
                  m_nrc.timings.totalMs, m_nrc.timings.nrcResetMs, m_nrc.timings.pathTraceMs,
                  m_nrc.timings.inferencePrepareMs, m_nrc.timings.inferenceMs, m_nrc.timings.trainPrepareMs,
                  m_nrc.timings.gradientMs, m_nrc.timings.optimizeMs, m_nrc.timings.copyMs);
    }
    else
    {
      std::printf("[16_ray_query_nrc_live] NRC GPU ms: timestamp results unavailable\n");
    }
    std::printf("[16_ray_query_nrc_live] NRC KHR status: %s\n", m_nrc.khrStatusText.c_str());
    std::fflush(stdout);
#endif
    RtBase::onLastHeadlessFrame();
  }

private:
  uint32_t m_maxFrames = 200;  // Maximum number of frames for accumulation

#if defined(NRC_AVAILABLE)
  struct NrcBufferSet
  {
    nvvk::Buffer weights;
    nvvk::Buffer useWeights;
    nvvk::Buffer evalRecords;
    nvvk::Buffer trainRecords;
    nvvk::Buffer encodedInputs;
    nvvk::Buffer queueState;
    nvvk::Buffer gradients;
    nvvk::Buffer optimizerEntries;
    nvvk::Buffer optimizerState;
    nvvk::Buffer indirect;
  };

  struct NrcPipelineSet
  {
    nvvk::DescriptorPack producerPack;
    nvvk::DescriptorPack inferencePreparePack;
    nvvk::DescriptorPack inferencePack;
    nvvk::DescriptorPack trainPreparePack;
    nvvk::DescriptorPack gradientPack;
    nvvk::DescriptorPack optimizePack;
    nvvk::DescriptorPack visualizePack;

    VkPipelineLayout producerLayout     = VK_NULL_HANDLE;
    VkPipelineLayout inferencePrepareLayout = VK_NULL_HANDLE;
    VkPipelineLayout inferenceLayout    = VK_NULL_HANDLE;
    VkPipelineLayout trainPrepareLayout = VK_NULL_HANDLE;
    VkPipelineLayout gradientLayout     = VK_NULL_HANDLE;
    VkPipelineLayout optimizeLayout     = VK_NULL_HANDLE;
    VkPipelineLayout visualizeLayout    = VK_NULL_HANDLE;

    VkPipeline producer     = VK_NULL_HANDLE;
    VkPipeline inferencePrepare = VK_NULL_HANDLE;
    VkPipeline inference    = VK_NULL_HANDLE;
    VkPipeline trainPrepare = VK_NULL_HANDLE;
    VkPipeline gradient     = VK_NULL_HANDLE;
    VkPipeline optimize     = VK_NULL_HANDLE;
    VkPipeline visualize    = VK_NULL_HANDLE;
  };

  struct NrcRuntime
  {
    bool enabled          = true;
    NrcViewMode viewMode  = NrcViewMode::TrainOnly;
    bool trainingSupported = false;
    bool useEmaWeights    = true;

    uint32_t handoffDepth              = 1;
    float    trainProbability          = nrc::kDefaultTrainProbability;

    bool initialized          = false;
    bool imagesNeedTransition = true;
    bool buffersNeedInit      = true;
    bool dispatchSeen         = false;
    bool timingQueriesIssued  = false;
    uint32_t statsReadTick    = 0;

    VkExtent2D extent{};
    uint32_t   evalCapacity = 0;
    uint32_t   pixelCount   = 0;
    uint32_t   trainCapacity = 0;
    uint32_t   trainFeatureBase = 0;

    nvvk::Image biasFactorR;
    nvvk::Image factorGB;

    NrcBufferSet   buffers;
    NrcPipelineSet pipelines;
    NrcStats       stats;
    NrcGpuTimings  timings;
    VkQueryPool    timingQueryPool = VK_NULL_HANDLE;
    float          timestampPeriod = 0.0f;

    nrc::vulkan::VulkanDeviceCapabilities capabilities;
    nrc::vulkan::VulkanTrainingProfilePlan trainingPlan;

    std::string backendName = "KHR required";
    std::string profileName = "KHR required";
    std::string statusText  = "Not initialized";
    std::string khrStatusText = "Not initialized";
  } m_nrc;

  void renderNrcUI(bool& changed)
  {
    auto explainLastItem = [](const char* text) {
      if(ImGui::IsItemHovered())
      {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 34.0f);
        ImGui::TextUnformatted(text);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
      }
    };

    const bool nrcReady = m_nrc.enabled && m_nrc.initialized && m_nrc.evalCapacity != 0;
    const bool collecting = nrcReady && shouldTrainNrc();
    const bool applying = nrcReady && shouldApplyNrc();
    const char* modeNames[] = {"Baseline", "Live NRC", "Frozen NRC", "Train Only"};
    int modeIndex = static_cast<int>(m_nrc.viewMode);
    const char* frameMode = !nrcReady ? "Baseline, NRC inactive" : modeNames[modeIndex];
    uint32_t trainTotal = 0;
    for(uint32_t i = 0; i < nrc::kTrainBatchCount; ++i)
      trainTotal += m_nrc.stats.trainCounts[i];

    ImGui::SeparatorText("NRC");
    changed |= ImGui::Checkbox("Enable NRC", &m_nrc.enabled);
    explainLastItem("Enables the NRC buffers and compute passes. If this is off, the sample renders the ordinary path-traced baseline.");
    if(ImGui::Combo("Mode", &modeIndex, modeNames, IM_ARRAYSIZE(modeNames)))
    {
      m_nrc.viewMode = static_cast<NrcViewMode>(std::clamp(modeIndex, 0, 3));
      changed = true;
    }
    explainLastItem("Baseline traces the full path. Live NRC collects training records, applies NRC to the visible image, and updates weights each frame. Frozen NRC applies the current useWeights without training. Train Only updates weights without using NRC for the visible image.");
    ImGui::BeginDisabled(!m_nrc.trainingSupported);
    ImGui::TextDisabled("Training is controlled by Mode.");
    ImGui::EndDisabled();
    changed |= ImGui::Checkbox("Use EMA for Inference", &m_nrc.useEmaWeights);
    explainLastItem("Uses exponential-moving-average weights for inference/optimization output. EMA is usually more stable than the raw immediately-updated weights.");
    changed |= ImGui::DragInt("Handoff Depth", reinterpret_cast<int*>(&m_nrc.handoffDepth), 1.0f, 1, 8);
    explainLastItem("Surface bounce where NRC predicts the remaining indirect radiance. At that point the path shader writes bias + factor and an eval record instead of tracing the rest of that visible path.");
    changed |= ImGui::SliderFloat("Train Sample Probability", &m_nrc.trainProbability, 0.0f, 0.20f, "%.3f");
    explainLastItem("Probability that an eligible handoff path also traces a real continuation for training. Higher values train faster but spend more ray budget and GPU time.");

    ImGui::SeparatorText("NRC State");
    ImGui::Text("Frame mode: %s", frameMode);
    ImGui::Text("Collect train records: %s", collecting ? "yes" : "no");
    ImGui::Text("Apply to image: %s", applying ? "yes" : "no");
    ImGui::Text("Screen accumulation: %s", applying ? "off while NRC is applied" : "tutorial progressive");
    ImGui::Text("Stats readbacks: %u", m_nrc.stats.readbacks);
    ImGui::Text("Backend: %s", m_nrc.backendName.c_str());
    ImGui::Text("Profile: %s", m_nrc.profileName.c_str());
    ImGui::Text("Status: %s", m_nrc.statusText.c_str());
    ImGui::TextWrapped("KHR status: %s", m_nrc.khrStatusText.c_str());

    ImGui::SeparatorText("NRC Queues");
    ImGui::Text("Eval queue: %u / %u", m_nrc.stats.evalCount, m_nrc.evalCapacity);
    ImGui::Text("Screen records: %u", m_nrc.pixelCount);
    ImGui::Text("Train records: %u / %u", trainTotal, m_nrc.trainCapacity);
    for(uint32_t i = 0; i < nrc::kTrainBatchCount; ++i)
      ImGui::Text("Train batch %u: %u / %u", i, m_nrc.stats.trainCounts[i], nrc::kTrainBatchSize);

    ImGui::SeparatorText("NRC Optimizer");
    ImGui::Text("Optimizer step: %u", m_nrc.stats.optimizerStep);
    ImGui::Text("First weight: %.6f", m_nrc.stats.firstWeight);
    ImGui::Text("Weight delta: %.6f", m_nrc.stats.weightDelta);

    ImGui::SeparatorText("NRC GPU Time");
    if(m_nrc.timings.valid)
    {
      ImGui::Text("Total: %.3f ms", m_nrc.timings.totalMs);
      ImGui::Text("Path trace + records: %.3f ms", m_nrc.timings.pathTraceMs);
      ImGui::Text("NRC reset: %.3f ms", m_nrc.timings.nrcResetMs);
      ImGui::Text("Inference prepare: %.3f ms", m_nrc.timings.inferencePrepareMs);
      ImGui::Text("Inference: %.3f ms", m_nrc.timings.inferenceMs);
      ImGui::Text("Train prepare: %.3f ms", m_nrc.timings.trainPrepareMs);
      ImGui::Text("Gradient: %.3f ms", m_nrc.timings.gradientMs);
      ImGui::Text("Optimize: %.3f ms", m_nrc.timings.optimizeMs);
      ImGui::Text("Copy/apply: %.3f ms", m_nrc.timings.copyMs);
    }
    else
    {
      ImGui::TextDisabled("Waiting for timestamp results.");
    }
  }

  bool isNrcTrainingActive() const
  {
    return m_nrc.enabled && m_nrc.initialized && shouldTrainNrc();
  }

  bool shouldTrainNrc() const
  {
    return m_nrc.viewMode == NrcViewMode::Live || m_nrc.viewMode == NrcViewMode::TrainOnly;
  }

  bool shouldApplyNrc() const
  {
    return m_nrc.viewMode == NrcViewMode::Live || m_nrc.viewMode == NrcViewMode::Frozen;
  }

  void createNrcTimingQueries()
  {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(m_app->getPhysicalDevice(), &properties);
    m_nrc.timestampPeriod = properties.limits.timestampPeriod;

    VkQueryPoolCreateInfo queryPoolInfo{.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                        .queryType  = VK_QUERY_TYPE_TIMESTAMP,
                                        .queryCount = kNrcTimestampCount};
    NVVK_CHECK(vkCreateQueryPool(m_app->getDevice(), &queryPoolInfo, nullptr, &m_nrc.timingQueryPool));
  }

  void writeNrcTimestamp(VkCommandBuffer cmd, uint32_t index)
  {
    if(m_nrc.timingQueryPool == VK_NULL_HANDLE || index >= kNrcTimestampCount)
      return;
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, m_nrc.timingQueryPool, index);
  }

  void beginNrcTimings(VkCommandBuffer cmd)
  {
    if(m_nrc.timingQueryPool == VK_NULL_HANDLE)
      return;

    vkCmdResetQueryPool(cmd, m_nrc.timingQueryPool, 0, kNrcTimestampCount);
    m_nrc.timingQueriesIssued = true;
    m_nrc.timings.valid = false;
    writeNrcTimestamp(cmd, kNrcTimestampStart);
  }

  void markUnusedNrcTimestamps(VkCommandBuffer cmd, uint32_t firstIndex, uint32_t endIndex)
  {
    for(uint32_t index = firstIndex; index < endIndex; ++index)
      writeNrcTimestamp(cmd, index);
  }

  void refreshNrcStatsForUi()
  {
    if(!m_nrc.dispatchSeen)
      return;

    ++m_nrc.statsReadTick;
    if((m_nrc.statsReadTick % 8u) != 0u)
      return;

    readNrcStats();
  }

  bool initNrc()
  {
    std::printf("[16_ray_query_nrc_live] initNrc: query profile\n");
    std::fflush(stdout);
    if(!queryNrcProfile())
      return false;
    std::printf("[16_ray_query_nrc_live] initNrc: create pipelines\n");
    std::fflush(stdout);
    if(!createNrcPipelines())
      return false;
    createNrcTimingQueries();
    std::printf("[16_ray_query_nrc_live] initNrc: resize buffers\n");
    std::fflush(stdout);
    resizeNrc(m_app->getViewportSize());
    m_nrc.initialized = true;
    m_nrc.statusText  = "Initialized";
    std::printf("[16_ray_query_nrc_live] NRC initialized: %s/%s, evalCapacity=%u, trainCapacity=%u\n",
                m_nrc.backendName.c_str(), m_nrc.profileName.c_str(), m_nrc.evalCapacity, m_nrc.trainCapacity);
    std::fflush(stdout);
    return true;
  }

  bool queryNrcProfile()
  {
    std::printf("[16_ray_query_nrc_live] initNrc: query Vulkan NRC capabilities\n");
    std::fflush(stdout);
    NrcCapabilityProbeResult capabilityProbe =
        queryNrcVulkanCapabilitiesGuarded(m_app->getInstance(), m_app->getPhysicalDevice());
    m_nrc.capabilities = std::move(capabilityProbe.capabilities);
    m_nrc.khrStatusText = capabilityProbe.status;
    std::printf("[16_ray_query_nrc_live] initNrc: Vulkan NRC capabilities queried: %s\n", capabilityProbe.status.c_str());
    std::fflush(stdout);

    nrc::TrainingProfileSelectionOptions selectionOptions{};
    selectionOptions.allowScalarReferenceFallback = false;
    m_nrc.trainingPlan = nrc::vulkan::make_vulkan_training_profile_plan(m_nrc.capabilities, selectionOptions);
    const nrc::TrainingProfileSelection& selection = m_nrc.trainingPlan.selection;

    if(!m_nrc.trainingPlan.selected || !m_nrc.trainingPlan.shaderCompile.valid
       || m_nrc.trainingPlan.shaderCompile.backend != nrc::BackendKind::KhrCooperativeMatrix
       || selection.shaderConfig == nullptr)
    {
      m_nrc.statusText = "Fatal: KHR cooperative-matrix NRC profile is required";
      std::printf("[16_ray_query_nrc_live] %s; selection=%s, validation=%s\n", m_nrc.statusText.c_str(),
                  nrc::training_profile_selection_status_name(selection.status),
                  nrc::vulkan::vulkan_profile_validation_status_name(m_nrc.trainingPlan.validation.status));
      std::fflush(stdout);
      return false;
    }

    m_nrc.trainingSupported = true;
    m_nrc.viewMode          = NrcViewMode::TrainOnly;
    m_nrc.backendName       = "KhrRecompute";
    m_nrc.profileName       = selection.profileName;
    m_nrc.statusText        = "KHR cooperative-matrix NRC path active";
    m_nrc.khrStatusText += std::string("; selection=")
                            + nrc::training_profile_selection_status_name(selection.status)
                            + ", implementation=" + nrc::training_implementation_kind_name(selection.implementation)
                            + ", validation="
                            + nrc::vulkan::vulkan_profile_validation_status_name(m_nrc.trainingPlan.validation.status);
    return true;
  }

  bool createNrcPipelines()
  {
    VkDevice device = m_app->getDevice();

    auto makeLayout = [&](nvvk::DescriptorPack& pack, std::span<const VkPushConstantRange> pushRanges) {
      VkDescriptorSetLayout layout = pack.getLayout();
      VkPipelineLayout pipelineLayout{};
      NVVK_CHECK(nvvk::createPipelineLayout(device, &pipelineLayout, std::span(&layout, 1), pushRanges));
      return pipelineLayout;
    };

    nvvk::DescriptorBindings producerBindings;
    producerBindings.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    producerBindings.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    producerBindings.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    producerBindings.addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    producerBindings.addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    producerBindings.addBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    NVVK_CHECK(m_nrc.pipelines.producerPack.init(producerBindings, device, 1));

    nvvk::DescriptorBindings inferencePrepareBindings;
    inferencePrepareBindings.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    inferencePrepareBindings.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    NVVK_CHECK(m_nrc.pipelines.inferencePreparePack.init(inferencePrepareBindings, device, 1));

    nvvk::DescriptorBindings inferenceBindings;
    inferenceBindings.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    inferenceBindings.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    inferenceBindings.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    inferenceBindings.addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    inferenceBindings.addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    inferenceBindings.addBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    inferenceBindings.addBinding(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    NVVK_CHECK(m_nrc.pipelines.inferencePack.init(inferenceBindings, device, 1));

    nvvk::DescriptorBindings trainPrepareBindings;
    trainPrepareBindings.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    trainPrepareBindings.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    trainPrepareBindings.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    NVVK_CHECK(m_nrc.pipelines.trainPreparePack.init(trainPrepareBindings, device, 1));

    nvvk::DescriptorBindings gradientBindings;
    gradientBindings.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    gradientBindings.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    gradientBindings.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    gradientBindings.addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    gradientBindings.addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    NVVK_CHECK(m_nrc.pipelines.gradientPack.init(gradientBindings, device, 1));

    nvvk::DescriptorBindings optimizeBindings;
    optimizeBindings.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    optimizeBindings.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    optimizeBindings.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    optimizeBindings.addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    optimizeBindings.addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    optimizeBindings.addBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    NVVK_CHECK(m_nrc.pipelines.optimizePack.init(optimizeBindings, device, 1));

    nvvk::DescriptorBindings visualizeBindings;
    visualizeBindings.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    visualizeBindings.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    NVVK_CHECK(m_nrc.pipelines.visualizePack.init(visualizeBindings, device, 1));

    VkPushConstantRange producerPush{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(NrcProducerConstants)};
    VkPushConstantRange batchPush{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(NrcBatchConstants)};
    VkPushConstantRange optimizePush{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(NrcOptimizeConstants)};
    VkPushConstantRange visualizePush{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(NrcVisualizeConstants)};

    m_nrc.pipelines.producerLayout = makeLayout(m_nrc.pipelines.producerPack, std::span(&producerPush, 1));
    m_nrc.pipelines.inferencePrepareLayout = makeLayout(m_nrc.pipelines.inferencePreparePack, {});
    m_nrc.pipelines.inferenceLayout = makeLayout(m_nrc.pipelines.inferencePack, {});
    m_nrc.pipelines.trainPrepareLayout = makeLayout(m_nrc.pipelines.trainPreparePack, std::span(&batchPush, 1));
    m_nrc.pipelines.gradientLayout = makeLayout(m_nrc.pipelines.gradientPack, std::span(&batchPush, 1));
    m_nrc.pipelines.optimizeLayout = makeLayout(m_nrc.pipelines.optimizePack, std::span(&optimizePush, 1));
    m_nrc.pipelines.visualizeLayout = makeLayout(m_nrc.pipelines.visualizePack, std::span(&visualizePush, 1));

    std::printf("[16_ray_query_nrc_live] initNrc: create producer pipeline\n");
    std::fflush(stdout);
    m_nrc.pipelines.producer = createComputePipeline(std::span(nrc_demo_producer_comp), m_nrc.pipelines.producerLayout);
    std::printf("[16_ray_query_nrc_live] initNrc: create inference prepare pipeline\n");
    std::fflush(stdout);
    m_nrc.pipelines.inferencePrepare =
        createComputePipeline(std::span(nrc_demo_inference_prepare_comp), m_nrc.pipelines.inferencePrepareLayout);
    std::printf("[16_ray_query_nrc_live] initNrc: create KHR inference pipeline\n");
    std::fflush(stdout);
    m_nrc.pipelines.inference = createKhrProfilePipeline("inference", selectedKhrInferenceSpirvPath(),
                                                         m_nrc.pipelines.inferenceLayout);
    if(m_nrc.pipelines.inference == VK_NULL_HANDLE)
      return false;

    std::printf("[16_ray_query_nrc_live] initNrc: create train prepare pipeline\n");
    std::fflush(stdout);
    m_nrc.pipelines.trainPrepare =
        createComputePipeline(std::span(nrc_demo_train_prepare_comp), m_nrc.pipelines.trainPrepareLayout);
    std::printf("[16_ray_query_nrc_live] initNrc: create KHR recompute gradient pipeline\n");
    std::fflush(stdout);
    m_nrc.pipelines.gradient = createKhrProfilePipeline("gradient", selectedKhrGradientSpirvPath(),
                                                        m_nrc.pipelines.gradientLayout);
    if(m_nrc.pipelines.gradient == VK_NULL_HANDLE)
      return false;

    std::printf("[16_ray_query_nrc_live] initNrc: create optimize pipeline\n");
    std::fflush(stdout);
    m_nrc.pipelines.optimize = createComputePipeline(std::span(nrc_demo_optimize_comp), m_nrc.pipelines.optimizeLayout);
    std::printf("[16_ray_query_nrc_live] initNrc: create visualize pipeline\n");
    std::fflush(stdout);
    m_nrc.pipelines.visualize = createComputePipeline(std::span(nrc_demo_visualize_comp), m_nrc.pipelines.visualizeLayout);
    if(m_nrc.pipelines.producer == VK_NULL_HANDLE || m_nrc.pipelines.inferencePrepare == VK_NULL_HANDLE
       || m_nrc.pipelines.trainPrepare == VK_NULL_HANDLE
       || m_nrc.pipelines.optimize == VK_NULL_HANDLE || m_nrc.pipelines.visualize == VK_NULL_HANDLE)
    {
      m_nrc.statusText = "Fatal: non-NN NRC support pipeline creation failed";
      std::printf("[16_ray_query_nrc_live] %s\n", m_nrc.statusText.c_str());
      std::fflush(stdout);
      return false;
    }

    m_nrc.khrStatusText += ", active execution=KHR cooperative matrix";
    std::printf("[16_ray_query_nrc_live] initNrc: pipeline creation finished\n");
    std::fflush(stdout);
    return true;
  }

  const char* selectedKhrInferenceSpirvPath() const
  {
#if defined(NRC_LIVE_INFERENCE_KHR_SG32_SPV) && defined(NRC_LIVE_INFERENCE_KHR_SG16_SPV) \
    && defined(NRC_LIVE_INFERENCE_KHR_M8_SG32_SPV)
    const nrc::ProfileShaderConfig* config = m_nrc.trainingPlan.selection.shaderConfig;
    if(config == nullptr || config->backend != nrc::BackendKind::KhrCooperativeMatrix)
      return nullptr;

    if(config->cooperativeMatrixM == 16u && config->subgroupSize == 32u)
      return NRC_LIVE_INFERENCE_KHR_SG32_SPV;
    if(config->cooperativeMatrixM == 16u && config->subgroupSize == 16u)
      return NRC_LIVE_INFERENCE_KHR_SG16_SPV;
    if(config->cooperativeMatrixM == 8u && config->subgroupSize == 32u)
      return NRC_LIVE_INFERENCE_KHR_M8_SG32_SPV;
#endif
    return nullptr;
  }

  const char* selectedKhrGradientSpirvPath() const
  {
#if defined(NRC_LIVE_GRADIENT_KHR_SG32_SPV) && defined(NRC_LIVE_GRADIENT_KHR_SG16_SPV) \
    && defined(NRC_LIVE_GRADIENT_KHR_M8_SG32_SPV)
    const nrc::ProfileShaderConfig* config = m_nrc.trainingPlan.selection.shaderConfig;
    if(config == nullptr || config->backend != nrc::BackendKind::KhrCooperativeMatrix)
      return nullptr;

    if(config->cooperativeMatrixM == 16u && config->subgroupSize == 32u)
      return NRC_LIVE_GRADIENT_KHR_SG32_SPV;
    if(config->cooperativeMatrixM == 16u && config->subgroupSize == 16u)
      return NRC_LIVE_GRADIENT_KHR_SG16_SPV;
    if(config->cooperativeMatrixM == 8u && config->subgroupSize == 32u)
      return NRC_LIVE_GRADIENT_KHR_M8_SG32_SPV;
#endif
    return nullptr;
  }

  VkPipeline createKhrProfilePipeline(const char* label, const char* spirvPath, VkPipelineLayout layout)
  {
    if(spirvPath == nullptr)
    {
      m_nrc.statusText = std::string("Fatal: no KHR SPIR-V path for ") + label;
      std::printf("[16_ray_query_nrc_live] %s\n", m_nrc.statusText.c_str());
      std::fflush(stdout);
      return VK_NULL_HANDLE;
    }

    const std::vector<uint32_t> spirv = readSpirvFile(spirvPath);
    if(spirv.empty())
    {
      m_nrc.statusText = std::string("Fatal: could not read KHR SPIR-V for ") + label;
      std::printf("[16_ray_query_nrc_live] %s: %s\n", m_nrc.statusText.c_str(), spirvPath);
      std::fflush(stdout);
      return VK_NULL_HANDLE;
    }

    nrc::vulkan::VulkanShaderModuleCreateInfo shaderCreateInfo{};
    shaderCreateInfo.device = m_app->getDevice();
    shaderCreateInfo.code   = std::span<const uint32_t>(spirv.data(), spirv.size());
    shaderCreateInfo.getDeviceProcAddr = vkGetDeviceProcAddr;
    std::printf("[16_ray_query_nrc_live] KHR %s: create shader module from %s\n", label, spirvPath);
    std::fflush(stdout);
    const nrc::vulkan::VulkanShaderModuleCreateResult shaderResult =
        nrc::vulkan::create_shader_module(shaderCreateInfo);
    std::printf("[16_ray_query_nrc_live] KHR %s: shader module returned %s%s\n", label,
                nrc::vulkan::vulkan_validation_step_status_name(shaderResult.validation.status),
                shaderResult.validation.structuredException ? " (structured exception)" : "");
    std::fflush(stdout);

    if(shaderResult.validation.status != nrc::vulkan::VulkanValidationStepStatus::Passed)
    {
      m_nrc.statusText = std::string("Fatal: KHR shader module creation failed for ") + label;
      return VK_NULL_HANDLE;
    }

    std::printf("[16_ray_query_nrc_live] KHR %s: create profile compute pipeline\n", label);
    std::fflush(stdout);
    nrc::vulkan::VulkanComputePipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.device         = m_app->getDevice();
    pipelineCreateInfo.pipelineLayout = layout;
    pipelineCreateInfo.shaderModule   = shaderResult.shaderModule;
    pipelineCreateInfo.shaderConfig   = m_nrc.trainingPlan.selection.shaderConfig;
    pipelineCreateInfo.getDeviceProcAddr = vkGetDeviceProcAddr;
    const nrc::vulkan::VulkanComputePipelineCreateResult pipelineResult =
        nrc::vulkan::create_profile_compute_pipeline(pipelineCreateInfo);
    std::printf("[16_ray_query_nrc_live] KHR %s: profile compute pipeline returned %s%s\n", label,
                nrc::vulkan::vulkan_validation_step_status_name(pipelineResult.validation.status),
                pipelineResult.validation.structuredException ? " (structured exception)" : "");
    std::fflush(stdout);

    vkDestroyShaderModule(m_app->getDevice(), shaderResult.shaderModule, nullptr);
    if(pipelineResult.validation.status != nrc::vulkan::VulkanValidationStepStatus::Passed
       || pipelineResult.pipeline == VK_NULL_HANDLE)
    {
      m_nrc.statusText = std::string("Fatal: KHR compute pipeline creation failed for ") + label;
      return VK_NULL_HANDLE;
    }
    return pipelineResult.pipeline;
  }

  VkPipeline createComputePipeline(std::span<const uint32_t> spirv, VkPipelineLayout layout)
  {
    VkShaderModule           shaderModule{};
    VkShaderModuleCreateInfo moduleInfo{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                        .codeSize = spirv.size_bytes(),
                                        .pCode    = spirv.data()};
    VkResult                 moduleResult = vkCreateShaderModule(m_app->getDevice(), &moduleInfo, nullptr, &shaderModule);
    if(moduleResult != VK_SUCCESS || shaderModule == VK_NULL_HANDLE)
      return VK_NULL_HANDLE;

    VkPipelineShaderStageCreateInfo stageInfo{.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                              .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
                                              .module = shaderModule,
                                              .pName  = "main"};
    VkComputePipelineCreateInfo     pipelineInfo{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .stage = stageInfo, .layout = layout};
    VkPipeline pipeline{};
    VkResult   pipelineResult = vkCreateComputePipelines(m_app->getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    vkDestroyShaderModule(m_app->getDevice(), shaderModule, nullptr);
    return pipelineResult == VK_SUCCESS ? pipeline : VK_NULL_HANDLE;
  }

  void resizeNrc(const VkExtent2D& size)
  {
    if(size.width == 0 || size.height == 0)
      return;

    destroyNrcImages();
    destroyNrcBuffers();

    const nrc::ResourceLayout resourceLayout =
        nrc::compute_resource_layout({.width = size.width, .height = size.height});

    m_nrc.extent        = size;
    m_nrc.pixelCount    = size.width * size.height;
    m_nrc.evalCapacity  = static_cast<uint32_t>(resourceLayout.evalRecordCount);
    m_nrc.trainCapacity = static_cast<uint32_t>(resourceLayout.trainBatchCount * resourceLayout.trainBatchSize);
    m_nrc.trainFeatureBase = m_nrc.evalCapacity;

    createNrcImages(size);
    createNrcBuffers();
    updateNrcDescriptors();
    m_nrc.imagesNeedTransition = true;
    m_nrc.buffersNeedInit      = true;
    std::printf("[16_ray_query_nrc_live] NRC resources: %ux%u evalCapacity=%u trainCapacity=%u trainFeatureBase=%u\n",
                size.width, size.height, m_nrc.evalCapacity, m_nrc.trainCapacity, m_nrc.trainFeatureBase);
    std::fflush(stdout);
  }

  void createNrcImages(const VkExtent2D& size)
  {
    VkImageCreateInfo imageInfo = DEFAULT_VkImageCreateInfo;
    imageInfo.imageType         = VK_IMAGE_TYPE_2D;
    imageInfo.extent            = {size.width, size.height, 1};
    imageInfo.mipLevels         = 1;
    imageInfo.arrayLayers       = 1;
    imageInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage             = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    imageInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    NVVK_CHECK(m_allocator.createImage(m_nrc.biasFactorR, imageInfo, DEFAULT_VkImageViewCreateInfo));

    imageInfo.format = VK_FORMAT_R32G32_SFLOAT;
    VkImageViewCreateInfo viewInfo = DEFAULT_VkImageViewCreateInfo;
    viewInfo.format                = imageInfo.format;
    NVVK_CHECK(m_allocator.createImage(m_nrc.factorGB, imageInfo, viewInfo));
  }

  void createNrcBuffers()
  {
    constexpr VkBufferUsageFlags2 usage = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
                                          | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT;
    constexpr VmaAllocationCreateFlags hostFlags =
        VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

    auto createHost = [&](nvvk::Buffer& buffer, VkDeviceSize size) {
      NVVK_CHECK(m_allocator.createBuffer(buffer, size, usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, hostFlags));
    };

    const nrc::ResourceLayout resourceLayout =
        nrc::compute_resource_layout({.width = m_nrc.extent.width, .height = m_nrc.extent.height});

    createHost(m_nrc.buffers.weights, resourceLayout.weightsBytes);
    createHost(m_nrc.buffers.useWeights, resourceLayout.weightsBytes);
    createHost(m_nrc.buffers.evalRecords, resourceLayout.evalRecordsBytes);
    createHost(m_nrc.buffers.trainRecords, resourceLayout.trainBatchRecordsTotalBytes);
    const VkDeviceSize encodedInputCount = (resourceLayout.evalRecordCount
                                            + resourceLayout.trainBatchCount * resourceLayout.trainBatchSize);
    createHost(m_nrc.buffers.encodedInputs, encodedInputCount * nrc::kEncodedInputUvec4Count * sizeof(uint32_t) * 4);
    createHost(m_nrc.buffers.queueState, resourceLayout.queueStateBytes);
    createHost(m_nrc.buffers.gradients, resourceLayout.gradientsBytes);
    createHost(m_nrc.buffers.optimizerEntries, resourceLayout.optimizerEntriesBytes);
    createHost(m_nrc.buffers.optimizerState, resourceLayout.optimizerStateBytes);
    createHost(m_nrc.buffers.indirect, sizeof(VkDispatchIndirectCommand));

    initializeNrcWeights();
  }

  void initializeNrcWeights()
  {
    std::vector<float> weights = nrc::make_kaiming_weights(7);
    std::vector<nrc::PackedFp16x8> packed = nrc::pack_weights_fp16_uvec4(weights);
    std::memcpy(m_nrc.buffers.weights.mapping, packed.data(), packed.size() * sizeof(nrc::PackedFp16x8));
    NVVK_CHECK(m_allocator.autoFlushBuffer(m_nrc.buffers.weights));
    std::memcpy(m_nrc.buffers.useWeights.mapping, packed.data(), packed.size() * sizeof(nrc::PackedFp16x8));
    NVVK_CHECK(m_allocator.autoFlushBuffer(m_nrc.buffers.useWeights));

    std::vector<nrc::NrcOptimizerEntry> entries(nrc::weight_count());
    nrc::initialize_optimizer_entries(weights, entries);
    std::memcpy(m_nrc.buffers.optimizerEntries.mapping, entries.data(), entries.size() * sizeof(nrc::NrcOptimizerEntry));
    NVVK_CHECK(m_allocator.autoFlushBuffer(m_nrc.buffers.optimizerEntries));

    nrc::NrcOptimizerState state{};
    std::memcpy(m_nrc.buffers.optimizerState.mapping, &state, sizeof(state));
    NVVK_CHECK(m_allocator.autoFlushBuffer(m_nrc.buffers.optimizerState));
    m_nrc.stats.firstWeight = entries.empty() ? 0.0f : entries[0].weight;
  }

  void updateNrcDescriptors()
  {
    VkDevice device = m_app->getDevice();
    nvvk::WriteSetContainer write;

    write.append(m_nrc.pipelines.producerPack.makeWrite(0), m_nrc.buffers.queueState);
    write.append(m_nrc.pipelines.producerPack.makeWrite(1), m_nrc.buffers.evalRecords);
    write.append(m_nrc.pipelines.producerPack.makeWrite(2), m_nrc.buffers.trainRecords);
    write.append(m_nrc.pipelines.producerPack.makeWrite(3), m_nrc.buffers.encodedInputs);
    write.append(m_nrc.pipelines.producerPack.makeWrite(4), m_nrc.biasFactorR.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL);
    write.append(m_nrc.pipelines.producerPack.makeWrite(5), m_nrc.factorGB.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL);

    write.append(m_nrc.pipelines.inferencePreparePack.makeWrite(0), m_nrc.buffers.queueState);
    write.append(m_nrc.pipelines.inferencePreparePack.makeWrite(1), m_nrc.buffers.indirect);

    write.append(m_nrc.pipelines.inferencePack.makeWrite(0), m_nrc.buffers.useWeights);
    write.append(m_nrc.pipelines.inferencePack.makeWrite(1), m_nrc.buffers.evalRecords);
    write.append(m_nrc.pipelines.inferencePack.makeWrite(2), m_nrc.buffers.queueState.buffer, 0, sizeof(uint32_t));
    write.append(m_nrc.pipelines.inferencePack.makeWrite(3), m_nrc.biasFactorR.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL);
    write.append(m_nrc.pipelines.inferencePack.makeWrite(4), m_nrc.factorGB.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL);
    write.append(m_nrc.pipelines.inferencePack.makeWrite(5), m_nrc.buffers.trainRecords);
    write.append(m_nrc.pipelines.inferencePack.makeWrite(6), m_nrc.buffers.encodedInputs);

    write.append(m_nrc.pipelines.trainPreparePack.makeWrite(0), m_nrc.buffers.queueState);
    write.append(m_nrc.pipelines.trainPreparePack.makeWrite(1), m_nrc.buffers.indirect);
    write.append(m_nrc.pipelines.trainPreparePack.makeWrite(2), m_nrc.buffers.optimizerState);

    write.append(m_nrc.pipelines.gradientPack.makeWrite(0), m_nrc.buffers.trainRecords);
    write.append(m_nrc.pipelines.gradientPack.makeWrite(1), m_nrc.buffers.queueState);
    write.append(m_nrc.pipelines.gradientPack.makeWrite(2), m_nrc.buffers.weights);
    write.append(m_nrc.pipelines.gradientPack.makeWrite(3), m_nrc.buffers.gradients);
    write.append(m_nrc.pipelines.gradientPack.makeWrite(4), m_nrc.buffers.encodedInputs);

    write.append(m_nrc.pipelines.optimizePack.makeWrite(0), m_nrc.buffers.weights);
    write.append(m_nrc.pipelines.optimizePack.makeWrite(1), m_nrc.buffers.useWeights);
    write.append(m_nrc.pipelines.optimizePack.makeWrite(2), m_nrc.buffers.gradients);
    write.append(m_nrc.pipelines.optimizePack.makeWrite(3), m_nrc.buffers.optimizerEntries);
    write.append(m_nrc.pipelines.optimizePack.makeWrite(4), m_nrc.buffers.queueState);
    write.append(m_nrc.pipelines.optimizePack.makeWrite(5), m_nrc.buffers.optimizerState);

    write.append(m_nrc.pipelines.visualizePack.makeWrite(0), m_gBuffers.getColorImageView(eImgRendered), VK_IMAGE_LAYOUT_GENERAL);
    write.append(m_nrc.pipelines.visualizePack.makeWrite(1), m_nrc.biasFactorR.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL);

    vkUpdateDescriptorSets(device, write.size(), write.data(), 0, nullptr);
  }

  void resetNrcFrame(VkCommandBuffer cmd, const VkExtent2D& size)
  {
    if(!m_nrc.enabled || !m_nrc.initialized || m_nrc.evalCapacity == 0)
      return;

    if(m_nrc.imagesNeedTransition)
    {
      nvvk::cmdImageMemoryBarrier(cmd, {m_nrc.biasFactorR.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL});
      nvvk::cmdImageMemoryBarrier(cmd, {m_nrc.factorGB.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL});
      m_nrc.imagesNeedTransition = false;
    }

    if(m_nrc.buffersNeedInit)
    {
      vkCmdFillBuffer(cmd, m_nrc.buffers.queueState.buffer, 0, VK_WHOLE_SIZE, 0);
      vkCmdFillBuffer(cmd, m_nrc.buffers.gradients.buffer, 0, VK_WHOLE_SIZE, 0);
      vkCmdFillBuffer(cmd, m_nrc.buffers.indirect.buffer, 0, VK_WHOLE_SIZE, 0);
      nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
      m_nrc.buffersNeedInit = false;
    }

    vkCmdFillBuffer(cmd, m_nrc.buffers.gradients.buffer, 0, VK_WHOLE_SIZE, 0);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    const uint32_t nrcFrame = m_pushValues.frame > 0 ? static_cast<uint32_t>(m_pushValues.frame) : 0u;
    NrcProducerConstants resetConstants{
        .mode = 0, .evalCapacity = m_nrc.evalCapacity, .pixelCount = m_nrc.pixelCount, .width = size.width,
        .height = size.height, .trainRecordCount = 0, .frame = nrcFrame};
    dispatchProducer(cmd, resetConstants, divUp(std::max(m_nrc.pixelCount, nrc::kTrainBatchCount), kNrcQueueWorkgroup));
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  }

  void consumeNrc(VkCommandBuffer cmd, const VkExtent2D& size)
  {
    if(!m_nrc.enabled || !m_nrc.initialized || m_nrc.evalCapacity == 0)
      return;

    const uint32_t khrWorkgroup = m_nrc.trainingPlan.selection.shaderConfig != nullptr
                                      ? m_nrc.trainingPlan.selection.shaderConfig->workgroupSize
                                      : 128u;
    (void)khrWorkgroup;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_nrc.pipelines.inferencePrepare);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_nrc.pipelines.inferencePrepareLayout, 0, 1,
                            m_nrc.pipelines.inferencePreparePack.getSetPtr(), 0, nullptr);
    vkCmdDispatch(cmd, 1, 1, 1);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    writeNrcTimestamp(cmd, kNrcTimestampAfterInferencePrepare);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_nrc.pipelines.inference);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_nrc.pipelines.inferenceLayout, 0, 1,
                            m_nrc.pipelines.inferencePack.getSetPtr(), 0, nullptr);
    vkCmdDispatchIndirect(cmd, m_nrc.buffers.indirect.buffer, 0);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    writeNrcTimestamp(cmd, kNrcTimestampAfterInference);

    if(shouldTrainNrc())
    {
      for(uint32_t batch = 0; batch < nrc::kTrainBatchCount; ++batch)
      {
        const uint32_t trainTimestampBase = kNrcTimestampTrainBase + batch * kNrcTimestampPerTrainBatch;
        vkCmdFillBuffer(cmd, m_nrc.buffers.gradients.buffer, 0, VK_WHOLE_SIZE, 0);
        nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

        NrcBatchConstants batchConstants{batch};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_nrc.pipelines.trainPrepare);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_nrc.pipelines.trainPrepareLayout, 0, 1,
                                m_nrc.pipelines.trainPreparePack.getSetPtr(), 0, nullptr);
        vkCmdPushConstants(cmd, m_nrc.pipelines.trainPrepareLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(batchConstants), &batchConstants);
        vkCmdDispatch(cmd, 1, 1, 1);
        nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        writeNrcTimestamp(cmd, trainTimestampBase + 0);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_nrc.pipelines.gradient);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_nrc.pipelines.gradientLayout, 0, 1,
                                m_nrc.pipelines.gradientPack.getSetPtr(), 0, nullptr);
        vkCmdPushConstants(cmd, m_nrc.pipelines.gradientLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(batchConstants), &batchConstants);
        vkCmdDispatchIndirect(cmd, m_nrc.buffers.indirect.buffer, 0);
        nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        writeNrcTimestamp(cmd, trainTimestampBase + 1);

        NrcOptimizeConstants optimizeConstants{batch, m_nrc.useEmaWeights ? 1u : 0u};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_nrc.pipelines.optimize);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_nrc.pipelines.optimizeLayout, 0, 1,
                                m_nrc.pipelines.optimizePack.getSetPtr(), 0, nullptr);
        vkCmdPushConstants(cmd, m_nrc.pipelines.optimizeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(optimizeConstants), &optimizeConstants);
        vkCmdDispatch(cmd, divUp(nrc::weight_uvec4_count(), kNrcOptimizeWorkgroup), 1, 1);
        nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        writeNrcTimestamp(cmd, trainTimestampBase + 2);
      }
    }
    else
    {
      markUnusedNrcTimestamps(cmd, kNrcTimestampTrainBase, kNrcTimestampAfterCopy);
    }

    if(shouldApplyNrc())
    {
      NrcVisualizeConstants constants{1.0f};
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_nrc.pipelines.visualize);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_nrc.pipelines.visualizeLayout, 0, 1,
                              m_nrc.pipelines.visualizePack.getSetPtr(), 0, nullptr);
      vkCmdPushConstants(cmd, m_nrc.pipelines.visualizeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants),
                         &constants);
      vkCmdDispatch(cmd, divUp(size.width, kNrcVisualWorkgroup), divUp(size.height, kNrcVisualWorkgroup), 1);
      nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    }
    writeNrcTimestamp(cmd, kNrcTimestampAfterCopy);

    m_nrc.dispatchSeen = true;
  }

  void dispatchProducer(VkCommandBuffer cmd, const NrcProducerConstants& constants, uint32_t groups)
  {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_nrc.pipelines.producer);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_nrc.pipelines.producerLayout, 0, 1,
                            m_nrc.pipelines.producerPack.getSetPtr(), 0, nullptr);
    vkCmdPushConstants(cmd, m_nrc.pipelines.producerLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
    vkCmdDispatch(cmd, groups, 1, 1);
  }

  void readNrcStats()
  {
    if(!m_nrc.dispatchSeen)
      return;

    readNrcTimings();

    NVVK_CHECK(m_allocator.autoInvalidateBuffer(m_nrc.buffers.queueState));
    NVVK_CHECK(m_allocator.autoInvalidateBuffer(m_nrc.buffers.optimizerState));
    NVVK_CHECK(m_allocator.autoInvalidateBuffer(m_nrc.buffers.optimizerEntries));

    const auto* queueState = reinterpret_cast<const nrc::NrcQueueState*>(m_nrc.buffers.queueState.mapping);
    m_nrc.stats.evalCount  = queueState->evalCount;
    for(uint32_t i = 0; i < nrc::kTrainBatchCount; ++i)
      m_nrc.stats.trainCounts[i] = queueState->trainCounts[i];

    const auto* optimizerState = reinterpret_cast<const nrc::NrcOptimizerState*>(m_nrc.buffers.optimizerState.mapping);
    m_nrc.stats.optimizerStep  = optimizerState->step;

    const auto* entries = reinterpret_cast<const nrc::NrcOptimizerEntry*>(m_nrc.buffers.optimizerEntries.mapping);
    m_nrc.stats.weightDelta = entries[0].weight - m_nrc.stats.firstWeight;
    ++m_nrc.stats.readbacks;
  }

  void readNrcTimings()
  {
    if(!m_nrc.timingQueriesIssued || m_nrc.timingQueryPool == VK_NULL_HANDLE || m_nrc.timestampPeriod <= 0.0f)
      return;

    struct TimestampResult
    {
      uint64_t value     = 0;
      uint64_t available = 0;
    };

    std::array<TimestampResult, kNrcTimestampCount> results{};
    VkResult result = vkGetQueryPoolResults(m_app->getDevice(), m_nrc.timingQueryPool, 0, kNrcTimestampCount,
                                            sizeof(results), results.data(), sizeof(TimestampResult),
                                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if(result != VK_SUCCESS && result != VK_NOT_READY)
      return;

    for(const TimestampResult& timestamp : results)
    {
      if(timestamp.available == 0)
        return;
    }

    auto elapsedMs = [&](uint32_t begin, uint32_t end) {
      if(end >= kNrcTimestampCount || begin >= kNrcTimestampCount || results[end].value < results[begin].value)
        return 0.0f;
      return static_cast<float>(static_cast<double>(results[end].value - results[begin].value)
                                * static_cast<double>(m_nrc.timestampPeriod) / 1000000.0);
    };

    NrcGpuTimings timings{};
    timings.valid              = true;
    timings.nrcResetMs         = elapsedMs(kNrcTimestampStart, kNrcTimestampAfterReset);
    timings.pathTraceMs        = elapsedMs(kNrcTimestampAfterReset, kNrcTimestampAfterPathTrace);
    timings.inferencePrepareMs = elapsedMs(kNrcTimestampAfterPathTrace, kNrcTimestampAfterInferencePrepare);
    timings.inferenceMs        = elapsedMs(kNrcTimestampAfterInferencePrepare, kNrcTimestampAfterInference);

    uint32_t previous = kNrcTimestampAfterInference;
    for(uint32_t batch = 0; batch < nrc::kTrainBatchCount; ++batch)
    {
      const uint32_t trainTimestampBase = kNrcTimestampTrainBase + batch * kNrcTimestampPerTrainBatch;
      timings.trainPrepareMs += elapsedMs(previous, trainTimestampBase + 0);
      timings.gradientMs += elapsedMs(trainTimestampBase + 0, trainTimestampBase + 1);
      timings.optimizeMs += elapsedMs(trainTimestampBase + 1, trainTimestampBase + 2);
      previous = trainTimestampBase + 2;
    }

    timings.copyMs  = elapsedMs(previous, kNrcTimestampAfterCopy);
    timings.totalMs = elapsedMs(kNrcTimestampStart, kNrcTimestampAfterCopy);
    m_nrc.timings   = timings;
  }

  void destroyNrc()
  {
    VkDevice device = m_app ? m_app->getDevice() : VK_NULL_HANDLE;
    if(device != VK_NULL_HANDLE)
      vkDeviceWaitIdle(device);

    destroyNrcImages();
    destroyNrcBuffers();
    destroyNrcPipelines();
    m_nrc.initialized = false;
  }

  void destroyNrcImages()
  {
    m_allocator.destroyImage(m_nrc.biasFactorR);
    m_allocator.destroyImage(m_nrc.factorGB);
  }

  void destroyNrcBuffers()
  {
    m_allocator.destroyBuffer(m_nrc.buffers.weights);
    m_allocator.destroyBuffer(m_nrc.buffers.useWeights);
    m_allocator.destroyBuffer(m_nrc.buffers.evalRecords);
    m_allocator.destroyBuffer(m_nrc.buffers.trainRecords);
    m_allocator.destroyBuffer(m_nrc.buffers.encodedInputs);
    m_allocator.destroyBuffer(m_nrc.buffers.queueState);
    m_allocator.destroyBuffer(m_nrc.buffers.gradients);
    m_allocator.destroyBuffer(m_nrc.buffers.optimizerEntries);
    m_allocator.destroyBuffer(m_nrc.buffers.optimizerState);
    m_allocator.destroyBuffer(m_nrc.buffers.indirect);
  }

  void destroyNrcPipelines()
  {
    VkDevice device = m_app ? m_app->getDevice() : VK_NULL_HANDLE;
    if(device == VK_NULL_HANDLE)
      return;

    vkDestroyPipeline(device, m_nrc.pipelines.producer, nullptr);
    vkDestroyPipeline(device, m_nrc.pipelines.inferencePrepare, nullptr);
    vkDestroyPipeline(device, m_nrc.pipelines.inference, nullptr);
    vkDestroyPipeline(device, m_nrc.pipelines.trainPrepare, nullptr);
    vkDestroyPipeline(device, m_nrc.pipelines.gradient, nullptr);
    vkDestroyPipeline(device, m_nrc.pipelines.optimize, nullptr);
    vkDestroyPipeline(device, m_nrc.pipelines.visualize, nullptr);

    vkDestroyQueryPool(device, m_nrc.timingQueryPool, nullptr);
    m_nrc.timingQueryPool = VK_NULL_HANDLE;

    vkDestroyPipelineLayout(device, m_nrc.pipelines.producerLayout, nullptr);
    vkDestroyPipelineLayout(device, m_nrc.pipelines.inferencePrepareLayout, nullptr);
    vkDestroyPipelineLayout(device, m_nrc.pipelines.inferenceLayout, nullptr);
    vkDestroyPipelineLayout(device, m_nrc.pipelines.trainPrepareLayout, nullptr);
    vkDestroyPipelineLayout(device, m_nrc.pipelines.gradientLayout, nullptr);
    vkDestroyPipelineLayout(device, m_nrc.pipelines.optimizeLayout, nullptr);
    vkDestroyPipelineLayout(device, m_nrc.pipelines.visualizeLayout, nullptr);

    m_nrc.pipelines.producerPack.deinit();
    m_nrc.pipelines.inferencePreparePack.deinit();
    m_nrc.pipelines.inferencePack.deinit();
    m_nrc.pipelines.trainPreparePack.deinit();
    m_nrc.pipelines.gradientPack.deinit();
    m_nrc.pipelines.optimizePack.deinit();
    m_nrc.pipelines.visualizePack.deinit();
  }
#endif
};


//---------------------------------------------------------------------------------------------------------------
// The main function, entry point of the application
int main(int argc, char** argv)
{
  nvvk::CheckError::getInstance().setCallbackFunction(printVkFailure);

  nvapp::ApplicationCreateInfo appInfo{};

  // Parsing the command line
  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;
  reg.add({"headless", "Run in headless mode"}, &appInfo.headless, true);
  cli.add(reg);
  cli.parse(argc, argv);

  std::printf("[16_ray_query_nrc_live] KHR cooperative matrix device support: required\n");
  std::fflush(stdout);

  // Setting up the Vulkan context, instance and device extensions
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};

  // Add ray tracing features
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR rayqueryFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
  atomicFloatFeature.shaderBufferFloat32AtomicAdd = VK_TRUE;
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperativeMatrixFeature{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  cooperativeMatrixFeature.cooperativeMatrix = VK_TRUE;

  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions =
          {
              {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME},
              {VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjectFeatures},
              {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature},     // Build acceleration structures
              {VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature},  // Use vkCmdTraceRaysKHR
              {VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME},                  // Required by ray tracing pipeline
              {VK_KHR_RAY_QUERY_EXTENSION_NAME, &rayqueryFeature},  // Enable ray queries in compute shaders
              {VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, &atomicFloatFeature},  // NRC KHR training atomics
          },
  };

  vkSetup.deviceExtensions.push_back(
      {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, &cooperativeMatrixFeature});  // Required NRC KHR path

  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions, &vkSetup.deviceExtensions);
  }

  // Create Vulkan context using the new method
  std::printf("[16_ray_query_nrc_live] creating Vulkan context\n");
  std::fflush(stdout);
  auto vkContext = RtBase::createVulkanContext(vkSetup);
  if(!vkContext)
  {
    std::printf("[16_ray_query_nrc_live] Vulkan context creation failed\n");
    std::fflush(stdout);
    return 1;
  }
  std::printf("[16_ray_query_nrc_live] Vulkan context ready\n");
  std::fflush(stdout);

  // Setting up the application
  appInfo.name           = "Ray Tracing Tutorial - 16 Ray Query NRC Live";
  appInfo.instance       = vkContext->getInstance();
  appInfo.device         = vkContext->getDevice();
  appInfo.physicalDevice = vkContext->getPhysicalDevice();
  appInfo.queues         = vkContext->getQueueInfos();

  // Create the application
  nvapp::Application application;
  VkResult appInitResult = application.init(appInfo);
  std::printf("[16_ray_query_nrc_live] application.init returned: %d\n", static_cast<int>(appInitResult));
  std::fflush(stdout);
  if(appInitResult != VK_SUCCESS)
  {
    std::printf("[16_ray_query_nrc_live] application.init failed: %d\n", static_cast<int>(appInitResult));
    std::fflush(stdout);
    vkContext->deinit();
    return 1;
  }

  // Elements added to the application
  auto tutorial   = std::make_shared<Rt16RayQueryNrcLive>();          // Our tutorial element
  auto elemCamera = std::make_shared<nvapp::ElementCamera>();  // Element to control the camera movement
  auto windowTitle = std::make_shared<nvapp::ElementDefaultWindowTitle>();  // Element displaying the window title with application name and size
  auto windowMenu = std::make_shared<nvapp::ElementDefaultMenu>();  // Element displaying a menu, File->Exit ...
  auto camManip   = tutorial->getCameraManipulator();
  elemCamera->setCameraManipulator(camManip);

  // Adding all elements
  application.addElement(windowMenu);
  application.addElement(windowTitle);
  application.addElement(elemCamera);
  application.addElement(tutorial);

  application.run();     // Start the application, loop until the window is closed
  application.deinit();  // Closing application
  vkContext->deinit();   // De-initialize the Vulkan context

  return 0;
}
