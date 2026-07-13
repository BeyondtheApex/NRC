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

#pragma once


#include "common/io_gltf.h"

NAMESPACE_SHADERIO_BEGIN()

#define WORKGROUP_SIZE 16

// Binding Points
enum BindingPoints
{
  eTextures = 0,  // Binding point for textures
  eOutImage,      // Binding point for output image
  eTlas,          // Top-level acceleration structure
  eNrcQueueState,
  eNrcEvalRecords,
  eNrcTrainRecords,
  eNrcEncodedInputs,
  eNrcBiasFactorR,
  eNrcFactorGB,
};


struct TutoPushConstant
{
  float3x3       normalMatrix;
  int            instanceIndex;              // Instance index for the current draw call
  GltfSceneInfo* sceneInfoAddress;           // Address of the scene information buffer
  float2         metallicRoughnessOverride;  // Metallic and roughness override values
  float          lightRadius = 1.0f;         // Area light radius
  uint32_t       maxDepth    = 3;            // Max ray depth
  uint32_t       frame       = 0;            // Current frame index
  uint32_t       nrcMode     = 0;            // bit 0: collect train records, bit 1: apply screen NRC
  uint32_t       nrcEvalCapacity = 0;        // Maximum eval records in the NRC queue
  uint32_t       nrcTrainFeatureBase = 0;    // First encoded-input slot reserved for train records
  uint32_t       nrcHandoffDepth = 1;        // Bounce depth where NRC predicts the remaining path
  float          nrcTrainProbability = 0.03f;// Probability that a path contributes a training sample
};

NAMESPACE_SHADERIO_END()
