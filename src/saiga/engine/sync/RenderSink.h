/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The engine's interface to rendering (docs/engine_design.md §10.1, §10.3; S19b): the World forwards its mesh and
// material registrations and the extract sends the render instances of entities. Implemented by VulkanRenderSink (over
// RenderScene) and RecordingRenderSink (headless tests and scenarios).

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"

#include <memory>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
using Vulkan::DynamicPose;
using Vulkan::InstanceHandle;
using Vulkan::InstanceSetHandle;
using Vulkan::MaterialData;
using Vulkan::MaterialId;
using Vulkan::MeshId;
using Vulkan::MeshLods;
using Vulkan::StaticInstance;

// Failure contract: a call that throws has no effect (addDynamic creates no instance, removeDynamic leaves the handle
// live, setDynamic keeps the previous pose); the extract retries such calls (§10.3). VulkanRenderSink throws only on
// allocation failure, before it changes anything; RecordingRenderSink asserts on misuse instead of throwing.
class RenderSink
{
   public:
    virtual ~RenderSink() = default;
    // IDs are assigned by the World's name tables (World::registerMesh, §7.10; from Vulkan::kGameReservedIds upward);
    // the World forwards every registration (also those made before the sink was set) with its ID. An identical
    // (ID, name) registration again is a no-op (the same sink set twice); a different one asserts.
    virtual void registerMaterial(MaterialId id, const std::string& name, const MaterialData& data) = 0;
    virtual void registerMesh(MeshId id, const std::string& name, std::shared_ptr<const MeshLods> lods,
                              MaterialId material)                                                  = 0;
    // Static instance sets (tier 1, far content; S20+)
    virtual InstanceSetHandle addInstanceSet(const RegionId& region, const std::vector<StaticInstance>& instances,
                                             uint32_t capacity = 0)                                           = 0;
    virtual void updateInstances(InstanceSetHandle set, uint32_t first, const std::vector<StaticInstance>& v) = 0;
    virtual void removeInstance(InstanceSetHandle set, uint32_t index)                                        = 0;
    virtual void removeInstanceSet(InstanceSetHandle set)                                                     = 0;
    // Dynamic instances (drawn after their first setDynamic); poses relative to the chunk's origin
    virtual InstanceHandle addDynamic(MeshId mesh)   = 0;
    virtual void setDynamic(InstanceHandle h, const ChunkId& chunk, const DynamicPose& prev, const DynamicPose& curr,
                            uint32_t boneOffset = 0) = 0;
    virtual void removeDynamic(InstanceHandle h)     = 0;
    // The view (double; fovy in radians) and the interpolation factor of the dynamic instances
    virtual void setCamera(SpaceId space, const Vec3& position, const Quat& rotation, double fovy) = 0;
    virtual void setInterpolation(float alpha)                                                     = 0;
};

}  // namespace Engine
}  // namespace Saiga
