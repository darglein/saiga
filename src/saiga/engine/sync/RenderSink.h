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
using Vulkan::GrassMaskData;
using Vulkan::InstanceHandle;
using Vulkan::InstanceSetHandle;
using Vulkan::MaterialData;
using Vulkan::MaterialId;
using Vulkan::MeshId;
using Vulkan::MeshLods;
using Vulkan::StaticInstance;
using Vulkan::TerrainLayout;
using Vulkan::TerrainPyramidData;
using Vulkan::TerrainTileData;

// Failure contract of the dynamic-instance calls (the ones the extract retries, §10.3): a call that throws has no
// effect (addDynamic creates no instance, removeDynamic leaves the handle live, setDynamic keeps the previous pose).
// The other calls (registration, instance sets) are not retried and give no such guarantee. VulkanRenderSink throws
// only on allocation failure, before it changes anything; RecordingRenderSink asserts on misuse instead of throwing.
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

    // ---- terrain (S20, §10.1, §12.2): sent only by TerrainSystem (its single send point). The sink holds the terrain
    // of one space: a pyramid of another space or layout starts a new GPU terrain set (the sink drops the old set's
    // tiles and masks itself, no removal calls); the pyramid comes before any tile or mask of its space; a tile is
    // removed before its level would exceed tilesPerLevel, a mask before the masks would exceed grassMaskLayers
    virtual void setTerrainPyramid(SpaceId space, const TerrainPyramidData& pyramid) = 0;
    // A set tile is replaced in place
    virtual void setTerrainTile(const RegionId& region, const TerrainTileData& tile) = 0;
    virtual void removeTerrainTile(const RegionId& region)                           = 0;
    // A set mask is replaced in place
    virtual void setGrassMask(const ChunkId& chunk, const GrassMaskData& mask) = 0;
    virtual void removeGrassMask(const ChunkId& chunk)                         = 0;
    // E1: drop the set (the camera's space has no terrain): no terrain, grass or water until the next pyramid
    virtual void clearTerrain() = 0;
    // E2: absolute heights of a sub-rectangle (x0, z0, w x h samples of the tile's / pyramid's own grid, row-major, w
    // values per row; inside, asserted), copied at the call, queued behind a pending full copy; a region without a
    // layer or another space than the set's: ignored and counted
    virtual void patchTerrainTile(const RegionId& region, int x0, int z0, int w, int h, const int32_t* heights) = 0;
    virtual void patchTerrainPyramid(SpaceId space, int x0, int z0, int w, int h, const int32_t* heights)       = 0;
    // E8: the bytes sent but not yet recorded (uploads pending)
    virtual size_t terrainUploadsPending() const { return 0; }
};

}  // namespace Engine
}  // namespace Saiga
