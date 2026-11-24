/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// VulkanRenderSink: the RenderSink over RenderScene.

#include "saiga/engine/sync/VulkanRenderSink.h"

#include "saiga/core/util/assert.h"
#include "saiga/vulkan/scene/RenderScene.h"

namespace Saiga
{
namespace Engine
{
void VulkanRenderSink::registerMaterial(MaterialId id, const std::string& name, const MaterialData& data)
{
    const MaterialId existing = renderScene.findMaterial(name);
    if (existing == id) return;
    SAIGA_ASSERT(existing == 0, "VulkanRenderSink: the material \"" + name + "\" is registered with the ID " +
                                    std::to_string(existing) + ", not " + std::to_string(id));
    renderScene.addMaterial(id, name, data);
}

void VulkanRenderSink::registerMesh(MeshId id, const std::string& name, std::shared_ptr<const MeshLods> lods,
                                    MaterialId material)
{
    SAIGA_ASSERT(lods, "VulkanRenderSink: the mesh \"" + name + "\" has no data");
    const MeshId existing = renderScene.findMesh(name);
    if (existing == id) return;
    SAIGA_ASSERT(existing == 0, "VulkanRenderSink: the mesh \"" + name + "\" is registered with the ID " +
                                    std::to_string(existing) + ", not " + std::to_string(id));
    renderScene.addMesh(id, name, *lods, material);
}

InstanceSetHandle VulkanRenderSink::addInstanceSet(const RegionId& region, const std::vector<StaticInstance>& instances,
                                                   uint32_t capacity)
{
    return renderScene.addInstanceSet(region, instances, capacity);
}

void VulkanRenderSink::updateInstances(InstanceSetHandle set, uint32_t first, const std::vector<StaticInstance>& v)
{
    renderScene.updateInstances(set, first, v);
}

void VulkanRenderSink::removeInstance(InstanceSetHandle set, uint32_t index)
{
    renderScene.removeInstance(set, index);
}

void VulkanRenderSink::removeInstanceSet(InstanceSetHandle set)
{
    renderScene.removeInstanceSet(set);
}

InstanceHandle VulkanRenderSink::addDynamic(MeshId mesh)
{
    return renderScene.addDynamic(mesh);
}

void VulkanRenderSink::setDynamic(InstanceHandle h, const ChunkId& chunk, const DynamicPose& prev,
                                  const DynamicPose& curr, uint32_t boneOffset)
{
    renderScene.setDynamic(h, chunk, prev, curr, boneOffset);
}

void VulkanRenderSink::removeDynamic(InstanceHandle h)
{
    renderScene.removeDynamic(h);
}

void VulkanRenderSink::setCamera(SpaceId space, const Vec3& position, const Quat& rotation, double fovy)
{
    renderScene.setCamera(space, position, rotation, fovy);
}

void VulkanRenderSink::setInterpolation(float alpha)
{
    renderScene.setInterpolation(alpha);
}

// ---- terrain (S20): forwarded to the scene's GPU terrain set

void VulkanRenderSink::setTerrainPyramid(SpaceId space, const TerrainPyramidData& pyramid)
{
    renderScene.setTerrainPyramid(space, pyramid);
}

void VulkanRenderSink::setTerrainTile(const RegionId& region, const TerrainTileData& tile)
{
    renderScene.setTerrainTile(region, tile);
}

void VulkanRenderSink::removeTerrainTile(const RegionId& region)
{
    renderScene.removeTerrainTile(region);
}

void VulkanRenderSink::setGrassMask(const ChunkId& chunk, const GrassMaskData& mask)
{
    renderScene.setGrassMask(chunk, mask);
}

void VulkanRenderSink::removeGrassMask(const ChunkId& chunk)
{
    renderScene.removeGrassMask(chunk);
}

void VulkanRenderSink::setSpaceSettings(SpaceId space, const SpaceRenderSettings& settings)
{
    renderScene.setSpaceSettings(space, settings);
}

void VulkanRenderSink::setEnvironment(const EnvironmentState& environment)
{
    renderScene.setEnvironment(environment);
}

void VulkanRenderSink::clearTerrain()
{
    renderScene.clearTerrain();
}

void VulkanRenderSink::patchTerrainTile(const RegionId& region, int x0, int z0, int w, int h, const int32_t* heights)
{
    renderScene.patchTerrainTile(region, x0, z0, w, h, heights);
}

void VulkanRenderSink::patchTerrainPyramid(SpaceId space, int x0, int z0, int w, int h, const int32_t* heights)
{
    renderScene.patchTerrainPyramid(space, x0, z0, w, h, heights);
}

size_t VulkanRenderSink::terrainUploadsPending() const
{
    return renderScene.terrainUploadsPending();
}

}  // namespace Engine
}  // namespace Saiga
