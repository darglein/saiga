/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// VulkanRenderSink (docs/engine_design.md §10.1; S19b): the RenderSink over a RenderScene. The World's registrations
// take the IDs from Vulkan::kGameReservedIds upward; content the game registers directly on the scene (the
// prototype's trees, level geometry; scene()) takes IDs of the game range through RenderScene::reserveIds.

#pragma once

#include "saiga/engine/sync/RenderSink.h"
#include "saiga/export.h"

namespace Saiga
{
namespace Vulkan
{
class RenderScene;
}
namespace Engine
{
class SAIGA_ENGINE_API VulkanRenderSink : public RenderSink
{
   public:
    explicit VulkanRenderSink(Vulkan::RenderScene& scene) : renderScene(scene) {}

    // An identical (ID, name) registration again is a no-op (a second setRenderSink of this sink); another one asserts
    void registerMaterial(MaterialId id, const std::string& name, const MaterialData& data) override;
    void setSpaceSettings(SpaceId space, const SpaceRenderSettings& settings) override;
    void registerMesh(MeshId id, const std::string& name, std::shared_ptr<const MeshLods> lods,
                      MaterialId material) override;
    InstanceSetHandle addInstanceSet(const RegionId& region, const std::vector<StaticInstance>& instances,
                                     uint32_t capacity = 0) override;
    void updateInstances(InstanceSetHandle set, uint32_t first, const std::vector<StaticInstance>& v) override;
    void removeInstance(InstanceSetHandle set, uint32_t index) override;
    void removeInstanceSet(InstanceSetHandle set) override;
    InstanceHandle addDynamic(MeshId mesh) override;
    void setDynamic(InstanceHandle h, const ChunkId& chunk, const DynamicPose& prev, const DynamicPose& curr,
                    uint32_t boneOffset = 0) override;
    void removeDynamic(InstanceHandle h) override;
    void setCamera(SpaceId space, const Vec3& position, const Quat& rotation, double fovy) override;
    void setInterpolation(float alpha) override;
    void setEnvironment(const EnvironmentState& environment) override;
    void setTerrainPyramid(SpaceId space, const TerrainPyramidData& pyramid) override;
    void setTerrainTile(const RegionId& region, const TerrainTileData& tile) override;
    void removeTerrainTile(const RegionId& region) override;
    void setGrassMask(const ChunkId& chunk, const GrassMaskData& mask) override;
    void removeGrassMask(const ChunkId& chunk) override;
    void clearTerrain() override;
    void patchTerrainTile(const RegionId& region, int x0, int z0, int w, int h, const int32_t* heights) override;
    void patchTerrainPyramid(SpaceId space, int x0, int z0, int w, int h, const int32_t* heights) override;
    size_t terrainUploadsPending() const override;

    // The scene (the game's own content: instance sets of level geometry, cosmetic dynamics)
    Vulkan::RenderScene& scene() { return renderScene; }

   private:
    Vulkan::RenderScene& renderScene;
};

}  // namespace Engine
}  // namespace Saiga
