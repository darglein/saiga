/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/world/World.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"

#include "games/common/Mesh.h"

#include <functional>
#include <memory>
#include <string>

namespace Saiga
{
namespace Vulkan
{
class RenderScene;
}
namespace Game
{
// The tech demos' meshes for RenderScene (S19b): the procedural MeshData as MeshLods (one LOD up to endDistance; the
// vertex colors, linear in MeshData, as sRGB8 color words)
std::shared_ptr<const Vulkan::MeshLods> meshLods(const MeshData& mesh, float endDistance);

// The demos' material: flat color, two-sided (some builders are not consistently wound; the normals follow the winding)
Vulkan::MaterialData demoMaterial();

// A mesh of WorldSetup::renderAssets: built once per process (a static cache by name, shared by every world of the
// process, e.g. the scenario worlds) and registered in the world
Vulkan::MeshId registerDemoMesh(Engine::World& world, const std::string& name, const std::function<MeshData()>& build,
                                Vulkan::MaterialId material, float endDistance = 3000);

// Constant level geometry (the platformer's level, the race track; S19b): one mesh registered directly on the scene
// (an ID of the game range, RenderScene::reserveIds; its material too) and one instance set in the region of level 0
// of the space (S21) that contains origin, vertices relative to that region's origin; the single LOD ends beyond the
// mesh's extent (no LOD-out by the distance to its origin). The game owns the handle (release() in shutdown)
class LevelGeometry
{
   public:
    void create(Vulkan::RenderScene& scene, const std::string& name, const MeshData& mesh, const Vec3& origin,
                SpaceId space = 0);
    void release();

   private:
    Vulkan::RenderScene* scene = nullptr;
    Vulkan::InstanceSetHandle set;
};

}  // namespace Game
}  // namespace Saiga
