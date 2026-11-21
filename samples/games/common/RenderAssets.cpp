/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The tech demos' meshes as RenderScene content (S19b).

#include "games/common/RenderAssets.h"

#include "saiga/vulkan/scene/RenderScene.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>

namespace Saiga
{
namespace Game
{
namespace
{
uint8_t srgb8(double linear)
{
    const double c = std::min(std::max(linear, 0.0), 1.0);
    const double s = c <= 0.0031308 ? c * 12.92 : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
    return uint8_t(std::lround(s * 255));
}

// The mesh with its vertices relative to origin (MeshData is in mesh space)
Vulkan::MeshLods convert(const MeshData& mesh, float endDistance, const Vec3& origin)
{
    Vulkan::MeshLods lods;
    Vulkan::MeshLod lod;
    lod.endDistance = endDistance;
    lod.vertices.reserve(mesh.vertices.size());
    vec3 lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
    for (const MeshVertex& v : mesh.vertices)
    {
        Vulkan::MeshVertex m;
        for (int i = 0; i < 3; ++i) m.position(i) = float(double(v.position[i]) - origin(i));
        m.normal = Vulkan::packNormalOct(vec3(v.normal[0], v.normal[1], v.normal[2]));
        m.color  = Vulkan::packColor(ucvec4(srgb8(v.color[0]), srgb8(v.color[1]), srgb8(v.color[2]), 255));
        lo       = lo.cwiseMin(m.position);
        hi       = hi.cwiseMax(m.position);
        lod.vertices.push_back(m);
    }
    lod.indices = mesh.indices;
    lods.lods.push_back(std::move(lod));
    lods.sphereCenter = (lo + hi) * 0.5f;
    lods.sphereRadius = (hi - lo).norm() * 0.5f;
    return lods;
}
}  // namespace

std::shared_ptr<const Vulkan::MeshLods> meshLods(const MeshData& mesh, float endDistance)
{
    return std::make_shared<const Vulkan::MeshLods>(convert(mesh, endDistance, Vec3::Zero()));
}

Vulkan::MaterialData demoMaterial()
{
    Vulkan::MaterialData m;
    m.model    = Vulkan::ShadingModel::FlatColor;
    m.twoSided = true;
    return m;
}

Vulkan::MeshId registerDemoMesh(Engine::World& world, const std::string& name, const std::function<MeshData()>& build,
                                Vulkan::MaterialId material, float endDistance)
{
    // Built once per process: the scenario worlds of a run share the data
    static std::mutex mutex;
    static std::map<std::string, std::shared_ptr<const Vulkan::MeshLods>> cache;
    std::shared_ptr<const Vulkan::MeshLods> lods;
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto& entry = cache[name];
        if (!entry) entry = meshLods(build(), endDistance);
        lods = entry;
    }
    return world.registerMesh(name, std::move(lods), material);
}

void LevelGeometry::create(Vulkan::RenderScene& s, const std::string& name, const MeshData& mesh, const Vec3& origin,
                           SpaceId space)
{
    release();
    scene = &s;
    RegionId region;
    region.space          = space;
    region.level          = 0;
    region.x              = int32_t(std::floor(origin(0) / chunkSize));
    region.z              = int32_t(std::floor(origin(2) / chunkSize));
    Vulkan::MeshLods lods = convert(mesh, 1, regionOrigin(region));
    // Beyond the mesh's extent: the distance is measured to the bounding sphere's centre
    lods.lods[0].endDistance = 2 * lods.sphereRadius + 20000;
    // The scene keeps registered content (a second init re-uses it by name)
    Vulkan::MeshId id = s.findMesh(name);
    if (!id)
    {
        Vulkan::MaterialId material = s.findMaterial("demo/level");
        if (!material)
        {
            material = s.reserveIds(Vulkan::SceneIdKind::Material, 1);
            s.addMaterial(material, "demo/level", demoMaterial());
        }
        id = s.reserveIds(Vulkan::SceneIdKind::Mesh, 1);
        s.addMesh(id, name, lods, material);
    }
    Vulkan::StaticInstance instance;
    instance.mesh = id;
    set           = s.addInstanceSet(region, {instance});
}

void LevelGeometry::release()
{
    if (scene && set) scene->removeInstanceSet(set);
    set   = Vulkan::InstanceSetHandle();
    scene = nullptr;
}

}  // namespace Game
}  // namespace Saiga
