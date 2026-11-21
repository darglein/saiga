/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/world/PrototypeForest.h"

#include "saiga/vulkan/scene/PrototypeWorld.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/world/ForestRenderer.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>

namespace Saiga
{
namespace Vulkan
{
namespace
{
// The InstanceKey generator slot of the prototype's trees (upper 8 bits)
constexpr uint32_t treeKeySlot = 1;
}  // namespace

static std::vector<PrototypeTreeSet> prototypeTreeSetsOf(const std::vector<ForestRenderer::TreeInstance>& trees,
                                                         const std::vector<TreeSpecies>& species,
                                                         const std::vector<MeshId>& meshIds, int chunks);

std::vector<PrototypeTreeSet> prototypeTreeSets(const HeightField& hf, int treeCount, uint32_t seed,
                                                const std::vector<TreeSpecies>& species,
                                                const std::vector<MeshId>& meshIds)
{
    // The chunks of the world (the last one may be partial; a tree at x = worldSize belongs to the last)
    const int chunks = std::max(1, int(std::ceil(double(hf.worldSize()) / chunkSize)));
    return prototypeTreeSetsOf(ForestRenderer::placeTrees(hf, treeCount, seed, species), species, meshIds, chunks);
}

std::vector<PrototypeTreeSet> prototypeTreeSets(const std::vector<ForestRenderer::TreeInstance>& trees,
                                                const std::vector<TreeSpecies>& species,
                                                const std::vector<MeshId>& meshIds)
{
    return prototypeTreeSetsOf(trees, species, meshIds, kPrototypeSpaceChunks);
}

static std::vector<PrototypeTreeSet> prototypeTreeSetsOf(const std::vector<ForestRenderer::TreeInstance>& trees,
                                                         const std::vector<TreeSpecies>& species,
                                                         const std::vector<MeshId>& meshIds, int chunks)
{
    SAIGA_ASSERT(meshIds.size() == species.size(), "prototypeTreeSets: one mesh per species");
    SAIGA_ASSERT(trees.size() < (1u << 24), "prototypeTreeSets: the keys' counters have 24 bits");
    auto chunkOf = [&](float v) { return std::min(std::max(int(std::floor(double(v) / chunkSize)), 0), chunks - 1); };

    std::map<std::pair<int, int>, size_t> index;
    std::vector<PrototypeTreeSet> sets;
    for (size_t i = 0; i < trees.size(); ++i)
    {
        const auto& t = trees[i];
        const int cx = chunkOf(t.posYaw(0)), cz = chunkOf(t.posYaw(2));
        auto it = index.find({cx, cz});
        if (it == index.end())
        {
            PrototypeTreeSet s;
            s.region = RegionId{0, 0, cx, cz};
            it       = index.insert({{cx, cz}, sets.size()}).first;
            sets.push_back(std::move(s));
        }
        const Vec3 origin = regionOrigin(sets[it->second].region);
        const int sp      = int(t.scaleSpecies(1));
        StaticInstance inst;
        // Region-relative (exact: the chunk origins are multiples of 256 m)
        inst.setPosition(
            vec3(float(double(t.posYaw(0)) - origin(0)), t.posYaw(1), float(double(t.posYaw(2)) - origin(2))));
        // The prototype's rotation about y by the yaw (tree.vert's rotateY)
        const float yaw = t.posYaw(3);
        inst.setRotation(quat(std::cos(yaw * 0.5f), 0, std::sin(yaw * 0.5f), 0));
        inst.scale = t.scaleSpecies(0);
        inst.mesh  = meshIds[size_t(sp)];
        inst.key   = (treeKeySlot << 24) | uint32_t(i);
        sets[it->second].instances.push_back(inst);
    }
    // Deterministic order of the sets (by chunk)
    std::sort(sets.begin(), sets.end(),
              [](const PrototypeTreeSet& a, const PrototypeTreeSet& b) { return a.region < b.region; });
    return sets;
}

MeshLods prototypeTreeMesh(const TreeSpecies& species, const std::array<float, 3>& endDistances, float boundsPadding)
{
    MeshLods m;
    for (int l = 0; l < 3; ++l)
    {
        const FoliageMesh& src = species.lods[size_t(l)];
        MeshLod lod;
        lod.endDistance = endDistances[size_t(l)];
        lod.indices     = src.indices;
        lod.vertices.reserve(src.vertices.size());
        for (const TreeVertex& v : src.vertices)
        {
            MeshVertex o;
            o.position      = vec3(v.posAo(0), v.posAo(1), v.posAo(2));
            o.normal        = packNormalOct(vec3(v.normalWind(0), v.normalWind(1), v.normalWind(2)).normalized());
            const int layer = v.uvLayer(2) < 0 ? -1 : int(std::lround(v.uvLayer(2)));
            o.color         = packPrototypeFoliageColor(layer, v.posAo(3), v.normalWind(3));
            o.uv            = vec2(v.uvLayer(0), v.uvLayer(1));
            lod.vertices.push_back(o);
        }
        m.lods.push_back(std::move(lod));
    }
    // tree_cull.comp's sphere: (0, bounds.x, 0) with radius bounds.y, in units of the tree height
    m.sphereCenter  = vec3(0, species.bounds(0), 0);
    m.sphereRadius  = species.bounds(1);
    m.boundsPadding = boundsPadding;
    return m;
}

std::array<float, 3> prototypeTreeLodEnds(const WorldSettings& s)
{
    const float l1 = s.treeLod1, l2 = s.treeLod2, max = s.treeMaxDistance;
    return {std::min(l1, max), std::min(std::max(l1, l2), max), max};
}

void PrototypeForest::init(RenderScene& _scene, const HeightField& hf, int treeCount, uint32_t seed,
                           const WorldSettings& settings)
{
    initMeshes(_scene, seed, settings);
    sets = prototypeTreeSets(hf, treeCount, seed, species, meshIds);
    initSets(settings);
}

void PrototypeForest::init(RenderScene& _scene, const std::vector<ForestRenderer::TreeInstance>& placed, uint32_t seed,
                           const WorldSettings& settings)
{
    initMeshes(_scene, seed, settings);
    sets = prototypeTreeSets(placed, species, meshIds);
    initSets(settings);
}

void PrototypeForest::initSets(const WorldSettings& settings)
{
    trees = 0;
    for (const auto& s : sets) trees += int(s.instances.size());
    shown = false;
    apply(settings);
}

void PrototypeForest::initMeshes(RenderScene& _scene, uint32_t seed, const WorldSettings& settings)
{
    SAIGA_ASSERT(!scene, "PrototypeForest::init: once per object");
    SAIGA_ASSERT(_scene.hasPrototypeTerrain() || _scene.hasPrototypeEnvironment(),
                 "PrototypeForest: the scene needs setPrototypeEnvironment (foliage texture)");
    scene      = &_scene;
    firstFrame = scene->frames() + 1;
    species    = FoliageGenerator::treeSpecies(seed);
    ends       = prototypeTreeLodEnds(settings);
    padding    = prototypeTreePadding(settings);

    // A scene keeps the meshes and the material after the forest is gone (RenderScene removes neither): a forest
    // attached to the same scene again reuses them by name (no reserved IDs leak, no name assert). The species meshes
    // depend on the seed, which is part of their names
    materialId = scene->findMaterial("prototype/foliage");
    if (materialId == 0)
    {
        MaterialData material;
        material.pipelineClass = uint8_t(PipelineClass::PrototypeFoliage);
        material.twoSided      = true;
        materialId             = MaterialId(scene->reserveIds(SceneIdKind::Material, 1));
        scene->addMaterial(materialId, "prototype/foliage", material);
    }
    meshIds.clear();
    for (size_t s = 0; s < species.size(); ++s)
    {
        const std::string name = "prototype/tree/" + std::to_string(seed) + "/" + species[s].name;
        MeshId id              = scene->findMesh(name);
        if (id != 0)
            scene->setMeshLodDistances(id, {ends[0], ends[1], ends[2]}, padding);
        else
        {
            id = MeshId(scene->reserveIds(SceneIdKind::Mesh, 1));
            scene->addMesh(id, name, prototypeTreeMesh(species[s], ends, padding), materialId);
        }
        meshIds.push_back(id);
    }
    scene->trackMeshStatistics(meshIds);
}

void PrototypeForest::apply(const WorldSettings& settings)
{
    if (!scene) return;
    // The trees toggle: the sets are removed and re-added (the meshes stay)
    if (settings.trees != shown)
    {
        if (settings.trees)
        {
            handles.clear();
            for (const auto& s : sets) handles.push_back(scene->addInstanceSet(s.region, s.instances));
        }
        else
            clear();
        shown = settings.trees;
    }
    const auto newEnds     = prototypeTreeLodEnds(settings);
    const float newPadding = prototypeTreePadding(settings);
    if (newEnds != ends || newPadding != padding)
    {
        ends    = newEnds;
        padding = newPadding;
        for (MeshId id : meshIds) scene->setMeshLodDistances(id, {ends[0], ends[1], ends[2]}, padding);
    }
}

void PrototypeForest::clear()
{
    if (scene)
        for (InstanceSetHandle h : handles) scene->removeInstanceSet(h);
    handles.clear();
    shown = false;
}

PrototypeForest::Stats PrototypeForest::stats()
{
    Stats r;
    if (!scene) return r;
    r.perSpecies.assign(species.size(), {0, 0, 0});
    const RenderStatistics& st = scene->statistics();
    // A forest attached again (the same mesh IDs) took the previous forest's counts (S19a review)
    if (st.frame < firstFrame) return r;
    r.frame = st.frame;
    for (const auto& t : st.trackedMeshes)
    {
        auto it = std::find(meshIds.begin(), meshIds.end(), t.mesh);
        if (it == meshIds.end()) continue;
        const size_t index    = size_t(it - meshIds.begin());
        const TreeSpecies& sp = species[index];
        for (int l = 0; l < 3; ++l)
        {
            r.perSpecies[index][size_t(l)] = t.instancesPerLod[size_t(l)];
            r.visible[size_t(l)] += t.instancesPerLod[size_t(l)];
            r.triangles += uint64_t(t.instancesPerLod[size_t(l)]) * uint64_t(sp.lods[size_t(l)].triangles());
        }
    }
    return r;
}

size_t PrototypeForest::memoryBytes() const
{
    size_t bytes = size_t(trees) * sizeof(StaticInstance);
    for (const auto& sp : species)
        for (const auto& l : sp.lods) bytes += l.vertices.size() * sizeof(MeshVertex) + l.indices.size() * 4;
    return bytes;
}

}  // namespace Vulkan
}  // namespace Saiga
