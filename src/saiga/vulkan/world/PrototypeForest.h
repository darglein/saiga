/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/scene/RenderSceneTypes.h"
#include "saiga/vulkan/world/FoliageGenerator.h"
#include "saiga/vulkan/world/ForestRenderer.h"
#include "saiga/vulkan/world/WorldSettings.h"

#include <array>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
class RenderScene;

// One instance set of the prototype's trees: a 256 m chunk (region level 0) with its trees, region-relative
struct PrototypeTreeSet
{
    RegionId region;
    std::vector<StaticInstance> instances;
};

/**
 * The prototype's trees as instance sets (S19a; from ForestRenderer::placeTrees): one set per 256 m chunk of space 0
 * (region level 0; the chunk index clamped at the world's edge), the instances in placeTrees order (deterministic keys:
 * generator slot 1, the placeTrees index as the counter), the rotation about y, the scale the tree height, the mesh
 * meshIds[species].
 */
// S20: the placed trees (ForestRenderer::placeTrees of a PrototypeArea) in the chunks of the prototype's space
SAIGA_VULKAN_API std::vector<PrototypeTreeSet> prototypeTreeSets(const std::vector<ForestRenderer::TreeInstance>& trees,
                                                                 const std::vector<TreeSpecies>& species,
                                                                 const std::vector<MeshId>& meshIds);

// The mesh of a tree species for the PrototypeFoliage class: its LODs on MeshVertex (the color word
// packPrototypeFoliageColor of the layer, AO and wind weight), the sphere (0, bounds.x, 0) with radius bounds.y (units
// of the tree height, as tree_cull.comp), the end distances and the bounds padding (the wind sway, 0.025 * wind)
SAIGA_VULKAN_API MeshLods prototypeTreeMesh(const TreeSpecies& species, const std::array<float, 3>& endDistances,
                                            float boundsPadding);

// The tree LOD end distances of the settings: the prototype's rule (LOD 0 below treeLod1, LOD 1 below treeLod2,
// culled beyond treeMaxDistance; the three are clamped independently) as non-decreasing end distances
// {min(l1, max), min(max(l1, l2), max), max} (equal distances: an empty LOD)
SAIGA_VULKAN_API std::array<float, 3> prototypeTreeLodEnds(const WorldSettings& settings);
// The bounds padding of the wind sway (tree_cull.comp: sway + flutter < 0.025 tree heights per wind unit)
inline float prototypeTreePadding(const WorldSettings& settings)
{
    return 0.025f * settings.wind;
}

/**
 * The prototype's trees in a RenderScene (S19a; used by game_world, flight, racing and the tests): registers one mesh
 * per species (IDs of RenderScene::reserveIds, names "prototype/tree/<seed>/<species>") with the PrototypeFoliage
 * material ("prototype/foliage"); on a scene that already has them (a PrototypeScene attached again) they are reused by
 * name. Creates the instance sets (prototypeTreeSets) and tracks the species meshes' statistics.
 * apply() follows the settings: the trees toggle removes and re-adds the sets, treeLod1/2/treeMaxDistance and wind
 * rewrite the mesh records (setMeshLodDistances).
 */
class SAIGA_VULKAN_API PrototypeForest
{
   public:
    // The scene must have the prototype terrain (the foliage texture of the PrototypeFoliage class). Once per object;
    // a new PrototypeForest on the same scene reuses the registered meshes and material
    // The placed trees (the species and mesh names of the seed; the scene needs setFoliageTexture)
    void init(RenderScene& scene, const std::vector<ForestRenderer::TreeInstance>& trees, uint32_t seed,
              const WorldSettings& settings);
    // The trees toggle, the LOD distances and the wind's bounds padding
    void apply(const WorldSettings& settings);
    // Removes the instance sets (the meshes stay registered); before the scene is destroyed nothing is needed
    void clear();

    struct Stats
    {
        uint64_t frame                  = 0;              // RenderScene's frame of the counts (0: none yet)
        std::array<uint32_t, 3> visible = {};             // drawn trees per LOD
        uint64_t triangles              = 0;              // drawn tree triangles
        std::vector<std::array<uint32_t, 3>> perSpecies;  // drawn trees per species and LOD
    };
    // From RenderScene::statistics(): the sums over the species meshes of the tracked per-LOD counts; none (frame 0)
    // before the first frame recorded after init (a forest attached again reuses the mesh IDs of the previous one)
    Stats stats();

    int treeCount() const { return trees; }
    int setCount() const { return int(sets.size()); }
    const std::vector<MeshId>& meshes() const { return meshIds; }
    MaterialId material() const { return materialId; }
    const std::vector<TreeSpecies>& treeSpecies() const { return species; }
    // GPU memory: the static instances and the meshes' vertices and indices
    size_t memoryBytes() const;

   private:
    void initMeshes(RenderScene& scene, uint32_t seed, const WorldSettings& settings);
    void initSets(const WorldSettings& settings);
    RenderScene* scene = nullptr;
    std::vector<TreeSpecies> species;
    std::vector<MeshId> meshIds;
    MaterialId materialId = 0;
    std::vector<PrototypeTreeSet> sets;
    std::vector<InstanceSetHandle> handles;
    int trees                 = 0;
    bool shown                = false;
    std::array<float, 3> ends = {0, 0, 0};
    float padding             = 0;
    // The first RenderScene frame recorded after init (the counts of earlier frames are not this forest's)
    uint64_t firstFrame = 0;
};

}  // namespace Vulkan
}  // namespace Saiga
