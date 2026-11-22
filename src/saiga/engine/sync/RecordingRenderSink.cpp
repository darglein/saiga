/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// RecordingRenderSink: the RenderSink of tests and scenarios (state and counters, an opt-in call log).

#include "saiga/engine/sync/RecordingRenderSink.h"

#include "saiga/core/util/assert.h"
#include "saiga/vulkan/scene/TerrainAtlas.h"

#include <algorithm>

namespace Saiga
{
namespace Engine
{
void RecordingRenderSink::registerMaterial(MaterialId id, const std::string& name, const MaterialData&)
{
    auto it = materialNames.find(id);
    if (it != materialNames.end())
    {
        // The same sink set again: identical registrations are no-ops
        SAIGA_ASSERT(it->second == name, "RecordingRenderSink: material " + std::to_string(id) + " registered as \"" +
                                             it->second + "\" and \"" + name + "\"");
        return;
    }
    for (const auto& m : materialNames)
        SAIGA_ASSERT(m.second != name, "RecordingRenderSink: material \"" + name + "\" registered with two IDs");
    materialNames[id] = name;
    count.registerMaterial++;
    if (logging) note("registerMaterial " + std::to_string(id) + " " + name);
}

void RecordingRenderSink::registerMesh(MeshId id, const std::string& name, std::shared_ptr<const MeshLods> lods,
                                       MaterialId material)
{
    SAIGA_ASSERT(lods && !lods->lods.empty(), "RecordingRenderSink: mesh \"" + name + "\" without LODs");
    auto it = meshNames.find(id);
    if (it != meshNames.end())
    {
        SAIGA_ASSERT(it->second == name, "RecordingRenderSink: mesh " + std::to_string(id) + " registered as \"" +
                                             it->second + "\" and \"" + name + "\"");
        return;
    }
    for (const auto& m : meshNames)
        SAIGA_ASSERT(m.second != name, "RecordingRenderSink: mesh \"" + name + "\" registered with two IDs");
    SAIGA_ASSERT(materialNames.count(material), "RecordingRenderSink: mesh \"" + name + "\" of an unknown material");
    meshNames[id]     = name;
    meshMaterials[id] = material;
    count.registerMesh++;
    if (logging) note("registerMesh " + std::to_string(id) + " " + name);
}

MaterialId RecordingRenderSink::meshMaterial(MeshId id) const
{
    auto it = meshMaterials.find(id);
    return it == meshMaterials.end() ? MaterialId(0) : it->second;
}

InstanceSetHandle RecordingRenderSink::addInstanceSet(const RegionId&, const std::vector<StaticInstance>& instances,
                                                      uint32_t capacity)
{
    for (const StaticInstance& i : instances)
        SAIGA_ASSERT(meshNames.count(i.mesh), "RecordingRenderSink: an instance of an unregistered mesh");
    InstanceSetHandle h;
    h.id       = nextSet++;
    sets[h.id] = capacity ? capacity : uint32_t(instances.size());
    count.addInstanceSet++;
    if (logging) note("addInstanceSet -> " + std::to_string(h.id));
    return h;
}

void RecordingRenderSink::updateInstances(InstanceSetHandle set, uint32_t first, const std::vector<StaticInstance>& v)
{
    auto it = sets.find(set.id);
    SAIGA_ASSERT(it != sets.end() && first + v.size() <= it->second, "RecordingRenderSink: updateInstances");
    count.updateInstances++;
    if (logging) note("updateInstances " + std::to_string(set.id));
}

void RecordingRenderSink::removeInstance(InstanceSetHandle set, uint32_t index)
{
    auto it = sets.find(set.id);
    SAIGA_ASSERT(it != sets.end() && index < it->second, "RecordingRenderSink: removeInstance");
    count.removeInstance++;
    if (logging) note("removeInstance " + std::to_string(set.id));
}

void RecordingRenderSink::removeInstanceSet(InstanceSetHandle set)
{
    // Erased in every build (SAIGA_ASSERT does not evaluate its condition without SAIGA_ASSERTS)
    const size_t erased = sets.erase(set.id);
    SAIGA_ASSERT(erased == 1, "RecordingRenderSink: removeInstanceSet of an unknown set");
    count.removeInstanceSet++;
    if (logging) note("removeInstanceSet " + std::to_string(set.id));
}

InstanceHandle RecordingRenderSink::addDynamic(MeshId mesh)
{
    SAIGA_ASSERT(meshNames.count(mesh),
                 "RecordingRenderSink: addDynamic of the unregistered mesh " + std::to_string(mesh));
    InstanceHandle h;
    h.id = nextHandle;
    std::string line;
    if (logging) line = prepareNote("addDynamic " + std::to_string(mesh) + " -> " + std::to_string(h.id));
    live.emplace(h.id, Dynamic{mesh, false, ChunkId(), DynamicPose(), DynamicPose(), 0});
    nextHandle++;
    count.addDynamic++;
    if (logging) calls.push_back(std::move(line));
    return h;
}

void RecordingRenderSink::setDynamic(InstanceHandle h, const ChunkId& chunk, const DynamicPose& prev,
                                     const DynamicPose& curr, uint32_t)
{
    auto it = live.find(h.id);
    SAIGA_ASSERT(it != live.end(), "RecordingRenderSink: setDynamic of a removed handle");
    std::string line;
    if (logging) line = prepareNote("setDynamic " + std::to_string(h.id));
    it->second.set   = true;
    it->second.chunk = chunk;
    it->second.prev  = prev;
    it->second.curr  = curr;
    it->second.updates++;
    count.setDynamic++;
    if (logging) calls.push_back(std::move(line));
}

void RecordingRenderSink::removeDynamic(InstanceHandle h)
{
    std::string line;
    if (logging) line = prepareNote("removeDynamic " + std::to_string(h.id));
    // Erased in every build (SAIGA_ASSERT does not evaluate its condition without SAIGA_ASSERTS)
    const size_t erased = live.erase(h.id);
    SAIGA_ASSERT(erased == 1, "RecordingRenderSink: removeDynamic of an unknown handle");
    count.removeDynamic++;
    if (logging) calls.push_back(std::move(line));
}

const RecordingRenderSink::Dynamic* RecordingRenderSink::dynamic(InstanceHandle h) const
{
    auto it = live.find(h.id);
    return it == live.end() ? nullptr : &it->second;
}

void RecordingRenderSink::setCamera(SpaceId, const Vec3&, const Quat&, double)
{
    count.setCamera++;
}

void RecordingRenderSink::setInterpolation(float a)
{
    alpha = a;
    count.setInterpolation++;
}

// ---- terrain (S20)

void RecordingRenderSink::setTerrainPyramid(SpaceId space, const TerrainPyramidData& pyramid)
{
    const int P = pyramid.layout.pyramidSamplesPerSide();
    SAIGA_ASSERT(pyramid.heights && pyramid.heights->size() == size_t(P) * size_t(P),
                 "RecordingRenderSink: wrong pyramid sample count");
    if (!terrainSet || space != terrainSetSpace || pyramid.layout != terrainSetLayout)
    {
        // A new set: the old tiles and masks are dropped (as RenderScene does)
        tiles.clear();
        masks.clear();
    }
    terrainSet       = true;
    terrainSetSpace  = space;
    terrainSetLayout = pyramid.layout;
    pyramidData      = pyramid;
    count.setTerrainPyramid++;
    if (logging) note("setTerrainPyramid " + std::to_string(space));
}

void RecordingRenderSink::setTerrainTile(const RegionId& region, const TerrainTileData& tile)
{
    SAIGA_ASSERT(terrainSet && region.space == terrainSetSpace,
                 "RecordingRenderSink: a terrain tile before its space's pyramid");
    SAIGA_ASSERT(region.level < terrainSetLayout.streamedLevels(), "RecordingRenderSink: not a streamed level");
    const size_t n = size_t(terrainSetLayout.samplesPerSide());
    SAIGA_ASSERT(tile.heights && tile.heights->size() == n * n, "RecordingRenderSink: wrong tile sample count");
    const bool replace = tiles.count(region) > 0;
    tiles[region]      = tile;
    SAIGA_ASSERT(replace || terrainTilesAtLevel(region.level) <= terrainSetLayout.tilesPerLevel,
                 "RecordingRenderSink: more tiles than tilesPerLevel on a level");
    count.setTerrainTile++;
    if (logging)
        note("setTerrainTile L" + std::to_string(region.level) + " " + std::to_string(region.x) + " " +
             std::to_string(region.z));
}

void RecordingRenderSink::removeTerrainTile(const RegionId& region)
{
    tiles.erase(region);
    count.removeTerrainTile++;
    if (logging)
        note("removeTerrainTile L" + std::to_string(region.level) + " " + std::to_string(region.x) + " " +
             std::to_string(region.z));
}

void RecordingRenderSink::setGrassMask(const ChunkId& chunk, const GrassMaskData& mask)
{
    SAIGA_ASSERT(terrainSet && chunk.space == terrainSetSpace,
                 "RecordingRenderSink: a grass mask before its space's pyramid");
    SAIGA_ASSERT(mask.density && mask.size == terrainSetLayout.grassMaskSize(), "RecordingRenderSink: wrong mask size");
    const bool replace = masks.count(chunk) > 0;
    masks[chunk]       = mask;
    SAIGA_ASSERT(replace || int(masks.size()) <= terrainSetLayout.grassMaskLayers,
                 "RecordingRenderSink: more masks than grassMaskLayers");
    count.setGrassMask++;
    if (logging) note("setGrassMask " + std::to_string(chunk.x) + " " + std::to_string(chunk.z));
}

void RecordingRenderSink::removeGrassMask(const ChunkId& chunk)
{
    masks.erase(chunk);
    count.removeGrassMask++;
    if (logging) note("removeGrassMask " + std::to_string(chunk.x) + " " + std::to_string(chunk.z));
}

void RecordingRenderSink::setSpaceSettings(SpaceId space, const SpaceRenderSettings& settings)
{
    spaces[space] = settings;
    spaceCalls[space]++;
    count.setSpaceSettings++;
    if (logging) note("setSpaceSettings " + std::to_string(space) + (settings.openWorld ? " open" : " enclosed"));
}

void RecordingRenderSink::clearTerrain()
{
    terrainSet  = false;
    pyramidData = TerrainPyramidData();
    tiles.clear();
    masks.clear();
    count.clearTerrain++;
    if (logging) note("clearTerrain");
}

namespace
{
// A copy of the heights with the sub-rectangle written (copy on write: the store's vectors stay base data)
std::shared_ptr<const std::vector<int32_t>> patched(const std::vector<int32_t>& base, int n, int x0, int z0, int w,
                                                    int h, const int32_t* heights)
{
    auto copy = std::make_shared<std::vector<int32_t>>(base);
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i) (*copy)[size_t(z0 + j) * size_t(n) + size_t(x0 + i)] = heights[j * w + i];
    return copy;
}
}  // namespace

void RecordingRenderSink::patchTerrainTile(const RegionId& region, int x0, int z0, int w, int h, const int32_t* heights)
{
    count.patchTerrainTile++;
    if (logging)
        note("patchTerrainTile L" + std::to_string(region.level) + " " + std::to_string(region.x) + " " +
             std::to_string(region.z));
    auto it = tiles.find(region);
    if (!terrainSet || region.space != terrainSetSpace || it == tiles.end())
    {
        patchesIgnored++;
        return;
    }
    const int n = terrainSetLayout.samplesPerSide();
    SAIGA_ASSERT(w > 0 && h > 0 && x0 >= 0 && z0 >= 0 && x0 + w <= n && z0 + h <= n,
                 "RecordingRenderSink: the patch must lie inside the tile");
    // The extrema before the assignment: the input may alias the recorded heights, which it releases
    for (int k = 0; k < w * h; ++k)
    {
        it->second.minHeight = std::min(it->second.minHeight, heights[k]);
        it->second.maxHeight = std::max(it->second.maxHeight, heights[k]);
    }
    it->second.heights = patched(*it->second.heights, n, x0, z0, w, h, heights);
}

void RecordingRenderSink::patchTerrainPyramid(SpaceId space, int x0, int z0, int w, int h, const int32_t* heights)
{
    count.patchTerrainPyramid++;
    if (logging) note("patchTerrainPyramid " + std::to_string(space));
    if (!terrainSet || space != terrainSetSpace)
    {
        patchesIgnored++;
        return;
    }
    const int P = terrainSetLayout.pyramidSamplesPerSide();
    SAIGA_ASSERT(w > 0 && h > 0 && x0 >= 0 && z0 >= 0 && x0 + w <= P && z0 + h <= P,
                 "RecordingRenderSink: the patch must lie inside the pyramid");
    pyramidData.heights = patched(*pyramidData.heights, P, x0, z0, w, h, heights);
}

int RecordingRenderSink::terrainTilesAtLevel(int level) const
{
    int n = 0;
    for (const auto& t : tiles) n += t.first.level == level ? 1 : 0;
    return n;
}

size_t RecordingRenderSink::terrainGpuBytes() const
{
    return terrainSet ? Vulkan::TerrainAtlas::memoryEstimate(terrainSetLayout).gpuBytes() : 0;
}

size_t RecordingRenderSink::terrainSceneCpuBytes() const
{
    return terrainSet ? Vulkan::TerrainAtlas::memoryEstimate(terrainSetLayout).cpuBytes() : 0;
}

}  // namespace Engine
}  // namespace Saiga
