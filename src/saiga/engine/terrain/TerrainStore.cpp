/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/terrain/TerrainStore.h"

#include "saiga/core/util/assert.h"

#include <iterator>

namespace Saiga
{
namespace Engine
{
TerrainStore::TerrainStore() : owner(std::this_thread::get_id()) {}

void TerrainStore::checkThread() const
{
    SAIGA_ASSERT(std::this_thread::get_id() == owner, "TerrainStore: mutated off the main thread");
}

const TerrainStore::Tile* TerrainStore::tile(const RegionId& region) const
{
    auto it = tileMap.find(region);
    return it == tileMap.end() ? nullptr : &it->second;
}

void TerrainStore::addTile(const RegionId& region, Vulkan::TerrainTileData data, uint64_t frame)
{
    checkThread();
    auto [it, added] = tileMap.try_emplace(region);
    // A pinned tile keeps its data (S24: the residency's base heights are bit-equal to the visual job's)
    if (added || !it->second.pinned) it->second.data = std::move(data);
    it->second.lastDesired = std::max(it->second.lastDesired, frame);
    if (!it->second.visual)
    {
        it->second.visual = true;
        perLevel[region.level]++;
    }
}

void TerrainStore::removeTile(const RegionId& region)
{
    checkThread();
    auto it = tileMap.find(region);
    if (it == tileMap.end()) return;
    if (it->second.visual)
    {
        it->second.visual = false;
        perLevel[region.level]--;
    }
    if (!it->second.pinned) tileMap.erase(it);
}

void TerrainStore::markDesired(const RegionId& region, uint64_t frame)
{
    checkThread();
    auto it = tileMap.find(region);
    if (it == tileMap.end()) return;
    it->second.lastDesired = frame;
    if (!it->second.visual)
    {
        it->second.visual = true;
        perLevel[region.level]++;
    }
}

void TerrainStore::pinTile(const RegionId& region, Vulkan::TerrainTileData data)
{
    checkThread();
    auto [it, added] = tileMap.try_emplace(region);
    if (added) it->second.data = std::move(data);
    if (!it->second.pinned)
    {
        it->second.pinned = true;
        pinnedCount++;
    }
}

void TerrainStore::unpinTile(const RegionId& region)
{
    checkThread();
    auto it = tileMap.find(region);
    if (it == tileMap.end() || !it->second.pinned) return;
    it->second.pinned = false;
    pinnedCount--;
    if (!it->second.visual) tileMap.erase(it);
}

size_t TerrainStore::pinnedBytes() const
{
    size_t b = 0;
    for (const auto& t : tileMap)
        if (t.second.pinned && t.second.data.heights) b += t.second.data.heights->size() * 4;
    return b;
}

int TerrainStore::tilesAtLevel(int level) const
{
    return perLevel[size_t(level)];
}

void TerrainStore::setPyramid(Vulkan::TerrainPyramidData pyramid)
{
    checkThread();
    if (pyramidData)
        *pyramidData = std::move(pyramid);
    else
        pyramidData = std::make_unique<Vulkan::TerrainPyramidData>(std::move(pyramid));
}

const Vulkan::GrassMaskData* TerrainStore::mask(const ChunkId& chunk) const
{
    auto it = maskMap.find(chunk);
    return it == maskMap.end() ? nullptr : &it->second;
}

void TerrainStore::addMask(const ChunkId& chunk, Vulkan::GrassMaskData mask)
{
    checkThread();
    maskMap[chunk] = std::move(mask);
}

void TerrainStore::removeMask(const ChunkId& chunk)
{
    checkThread();
    maskMap.erase(chunk);
}

void TerrainStore::dropAll()
{
    checkThread();
    tileMap.clear();
    perLevel.fill(0);
    pinnedCount = 0;
    maskMap.clear();
    pyramidData.reset();
}

void TerrainStore::dropStreamed()
{
    checkThread();
    for (auto it = tileMap.begin(); it != tileMap.end();)
    {
        it->second.visual = false;
        it                = it->second.pinned ? std::next(it) : tileMap.erase(it);
    }
    perLevel.fill(0);
    maskMap.clear();
}

void TerrainStore::dropMasks()
{
    checkThread();
    maskMap.clear();
}

size_t TerrainStore::cpuBytes() const
{
    size_t b = 0;
    for (const auto& t : tileMap) b += t.second.data.heights ? t.second.data.heights->size() * 4 : 0;
    if (pyramidData && pyramidData->heights) b += pyramidData->heights->size() * 4;
    for (const auto& m : maskMap) b += m.second.density ? m.second.density->size() : 0;
    return b;
}

}  // namespace Engine
}  // namespace Saiga
