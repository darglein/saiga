/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// TerrainStore (docs/engine_design.md §12.2, §12.3; S20): the base terrain data of one space, owned by TerrainSystem:
// the resident tiles of every streamed level with the frame in which each was last desired (LRU), the pyramid
// (generated on the space's first view, CPU-cached while the world lives, D5) and the grass masks. Never mutated by
// edits (S22 merges at the send point). Mutated only on the main thread and never inside parallelEach (S24: phase 7
// pins tiles inside a step; asserted with SAIGA_ASSERTS: the thread); TerrainSystem::sample reads its level-0 tiles
// (also from parallelEach, while nothing mutates it).
//
// S24: a tile is held by the visual path (visual: stored by a visual completion, or desired by the view) and / or
// pinned by the residency (the level-0 base heights of a resident chunk, any space). The visual budget counts visual
// tiles only (tilesAtLevel); a tile that loses its visual membership stays while pinned, and an unpinned one stays
// while visual.

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/export.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"

#include <array>
#include <map>
#include <memory>
#include <thread>

namespace Saiga
{
namespace Engine
{
class SAIGA_ENGINE_API TerrainStore
{
   public:
    struct Tile
    {
        Vulkan::TerrainTileData data;
        uint64_t lastDesired = 0;
        bool visual          = false;  // held by the visual path (S24)
        bool pinned          = false;  // held by the residency (S24)
    };

    TerrainStore();

    // Tiles (all streamed levels)
    const Tile* tile(const RegionId& region) const;
    // A visual completion: stored as visual (a pinned tile of the region keeps its data: the samples are equal)
    void addTile(const RegionId& region, Vulkan::TerrainTileData data, uint64_t frame);
    // The visual path drops the tile: removed unless pinned
    void removeTile(const RegionId& region);
    // The view desires a stored tile: visual from now on
    void markDesired(const RegionId& region, uint64_t frame);
    // Visual tiles of a level (the visual budget)
    int tilesAtLevel(int level) const;
    const std::map<RegionId, Tile>& tiles() const { return tileMap; }
    // S24: the residency's level-0 tiles (pinTile keeps the data of an existing visual tile; unpinTile removes the tile
    // unless visual)
    void pinTile(const RegionId& region, Vulkan::TerrainTileData data);
    void unpinTile(const RegionId& region);
    int pinnedTiles() const { return pinnedCount; }
    size_t pinnedBytes() const;

    // The pyramid (null before the space's first view); its pointer stays the same until dropAll (a host load with
    // other generator pins through TerrainSystem::replaceSources): read it at each use
    const Vulkan::TerrainPyramidData* pyramid() const { return pyramidData.get(); }
    void setPyramid(Vulkan::TerrainPyramidData pyramid);

    // Grass masks by chunk
    const Vulkan::GrassMaskData* mask(const ChunkId& chunk) const;
    void addMask(const ChunkId& chunk, Vulkan::GrassMaskData mask);
    void removeMask(const ChunkId& chunk);
    const std::map<ChunkId, Vulkan::GrassMaskData>& masks() const { return maskMap; }

    // Drops the visual streamed data (the visual membership of every tile, the unpinned tiles and the masks), keeps the
    // pyramid and the pinned tiles (S24)
    void dropStreamed();
    // Drops everything incl. the pyramid (a host load with other generator pins, S23a: a new pyramid gets a new
    // pointer)
    void dropAll();
    void dropMasks();
    // The bytes of the tiles, the pyramid and the masks
    size_t cpuBytes() const;

   private:
    std::map<RegionId, Tile> tileMap;
    std::array<int, 8> perLevel{};  // visual tiles
    int pinnedCount = 0;
    std::unique_ptr<Vulkan::TerrainPyramidData> pyramidData;
    std::map<ChunkId, Vulkan::GrassMaskData> maskMap;
    std::thread::id owner;
    void checkThread() const;
};

}  // namespace Engine
}  // namespace Saiga
