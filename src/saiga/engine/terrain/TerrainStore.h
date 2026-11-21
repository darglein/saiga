/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// TerrainStore (docs/engine_design.md §12.2, §12.3; S20): the base terrain data of one space, owned by TerrainSystem:
// the resident tiles of every streamed level with the frame in which each was last desired (LRU), the pyramid
// (generated on the space's first view, CPU-cached while the world lives, D5) and the grass masks. Never mutated by
// edits (S22 merges at the send point). Mutated only on the main thread between steps (asserted with SAIGA_ASSERTS);
// TerrainSystem::sample reads its level-0 tiles (also from parallelEach, while nothing mutates it).

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
    };

    TerrainStore();

    // Tiles (all streamed levels)
    const Tile* tile(const RegionId& region) const;
    void addTile(const RegionId& region, Vulkan::TerrainTileData data, uint64_t frame);
    void removeTile(const RegionId& region);
    void markDesired(const RegionId& region, uint64_t frame);
    int tilesAtLevel(int level) const;
    const std::map<RegionId, Tile>& tiles() const { return tileMap; }

    // The pyramid (null before the space's first view); its pointer stays the same while the world lives
    const Vulkan::TerrainPyramidData* pyramid() const { return pyramidData.get(); }
    void setPyramid(Vulkan::TerrainPyramidData pyramid);

    // Grass masks by chunk
    const Vulkan::GrassMaskData* mask(const ChunkId& chunk) const;
    void addMask(const ChunkId& chunk, Vulkan::GrassMaskData mask);
    void removeMask(const ChunkId& chunk);
    const std::map<ChunkId, Vulkan::GrassMaskData>& masks() const { return maskMap; }

    // Drops the streamed data (tiles and masks), keeps the pyramid
    void dropStreamed();
    // Drops everything incl. the pyramid (a host load with other generator pins, S23a: a new pyramid gets a new
    // pointer)
    void dropAll();
    void dropMasks();
    // The bytes of the tiles, the pyramid and the masks
    size_t cpuBytes() const;

   private:
    std::map<RegionId, Tile> tileMap;
    std::array<int, 8> perLevel{};
    std::unique_ptr<Vulkan::TerrainPyramidData> pyramidData;
    std::map<ChunkId, Vulkan::GrassMaskData> maskMap;
    std::thread::id owner;
    void checkThread() const;
};

}  // namespace Engine
}  // namespace Saiga
