/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The host's terrain edit (docs/engine_design.md §12.2; S22): applyTerrainEdit, the engine-internal application of a
// brush behind World::editTerrain (the command at a phase flush, at once between steps, the world API), and its small
// value types: the change of one applied modification (TerrainChange), the part of a TerrainModification message
// (TerrainPart) and the edit's result (TerrainEditResult). Runs serially on the main thread (R-PAR-1).

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/engine/terrain/BrushKernel.h"
#include "saiga/engine/terrain/TerrainBrush.h"
#include "saiga/export.h"

#include <cstdint>
#include <vector>

namespace Saiga
{
namespace Engine
{
class World;

// An applied terrain change: the space and the global box of the changed samples (inclusive, base-grid samples)
struct TerrainChange
{
    SpaceId space = 0;
    SampleRect box;
};

// One chunk of a TerrainModification (§13.2): the chunk's sequence after the change and the new deltas (new height
// minus base, absolute) of the chunk-local box x0, z0, w x h (row-major, row = z)
struct TerrainPart
{
    ChunkId chunk;
    uint32_t sequence = 0;
    int x0 = 0, z0 = 0, w = 0, h = 0;
    std::vector<int32_t> deltas;
};

// The result of applyTerrainEdit (the world API reports it as JSON; S23b appends its tombstones)
struct TerrainEditResult
{
    // The chunk-local box of a part (samples)
    struct Chunk
    {
        ChunkId chunk;
        uint32_t sequence = 0;
        int x0 = 0, z0 = 0, w = 0, h = 0;
    };
    SpaceId space = 0;
    // ChunkId order; empty when nothing changed
    std::vector<Chunk> chunks;
    uint64_t changedSamples = 0;
    // The global changed box (inclusive; empty when nothing changed)
    int64_t i0 = 0, k0 = 0, i1 = -1, k1 = -1;
};

// The base and the pre-edit heights of a rectangle of space samples (row-major): the base per level-0 tile as row
// copies of a resident tile, else TerrainSystem::sample (the height source, counted in baseHookSamples); height =
// clampTerrainHeight(base + delta). Allocates nothing in the store
SAIGA_ENGINE_API void gatherSamples(const World& world, SpaceId space, const SampleRect& rect,
                                    std::vector<int32_t>& base, std::vector<int32_t>& height);

// Applies a validated brush on the host (World::editTerrain checked the role, the space and the brush): quantize,
// gather, kernel, reserve (blocks, entries, the host send-list entry), commit (deltas, sequences, the send-list entry),
// one World::onTerrainChanged, statistics. Every allocation happens before the first write: an exception leaves the
// state unchanged. An edit that changes nothing changes no state (no sequence, part or event)
SAIGA_ENGINE_API TerrainEditResult applyTerrainEdit(World& world, SpaceId space, const TerrainBrush& brush);

}  // namespace Engine
}  // namespace Saiga
