/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// TerrainDirtyQueue (docs/engine_design.md §12.2; S22): the changed boxes (global base samples) that TerrainSystem's
// patch step still has to send to the render sink: the tile boxes of the view space (dropped on a view-space switch:
// the new set is sent merged) and the pyramid boxes per space (kept across switches). A new box is merged into an
// overlapping or touching one (repeatedly), so the lists stay short; the order is the insertion order. A list's front
// box that the patch step split by rows (active, its first rows sent) takes no new boxes: it drains first (else edits
// next to its sent rows grew it back each frame and its other rows starved).

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/engine/terrain/SampleRect.h"
#include "saiga/export.h"

#include <map>
#include <vector>

namespace Saiga
{
namespace Engine
{
class SAIGA_ENGINE_API TerrainDirtyQueue
{
   public:
    // The boxes of one list; frontActive: the front box was split (the patch step sets it, removing the front or
    // clearing the list resets it)
    struct List
    {
        std::vector<SampleRect> boxes;
        bool frontActive = false;
    };

    void addTileBox(const SampleRect& box) { merge(tiles, box); }
    void addPyramidBox(SpaceId space, const SampleRect& box) { merge(pyramids[space], box); }
    List& tileList() { return tiles; }
    List& pyramidList(SpaceId space) { return pyramids[space]; }
    const std::vector<SampleRect>& tileBoxes() const { return tiles.boxes; }
    // The pyramid boxes of a space (empty if none)
    size_t pyramidBoxCount(SpaceId space) const
    {
        auto it = pyramids.find(space);
        return it == pyramids.end() ? 0 : it->second.boxes.size();
    }
    void clearTileBoxes() { tiles = List(); }
    void clear()
    {
        tiles = List();
        pyramids.clear();
    }
    // The queued boxes (tiles and every space's pyramid)
    size_t size() const;
    // Nothing queued for this view space
    bool empty(SpaceId viewSpace) const;

   private:
    List tiles;
    std::map<SpaceId, List> pyramids;
    static void merge(List& list, SampleRect box);
};

}  // namespace Engine
}  // namespace Saiga
