/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// SpatialGrid: the uniform grid per space (§9.5).

#include "saiga/engine/world/SpatialGrid.h"

#include "saiga/core/util/assert.h"

#include <algorithm>

namespace Saiga
{
namespace Engine
{
int32_t SpatialGrid::cellCoordinate(double v)
{
    // Clamped: far coordinates share the outermost cell (still correct, the distance check decides)
    constexpr double kLimit = double(1 << 30);
    if (!std::isfinite(v)) return 0;
    const double c = std::floor(v / kCellSize);
    return int32_t(std::max(-kLimit, std::min(kLimit, c)));
}

void SpatialGrid::removeFromCell(const Entry& en)
{
    std::vector<Item>& items = *en.cell;
    // Swap with the last item (the order inside a cell does not matter: results are sorted)
    if (en.slot + 1 != items.size())
    {
        items[en.slot]                            = items.back();
        entries[items[en.slot].entity.index].slot = en.slot;
    }
    items.pop_back();
    if (items.empty()) spaces[en.space]->erase(key(en.cx, en.cz));
}

void SpatialGrid::update(Entity e, SpaceId space, const Vec3& position)
{
    SAIGA_ASSERT(e.index != 0, "SpatialGrid::update: invalid entity");
    const int32_t cx = cellCoordinate(position.x()), cz = cellCoordinate(position.z());
    if (e.index < entries.size())
    {
        Entry& en = entries[e.index];
        if (en.cell && en.space == space && en.cx == cx && en.cz == cz)
        {
            // Same cell (the common case): only the stored handle and position change, no lookup
            Item& it    = (*en.cell)[en.slot];
            it.entity   = e;
            it.position = position;
            return;
        }
    }
    else
        entries.resize(size_t(e.index) + 1);
    if (space >= spaces.size()) spaces.resize(size_t(space) + 1);
    if (!spaces[space]) spaces[space] = std::make_unique<Cells>();

    // Insert first (the only step that allocates), then remove from the old cell
    std::vector<Item>& items = (*spaces[space])[key(cx, cz)];
    items.push_back(Item{e, position});
    Entry& en = entries[e.index];
    if (en.cell)
        removeFromCell(en);
    else
        count++;
    en.cell  = &items;
    en.space = space;
    en.cx    = cx;
    en.cz    = cz;
    en.slot  = uint32_t(items.size() - 1);
}

void SpatialGrid::remove(uint32_t index)
{
    if (!contains(index)) return;
    removeFromCell(entries[index]);
    entries[index].cell = nullptr;
    count--;
}

void SpatialGrid::clear()
{
    spaces.clear();
    entries.clear();
    count = 0;
}

}  // namespace Engine
}  // namespace Saiga
