/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/core/math/math.h"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/export.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <unordered_map>

namespace Saiga
{
namespace Engine
{
/**
 * Uniform grid per space (docs/engine_design.md §9.5): square cells of kCellSize meters in x and z (columns: y is not
 * divided). Each entity is stored with the position it had at its last update in the cell of that position. Derived
 * data: the World updates it in phase 6 (spatialGridSystem) and never saves or hashes it.
 *
 * Plain store: it does not know whether an entity is alive (the World filters stale handles). Not thread-safe for
 * writes; concurrent queries are fine.
 */
class SAIGA_ENGINE_API SpatialGrid
{
   public:
    static constexpr double kCellSize = 32;

    SpatialGrid() = default;
    // Not copyable (MSVC instantiates the copy operations of an exported class: unique_ptr members)
    SpatialGrid(const SpatialGrid&)            = delete;
    SpatialGrid& operator=(const SpatialGrid&) = delete;

    struct Item
    {
        Entity entity;
        Vec3 position = Vec3::Zero();
    };

    // Puts the entity (by index; a newer handle of the same index replaces an older one) into the cell of (space,
    // position), or moves it there. A non-finite coordinate maps to cell 0 in that axis (such an entity is never
    // found, see forEachWithin).
    void update(Entity e, SpaceId space, const Vec3& position);
    // Removes the entity with this index (no-op if it is not in the grid)
    void remove(uint32_t index);
    void clear();

    // The handle stored for this index, if any
    bool contains(uint32_t index) const { return index < entries.size() && entries[index].cell; }
    Entity entityAt(uint32_t index) const
    {
        return contains(index) ? (*entries[index].cell)[entries[index].slot].entity : Entity{};
    }
    size_t size() const { return count; }

    // Calls fn(const Item&) for every item of the space within radius of center (Euclidean distance in 3D, the boundary
    // included), in an unspecified order. Visits the cells overlapping the square of the radius, or every occupied cell
    // of the space if those are fewer (large radii). A negative or NaN radius finds nothing; items with a non-finite
    // position are never found (nor is anything from a non-finite center). fn must not change the grid, directly or
    // through the World (no step, update, remove or clear while the query runs: the cell being visited may move);
    // nested read-only queries are fine. The Item reference is valid only during the call.
    template <class F>
    void forEachWithin(SpaceId space, const Vec3& center, double radius, F&& fn) const;

   private:
    struct Entry
    {
        // The cell's item vector (an element of an unordered_map: stable until the cell is erased, which happens only
        // when it is empty); null: not in the grid
        std::vector<Item>* cell = nullptr;
        SpaceId space           = 0;
        int32_t cx = 0, cz = 0;
        uint32_t slot = 0;  // position in the cell's item vector
    };
    using Cells = std::unordered_map<uint64_t, std::vector<Item>>;

    static int32_t cellCoordinate(double v);
    static uint64_t key(int32_t cx, int32_t cz) { return (uint64_t(uint32_t(cx)) << 32) | uint64_t(uint32_t(cz)); }
    void removeFromCell(const Entry& en);

    // By SpaceId (grown on demand); the maps are heap-allocated, so growing the vector never moves their elements
    std::vector<std::unique_ptr<Cells>> spaces;
    std::vector<Entry> entries;  // by entity index
    size_t count = 0;
};

template <class F>
void SpatialGrid::forEachWithin(SpaceId space, const Vec3& center, double radius, F&& fn) const
{
    if (!(radius >= 0) || space >= spaces.size() || !spaces[space]) return;
    // A non-finite center matches nothing (the distance is not finite); returning early avoids a scan of every cell
    if (!std::isfinite(center.x()) || !std::isfinite(center.y()) || !std::isfinite(center.z())) return;
    const Cells& cells = *spaces[space];
    if (cells.empty()) return;
    const double r2 = radius * radius;
    // The squared comparison is exact enough while r2 is a normal double; a squared distance that underflows is then
    // below r2 anyway. Otherwise (r2 overflows, or is 0 or subnormal) the scaled distance is compared with the radius.
    const bool squared  = std::isfinite(r2) && r2 >= std::numeric_limits<double>::min();
    const bool infinite = std::isinf(radius);
    auto within         = [&](const Vec3& p)
    {
        // An infinite radius: every finite position (the center is finite; p - center may overflow)
        if (infinite) return std::isfinite(p.x()) && std::isfinite(p.y()) && std::isfinite(p.z());
        const Vec3 d = p - center;
        // An infinite or NaN distance fails the comparison by itself
        if (squared) return d.squaredNorm() <= r2;
        // Overflowed: the true distance is above DBL_MAX >= radius
        if (!std::isfinite(d.x()) || !std::isfinite(d.y()) || !std::isfinite(d.z())) return false;
        const double m = std::max({std::abs(d.x()), std::abs(d.y()), std::abs(d.z())});
        return m == 0 || m * std::sqrt((d / m).squaredNorm()) <= radius;
    };
    auto visit = [&](const std::vector<Item>& items)
    {
        for (const Item& it : items)
            if (within(it.position)) fn(it);
    };

    // The square of cells around the center; every occupied cell if that is fewer (or the bounds are not finite)
    const double x0 = (center.x() - radius) / kCellSize, x1 = (center.x() + radius) / kCellSize;
    const double z0 = (center.z() - radius) / kCellSize, z1 = (center.z() + radius) / kCellSize;
    const double span = (std::floor(x1) - std::floor(x0) + 1) * (std::floor(z1) - std::floor(z0) + 1);
    if (!(span <= double(cells.size())))
    {
        for (auto& c : cells) visit(c.second);
        return;
    }
    const int32_t cx0 = cellCoordinate(center.x() - radius), cx1 = cellCoordinate(center.x() + radius);
    const int32_t cz0 = cellCoordinate(center.z() - radius), cz1 = cellCoordinate(center.z() + radius);
    for (int64_t cx = cx0; cx <= cx1; ++cx)
        for (int64_t cz = cz0; cz <= cz1; ++cz)
        {
            auto it = cells.find(key(int32_t(cx), int32_t(cz)));
            if (it != cells.end()) visit(it->second);
        }
}

}  // namespace Engine
}  // namespace Saiga
