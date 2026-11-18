/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// tickSystem (phase 6, after the grid; §8.4). Reads: Player and Transform of the players, Transform and Behaviour of
// the objects, the changed list. Writes: Behaviour::tickInterval (and sinceTick of spawned objects), the world's player
// cells.

#include "saiga/engine/components/Player.h"
#include "saiga/engine/core/Hash.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/gameplay/Systems.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>

namespace Saiga
{
namespace Engine
{
namespace
{
using Cell = std::array<int64_t, 3>;

// Cell coordinate; clamped far out (no overflow in the ring distance), non-finite: never near a player
bool cellCoordinate(double v, double size, int64_t& out)
{
    if (!std::isfinite(v)) return false;
    const double c = std::floor(v / size);
    out            = int64_t(std::max(-1e15, std::min(1e15, c)));
    return true;
}

bool cellOf(const Transform& t, double size, Cell& out)
{
    out[0] = t.space();
    return cellCoordinate(t.position().x(), size, out[1]) && cellCoordinate(t.position().z(), size, out[2]);
}

// The sorted, unique cells of all players
std::vector<Cell> playerCells(World& w)
{
    std::vector<Cell> cells;
    const double size = w.worldConfig().ticking.ringSize;
    w.viewAll<const Player, const Transform>().each(
        [&](Entity, const Player&, const Transform& t)
        {
            Cell c;
            if (cellOf(t, size, c)) cells.push_back(c);
        });
    std::sort(cells.begin(), cells.end());
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    return cells;
}

// The Chebyshev distance in cells to the nearest player cell of the same space (INT64_MAX: none)
int64_t ringOf(const std::vector<Cell>& players, const Cell& c)
{
    int64_t ring = INT64_MAX;
    for (const Cell& p : players)
        if (p[0] == c[0]) ring = std::min(ring, int64_t(std::max(std::llabs(p[1] - c[1]), std::llabs(p[2] - c[2]))));
    return ring;
}

void assign(World& w, const std::vector<Cell>& players, Entity e, Behaviour& b, bool spawned)
{
    const TickPolicy* policy = b.object ? b.object->tickPolicy() : nullptr;
    if (!policy) policy = &w.worldConfig().ticking;
    double interval = 0;
    if (!b.alwaysTickEveryStep && !w.has<Player>(e))
    {
        Cell c;
        const int64_t ring =
            cellOf(w.read<Transform>(e), w.worldConfig().ticking.ringSize, c) ? ringOf(players, c) : INT64_MAX;
        interval = ring <= policy->nearRing ? 0 : ring <= policy->midRing ? policy->midInterval : policy->farInterval;
    }
    b.tickInterval = interval;
    // Spread the first ticks of objects spawned together over the interval (deterministic, by index)
    if (spawned) b.sinceTick = double(hashCombine(0, e.index) % 1024) / 1024.0 * interval;
}
}  // namespace

void assignTickInterval(World& w, Entity e, Behaviour& b, bool spawned)
{
    assign(w, playerCells(w), e, b, spawned);
}

void tickSystem(World& w, double)
{
    std::vector<Cell> players = playerCells(w);
    if (players != w.tickPlayerCells)
    {
        // A player crossed a cell border, joined or left: every object
        w.tickPlayerCells = std::move(players);
        w.viewAll<Behaviour>().each([&](Entity e, Behaviour& b) { assign(w, w.tickPlayerCells, e, b, false); });
        return;
    }
    // Otherwise the objects that moved (changed in this or the previous step)
    Registry& reg = w.registry();
    for (uint32_t index : w.changedTransforms())
    {
        const Entity e = reg.aliveAt(index);
        if (!e.valid()) continue;
        if (Behaviour* b = reg.tryGet<Behaviour>(e)) assign(w, w.tickPlayerCells, e, *b, false);
    }
}

}  // namespace Engine
}  // namespace Saiga
