/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The World's arrival members (docs/engine_design.md §12.4; S24, G12, G14): requestArrival, checkArrival, the hard
// island limit, arrivalProgress, islandCount and the held state of player roots.

#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/PendingTransition.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/ecs/EntityLog.h"
#include "saiga/engine/streaming/ChunkResidency.h"
#include "saiga/engine/streaming/Islands.h"
#include "saiga/engine/world/World.h"

#include <cmath>

namespace Saiga
{
namespace Engine
{
Entity World::rootOf(Entity e) const
{
    if (!reg.alive(e)) return Entity{};
    const uint32_t limit = reg.slotCount();
    for (uint32_t n = 0; n <= limit; ++n)
    {
        const Parent* p = reg.tryRead<Parent>(e);
        if (!p || !reg.alive(p->entity)) return e;
        e = p->entity;
    }
    return e;
}

bool World::isPlayerRoot(Entity root) const
{
    if (!reg.alive(root)) return false;
    if (reg.has<Player>(root)) return true;
    if (childCount(root) == 0) return false;
    for (Entity x : collectSubtree(root))
        if (reg.has<Player>(x)) return true;
    return false;
}

bool World::heldRoot(Entity root) const
{
    if (!config.streaming || !reg.alive(root) || reg.has<Parent>(root)) return false;
    // A client keeps its player held until the move's OwnerCommand is applied (the component's removal can overtake it)
    if (!isHost() && residencyPtr->clientAwaitsMove(root)) return true;
    return reg.has<PendingTransition>(root) && isPlayerRoot(root);
}

bool World::checkIslandLimit(Entity e, SpaceId space, const Vec3& position, std::string* error) const
{
    if (!config.streaming || !config.hardIslandLimit || !isPlayerRoot(e)) return true;
    // Both counted with every accepted arrival at its target, so two arrivals that each pass alone cannot together
    // exceed the limit (the flush of a deferred request re-checks, addArrival)
    const int now   = Islands::countAccepted(*this);
    const int after = Islands::countWith(*this, e, space, position);
    if (after > config.maxSimulationIslands && after > now)
    {
        if (error)
            *error = "the arrival would raise the islands of players to " + std::to_string(after) +
                     " (hardIslandLimit, maxSimulationIslands " + std::to_string(config.maxSimulationIslands) + ")";
        return false;
    }
    return true;
}

bool World::checkArrival(Entity e, SpaceId space, const Vec3& position, std::string* error) const
{
    auto fail = [&](const std::string& m)
    {
        if (error) *error = m;
        return false;
    };
    if (!isHost()) return fail("requestArrival on a client (a host command)");
    if (!reg.alive(e)) return fail("the entity is not alive");
    if (reg.has<Parent>(e)) return fail("the entity is a child (its parent decides)");
    if (reg.flags(e) & (Registry::kWorldState | Registry::kSpaceState))
        return fail("a world-state or space-state entity");
    if (space >= spaceCount()) return fail("unknown space " + std::to_string(space));
    if (!position.allFinite()) return fail("the position is not finite");
    const double size = spaceTemplate(space).sizeMeters();
    if (position.x() < 0 || position.z() < 0 || position.x() > size || position.z() > size)
        return fail("the position is outside the space's x/z range [0, " + std::to_string(size) + "]");
    if (reg.has<PendingTransition>(e)) return fail("an arrival is pending already (the first request wins)");
    return checkIslandLimit(e, space, position, error);
}

bool World::requestArrival(Entity e, SpaceId space, const Vec3& position, std::optional<Quat> rotation)
{
    checkNotParallel("requestArrival");
    // Code errors (asserts, refused in every build): the role and the entity, as requestTransition
    if (!isHost())
    {
        SAIGA_ASSERT(false, "requestArrival on a client (a host command)");
        return false;
    }
    if (!reg.alive(e))
    {
        SAIGA_ASSERT(false, "requestArrival: the entity is not alive (or reserved)");
        return false;
    }
    if (reg.has<Parent>(e))
    {
        SAIGA_ASSERT(false, "requestArrival: the entity is a child (its parent decides)");
        return false;
    }
    if (reg.flags(e) & (Registry::kWorldState | Registry::kSpaceState))
    {
        SAIGA_ASSERT(false, "requestArrival: a world-state or space-state entity");
        return false;
    }
    std::string error;
    if (!checkArrival(e, space, position, &error))
    {
        SAIGA_LOG_E("world", e, "requestArrival refused: %s", error.c_str());
        return false;
    }
    PendingTransition p;
    p.target   = space;
    p.position = position;
    p.rotation = rotation ? *rotation : reg.read<Transform>(e).rotation();
    addArrival(e, p, "requestArrival");
    return true;
}

std::optional<double> World::arrivalProgress(Entity e) const
{
    if (!reg.alive(e) || !reg.has<PendingTransition>(e)) return std::nullopt;
    if (!config.streaming || !isPlayerRoot(e)) return 1.0;
    const PendingTransition& p = reg.read<PendingTransition>(e);
    return residencyPtr->islandProgress(p.target, p.position);
}

int World::islandCount() const
{
    return Islands::count(*this);
}

}  // namespace Engine
}  // namespace Saiga
