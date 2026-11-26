/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// spaceTransitionSystem (phase 1, host; S21). Reads: PendingTransition, Parent, the entity flags. Writes: Transform
// (World::moveToSpace), SpaceTransitionEvent, removes PendingTransition.

#include "saiga/engine/world/SpaceTransitionSystem.h"

#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/PendingTransition.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/ecs/EntityLog.h"
#include "saiga/engine/streaming/ChunkResidency.h"
#include "saiga/engine/world/Events.h"
#include "saiga/engine/world/World.h"

#include <algorithm>
#include <vector>

namespace Saiga
{
namespace Engine
{
void spaceTransitionSystem(World& w, double)
{
    // Collected first: moveToSpace and the removals must not run inside the view
    std::vector<std::pair<Entity, PendingTransition>> pending;
    // A request of this step (a message handler's in phase 1; or of an aborted run of it) completes in the next one
    const std::vector<Entity>& thisStep = w.transitionsRequestedInStep;
    w.viewAll<const PendingTransition>().each(
        [&](Entity e, const PendingTransition& p)
        {
            if (std::find(thisStep.begin(), thisStep.end(), e) == thisStep.end()) pending.push_back({e, p});
        });
    const Registry& reg = w.registry();
    for (const auto& [e, p] : pending)
    {
        const char* invalid = nullptr;
        if (reg.has<Parent>(e))
            invalid = "the entity is a child now";
        else if (reg.flags(e) & (Registry::kWorldState | Registry::kSpaceState))
            invalid = "the entity is a state entity";
        else if (p.target >= w.spaceCount())
            invalid = "the target is not a space";
        if (invalid)
        {
            SAIGA_LOG_E("world", e, "space transition to space %d dropped: %s", int(p.target), invalid);
            w.remove<PendingTransition>(e);
            continue;
        }
        // S24 (G12): in a streaming world a player root arrives when its island (the chunks intersecting the
        // simulationRadius disc of the arrival position) is requested and resident; it is held until then (re-checked
        // every step). Other roots and worlds without streaming arrive now (S21's rule)
        if (w.streaming() && w.isPlayerRoot(e) && w.residency().islandProgress(p.target, p.position) < 1) continue;
        const SpaceId from = reg.read<Transform>(e).space();
        w.moveToSpace(e, p.target, p.position, p.rotation);
        w.events.emit(SpaceTransitionEvent{e, p.link, from, p.target});
        w.remove<PendingTransition>(e);
    }
}

}  // namespace Engine
}  // namespace Saiga
