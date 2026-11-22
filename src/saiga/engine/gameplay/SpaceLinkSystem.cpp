/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// spaceLinkSystem (phase 4, host, optional; S21). Reads: InteractEvent of the previous step, SpaceLink, Parent,
// Transform. Writes: World::requestTransition (a command during the step).

#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/SpaceLink.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/ecs/EntityLog.h"
#include "saiga/engine/gameplay/Systems.h"
#include "saiga/engine/world/Events.h"
#include "saiga/engine/world/World.h"

namespace Saiga
{
namespace Engine
{
void spaceLinkSystem(World& w, double)
{
    for (const InteractEvent& ev : w.events.read<InteractEvent>())
    {
        if (!w.alive(ev.target) || !w.has<SpaceLink>(ev.target)) continue;
        if (!w.alive(ev.actor) || w.has<Parent>(ev.actor)) continue;
        const SpaceId actorSpace = w.read<Transform>(ev.actor).space();
        const SpaceId linkSpace  = w.read<Transform>(ev.target).space();
        if (actorSpace != linkSpace)
        {
            // A game bug: the interaction was decided on a stale pose
            SAIGA_LOG_E("world", ev.actor,
                        "interacts with the space link e%u:%u in another space (%d, the link in %d): ignored",
                        ev.target.index, ev.target.generation, int(actorSpace), int(linkSpace));
            continue;
        }
        w.requestTransition(ev.actor, ev.target);
    }
}

}  // namespace Engine
}  // namespace Saiga
