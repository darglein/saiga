/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// behaviourSystem (phase 3, host, serial; §8.4). Reads and writes: Behaviour (sinceTick); calls GameObject::update,
// whose structural changes are commands.

#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/gameplay/Systems.h"

namespace Saiga
{
namespace Engine
{
void behaviourSystem(World& w, double dt)
{
    double updated = 0;
    // Dense order of the Behaviour pool; only Simulated entities (phase 3)
    w.view<Behaviour>().each(
        [&](Entity, Behaviour& b)
        {
            if (!b.object) return;
            b.sinceTick += dt;
            // A small tolerance: 6 steps of 1/60 s sum to slightly less than 0.1 s
            if (!b.alwaysTickEveryStep && b.sinceTick + 1e-9 < b.tickInterval) return;
            const double since = b.sinceTick;
            b.sinceTick        = 0;
            updated++;
            b.object->update(since);
        });
    Profiler& p = w.profiler();
    p.addCounter(p.counter("objects_updated"), updated);
}

}  // namespace Engine
}  // namespace Saiga
