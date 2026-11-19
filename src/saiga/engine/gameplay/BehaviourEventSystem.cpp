/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// behaviourEventSystem (phase 2, host; §9.2). Reads: DamageEvent, InteractEvent, ContactEvent (of the previous step),
// Behaviour; calls the handlers (onDamage, onInteract, onContact of both entities), per event type in emit order.

#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/gameplay/Systems.h"

namespace Saiga
{
namespace Engine
{
namespace
{
template <class F>
void deliver(World& w, Entity target, F&& fn)
{
    Registry& reg = w.registry();
    // Dead targets, entities without a game object and entities outside the simulation are skipped
    if (!reg.alive(target) || !(reg.flags(target) & Registry::kSimulated)) return;
    Behaviour* b = reg.tryGet<Behaviour>(target);
    if (b && b->object) fn(*b->object);
}
}  // namespace

void behaviourEventSystem(World& w, double)
{
    for (const DamageEvent& ev : w.events.read<DamageEvent>())
        deliver(w, ev.target, [&](GameObject& o) { o.onDamage(ev); });
    for (const InteractEvent& ev : w.events.read<InteractEvent>())
        deliver(w, ev.target, [&](GameObject& o) { o.onInteract(ev); });
    for (const ContactEvent& ev : w.events.read<ContactEvent>())
    {
        deliver(w, ev.a, [&](GameObject& o) { o.onContact(ev); });
        deliver(w, ev.b, [&](GameObject& o) { o.onContact(ev); });
    }
}

}  // namespace Engine
}  // namespace Saiga
