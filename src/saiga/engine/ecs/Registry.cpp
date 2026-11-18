/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Registry: entity slots, the built-in Transform storage and registration checks.

#include "saiga/engine/ecs/Registry.h"

#include "saiga/engine/core/Log.h"

namespace Saiga
{
namespace Engine
{
Registry::Registry()
{
    // Slot 0 is never used: {0, 0} is never alive
    slots.emplace_back();
    transforms.emplace_back();

    // Transform (ID 0) is built in: stored by entity index, present on every entity
    ComponentInfo& t = infos[Transform::id];
    t.name           = Transform::name;
    t.version        = Transform::version;
    t.replicate      = Replicate::OwnerAuthoritative;
    t.saved          = Saved::Yes;
    t.registered     = true;
    registeredComponents++;
}

Registry::~Registry() = default;

void Registry::countRegistration()
{
    registeredComponents++;
    if (registeredComponents == 200)
        SAIGA_LOG("ecs", "200 component types registered (limit 256): prefer fields on existing components or tags");
}

void Registry::validateRegistration() const
{
    // Each range must be a prefix: no registered ID after an unregistered one
    auto checkRange = [&](int first, int last, const char* range)
    {
        bool gap  = false;
        int gapId = -1;
        for (int id = first; id <= last; ++id)
        {
            if (!infos[id].registered)
            {
                if (!gap) gapId = id;
                gap = true;
            }
            else
            {
                SAIGA_ASSERT(!gap, std::string("validateRegistration: ") + range + " component ID " +
                                       std::to_string(gapId) + " is not registered, but " + infos[id].name + " (ID " +
                                       std::to_string(id) + ") is");
            }
        }
    };
    checkRange(0, kFirstGameComponent - 1, "engine");
    checkRange(kFirstGameComponent, int(kMaxComponents) - 1, "game");
}

Entity Registry::create()
{
    checkNotIterating("create");
    uint32_t index;
    if (!freeSlots.empty())
    {
        index = freeSlots.front();
        freeSlots.pop_front();
    }
    else
    {
        SAIGA_ASSERT(slots.size() < 0xFFFFFFFFu, "create: out of entity slots");
        index = uint32_t(slots.size());
        slots.emplace_back();
        transforms.emplace_back();
    }
    Slot& s = slots[index];
    s.state = SlotState::Alive;
    s.flags = 0;
    s.mask.clear();
    s.mask.set(Transform::id);
    transforms[index] = Transform();
    aliveEntities++;
    return Entity{index, s.generation};
}

void Registry::destroy(Entity e)
{
    checkNotIterating("destroy");
    checkAlive(e, "destroy");
    Slot& s = slots[e.index];
    // Components in ID order (deterministic; the order matters for the structural log of S08)
    for (size_t id = 1; id < kMaxComponents; ++id)
        if (s.mask.test(ComponentId(id))) pools[id]->remove(e.index);
    s.mask.clear();
    s.state = SlotState::Free;
    s.flags = 0;
    s.generation++;
    // A slot whose generation wrapped around is retired (a stale handle could match again)
    if (s.generation != 0) freeSlots.push_back(e.index);
    aliveEntities--;
}

uint8_t Registry::flags(Entity e) const
{
    checkAlive(e, "flags");
    return slots[e.index].flags;
}

void Registry::setFlags(Entity e, uint8_t f)
{
    checkAlive(e, "setFlags");
    slots[e.index].flags = f;
}

const ComponentMask& Registry::entityMask(Entity e) const
{
    checkAlive(e, "entityMask");
    return slots[e.index].mask;
}

void Registry::checkAlive(Entity e, const char* operation) const
{
    SAIGA_ASSERT(alive(e), std::string(operation) + ": " + describe(e) + " is not alive");
}

void Registry::checkNotIterating(const char* operation) const
{
    SAIGA_ASSERT(iterationDepth == 0, std::string(operation) +
                                          " while a view iterates (structural changes move "
                                          "components; use commands, S08)");
}

std::string Registry::describe(Entity e) const
{
    std::string s = "entity " + std::to_string(e.index) + ":" + std::to_string(e.generation);
    if (e.index == 0)
        s += " (invalid handle)";
    else if (e.index >= slots.size())
        s += " (index out of range)";
    else if (slots[e.index].generation != e.generation)
        s += " (stale, slot generation " + std::to_string(slots[e.index].generation) + ")";
    else if (slots[e.index].state == SlotState::Reserved)
        s += " (reserved)";
    return s;
}

}  // namespace Engine
}  // namespace Saiga
