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
    slots.push_back();
    transforms.push_back();

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
        (void)gapId;  // read only in the assert message (unused without SAIGA_ASSERTS)
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
    Entity e = reserveSlot();
    activate(e);
    return e;
}

Entity Registry::reserveSlot()
{
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
        // Both arrays grow together: if the second allocation throws, the first is undone. Pages never move, so this
        // is safe while a view holds references.
        transforms.push_back();
        try
        {
            slots.push_back();
        }
        catch (...)
        {
            transforms.pop_back();
            throw;
        }
    }
    Slot& s = slots[index];
    s.state = SlotState::Reserved;
    s.flags = 0;
    s.mask.clear();
    // A fresh slot: no view holds a reference to this Transform
    transforms[index] = Transform();
    return Entity{index, s.generation};
}

Entity Registry::reserve()
{
    SAIGA_ASSERT(!destroying, "reserve while destroy removes the components of an entity");
    SAIGA_ASSERT(!inParallelEach(), "reserve inside parallelEach");
    return reserveSlot();
}

void Registry::activate(Entity e)
{
    checkNotIterating("activate");
    SAIGA_ASSERT(reserved(e), "activate: " + describe(e) + " is not reserved");
    Slot& s = slots[e.index];
    s.state = SlotState::Alive;
    s.flags = kSimulated;
    s.mask.set(Transform::id);
    aliveEntities++;
}

void Registry::release(Entity e)
{
    SAIGA_ASSERT(reserved(e), "release: " + describe(e) + " is not reserved");
    Slot& s                       = slots[e.index];
    const uint32_t nextGeneration = s.generation + 1;
    // The only step that can throw comes first
    if (nextGeneration != 0) freeSlots.push_back(e.index);
    s.state      = SlotState::Free;
    s.generation = nextGeneration;
}

void Registry::shiftTransformFlags()
{
    for (uint32_t i = 1; i < uint32_t(slots.size()); ++i)
    {
        Transform& t      = transforms[i];
        t.changedLastStep = t.changedThisStep;
        t.changedThisStep = 0;
        t.teleported      = 0;
    }
}

void Registry::hashInto(Hasher& h) const
{
    std::vector<uint8_t> bytes;
    h.addU64(uint64_t(slots.size()));
    for (uint32_t i = 1; i < uint32_t(slots.size()); ++i)
    {
        const Slot& s = slots[i];
        if (s.state != SlotState::Alive) continue;
        bytes.clear();
        BinaryWriter w(bytes);
        w.write(transforms[i]);
        // The saved components in ID order, each with its ID
        for (size_t id = 1; id < kMaxComponents; ++id)
        {
            if (!s.mask.test(ComponentId(id)) || !savedComponent(ComponentId(id))) continue;
            bytes.push_back(uint8_t(id));
            pools[id]->appendBytes(i, bytes);
        }
        h.addU32(i).addU32(s.generation).addU64(uint64_t(bytes.size())).add(bytes.data(), bytes.size());
    }
    // The free list decides the indices and generations of future entities
    h.addU64(uint64_t(freeSlots.size()));
    for (uint32_t index : freeSlots) h.addU32(index).addU32(slots[index].generation);
}

void Registry::destroy(Entity e)
{
    checkNotIterating("destroy");
    checkAlive(e, "destroy");
    Slot& s = slots[e.index];
    // The only step that can throw (the free list allocates) comes first: on failure nothing changed. A slot whose
    // generation wraps around is retired (a stale handle could match again).
    const uint32_t nextGeneration = s.generation + 1;
    if (nextGeneration != 0) freeSlots.push_back(e.index);
    // Components in ID order (deterministic; the order matters for the structural log of S08). Cannot throw:
    // components have noexcept moves and the sparse pages exist.
    destroying = true;
    for (size_t id = 1; id < kMaxComponents; ++id)
    {
        if (!s.mask.test(ComponentId(id))) continue;
        // The bit goes with the component: a destructor of a later component sees has() == false for this one
        s.mask.reset(ComponentId(id));
        pools[id]->remove(e.index);
    }
    destroying = false;
    s.mask.clear();
    s.state      = SlotState::Free;
    s.flags      = 0;
    s.generation = nextGeneration;
    aliveEntities--;
}

void Registry::debugSetGeneration(uint32_t index, uint32_t generation)
{
    // Only free slots: an alive entity's handles are also stored in the pools
    SAIGA_ASSERT(index != 0 && index < slots.size() && slots[index].state == SlotState::Free,
                 "debugSetGeneration: only on free slots");
    slots[index].generation = generation;
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
    // A component destructor that changes the registry while destroy removes the components would get the slot that
    // is being destroyed (it is already in the free list)
    SAIGA_ASSERT(!destroying, std::string(operation) + " while destroy removes the components of an entity");
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
