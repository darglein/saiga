/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// World::tryLoad (docs/engine_design.md §15.2, S11): the save is parsed, every value decoded into staged values and
// every rule checked before the world changes (a rejected save leaves it unchanged); then the commit replaces the
// entities, restores the world state, rebuilds the derived state and recreates the game objects.

#include "saiga/engine/components/Player.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/save/GenericDecode.h"
#include "saiga/engine/save/SaveReader.h"
#include "saiga/engine/world/SpatialGridSystem.h"
#include "saiga/engine/world/World.h"

#include <algorithm>

#include <unordered_set>

namespace Saiga
{
namespace Engine
{
class WorldLoader
{
   public:
    explicit WorldLoader(World& w) : w(w), reg(w.reg) {}
    LoadResult run(const std::vector<uint8_t>& data);

   private:
    bool fail(const std::string& message)
    {
        if (error.empty()) error = message;
        return false;
    }
    bool checkSlots();
    bool stageComponents(const std::vector<uint8_t>& data);
    bool stageEvents(const std::vector<uint8_t>& data);
    bool mapClasses();
    bool dropEntities();
    bool checkHierarchy();
    bool checkPersistent();
    bool checkFreedStateEntities();
    // The staged value index -> keep; removes the others from a typed stage
    template <class T>
    static void filterStage(StagedOf<T>& s, const std::vector<bool>& keep);
    bool claim(ComponentId id, uint32_t index, const std::string& what);
    void commit();
    void restoreStateEntities();
    void startObjects(LoadResult& result);

    World& w;
    Registry& reg;
    std::string error;
    ParsedSave p;
    SchemaTable current;
    std::vector<std::unique_ptr<StagedValues>> staged = std::vector<std::unique_ptr<StagedValues>>(kMaxComponents);
    StagedOf<Transform> transforms;
    std::vector<std::vector<bool>> seen = std::vector<std::vector<bool>>(kMaxComponents);
    std::vector<std::unique_ptr<StagedValues>> stagedEvents;
    std::vector<std::string> droppedEvents;
    std::vector<bool> drop;           // by entity index: dropped by a removed type (DropEntity)
    std::vector<int64_t> generation;  // by entity index after the load: the generation, -1 if not alive
};

namespace
{
bool alive(const ParsedSave::Slot& s)
{
    return s.state == uint8_t(SlotState::Alive);
}
}  // namespace

bool WorldLoader::checkSlots()
{
    const auto& slots = p.slots;
    if (slots.empty()) return fail("the slot table is empty");
    const uint8_t knownFlags = Registry::kSimulated | Registry::kWorldState | Registry::kSpaceState;
    int worldStates = 0, spaceStates = 0;
    for (size_t i = 1; i < slots.size(); ++i)
    {
        const auto& s = slots[i];
        if (s.state != uint8_t(SlotState::Free) && s.state != uint8_t(SlotState::Alive))
            return fail("slot " + std::to_string(i) + " has the invalid state " + std::to_string(s.state));
        if (s.flags & ~knownFlags) return fail("slot " + std::to_string(i) + " has unknown flags");
        if (!alive(s) && s.flags) return fail("the free slot " + std::to_string(i) + " has flags");
        if ((s.flags & Registry::kWorldState) && (s.flags & Registry::kSpaceState))
            return fail("slot " + std::to_string(i) + " is world state and space state");
        worldStates += (s.flags & Registry::kWorldState) ? 1 : 0;
        spaceStates += (s.flags & Registry::kSpaceState) ? 1 : 0;
    }
    if (worldStates > 1) return fail("more than one world-state entity");
    // One space before S21
    if (spaceStates > 1) return fail("more than one space-state entity");
    std::vector<bool> listed(slots.size(), false);
    for (uint32_t index : p.freeList)
    {
        if (index == 0 || index >= slots.size() || alive(slots[index]))
            return fail("the free list names the slot " + std::to_string(index) + ", which is not a free slot");
        if (listed[index]) return fail("the free list names the slot " + std::to_string(index) + " twice");
        listed[index] = true;
    }
    // A free slot that is not in the free list was retired (its generation wrapped around to 0)
    for (size_t i = 1; i < slots.size(); ++i)
        if (!alive(slots[i]) && !listed[i] && slots[i].generation != 0)
            return fail("the free slot " + std::to_string(i) + " is neither in the free list nor retired");
    drop.assign(slots.size(), false);
    return true;
}

bool WorldLoader::claim(ComponentId id, uint32_t index, const std::string& what)
{
    if (index == 0 || index >= p.slots.size() || !alive(p.slots[index]))
        return fail(what + ": the entity index " + std::to_string(index) + " is not an alive slot");
    auto& s = seen[id];
    if (s.empty()) s.assign(p.slots.size(), false);
    if (s[index]) return fail(what + ": entity index " + std::to_string(index) + " has it twice");
    s[index] = true;
    return true;
}

bool WorldLoader::stageComponents(const std::vector<uint8_t>& data)
{
    SaveDecoder decoder(p.schema, &current, &w.migrationList);
    bool transformSeen = false;
    for (const ParsedSave::Block& b : p.components)
    {
        const SchemaType& savedType = p.schema.types[b.schemaIndex];
        const std::string& name     = savedType.name;
        // A type without values needs no counterpart (Transform is required below anyway)
        if (b.count == 0 && name != Transform::name)
        {
            if (b.end != b.begin) return fail("component " + name + ": bytes in an empty block");
            continue;
        }
        BinaryReader r(data.data() + b.begin, b.end - b.begin);
        const std::string what = "component " + name;

        enum class Mode
        {
            Fast,
            Slow,
            Skip,
            DropEntity,
            Replace
        } mode                       = Mode::Slow;
        int target                   = reg.findComponent(name);
        const SchemaType* targetType = nullptr;
        if (name == Transform::name) target = Transform::id;
        if (target >= 0 && reg.registered(ComponentId(target)))
        {
            if (!reg.savedComponent(ComponentId(target)))
                mode = Mode::Skip;  // now Saved::No: dropped silently
            else
            {
                targetType = current.find(SchemaKind::Component, name);
                std::string reason;
                const SchemaMatch m = compareTypes(p.schema, savedType, current, *targetType, reason);
                if (m == SchemaMatch::Incompatible) return fail(reason);
                mode = m == SchemaMatch::Same ? Mode::Fast : Mode::Slow;
            }
        }
        else
        {
            const MigrationList::Removed* removed = w.migrationList.findRemoved(name);
            if (!removed)
                return fail("the save contains the component " + name +
                            ", which is not registered and has no Removed entry (MigrationList::addRemoved)");
            if (removed->policy == RemovedPolicy::DropComponent)
                mode = Mode::Skip;
            else if (removed->policy == RemovedPolicy::DropEntity)
                mode = Mode::DropEntity;
            else
            {
                mode   = Mode::Replace;
                target = reg.findComponent(removed->replacement);
                if (target < 0 || !reg.savedComponent(ComponentId(target)) || target == Transform::id)
                    return fail("the replacement " + removed->replacement + " of the removed component " + name +
                                " is not a registered saved component");
                targetType = current.find(SchemaKind::Component, removed->replacement);
            }
        }
        if (mode == Mode::Skip)
        {
            // Dropped, but checked like every other value (a malformed save must not replace the world): alive, each
            // entity once, every value readable
            std::vector<bool> seen(p.slots.size(), false);
            for (uint64_t k = 0; k < b.count; ++k)
            {
                uint32_t index = 0;
                r.read(index);
                if (!r.ok()) return fail(what + ": " + r.error());
                if (index == 0 || index >= p.slots.size() || !alive(p.slots[index]))
                    return fail(what + ": the entity index " + std::to_string(index) + " is not an alive slot");
                if (seen[index]) return fail(what + ": the entity index " + std::to_string(index) + " appears twice");
                seen[index] = true;
                std::string e;
                if (!decoder.skip(savedType, r, e))
                    return fail(what + ", entity index " + std::to_string(index) + ": " + e);
            }
            if (!r.atEnd()) return fail(what + ": bytes left over");
            continue;
        }
        if (target == Transform::id) transformSeen = true;
        const ComponentId id = ComponentId(std::max(target, 0));
        if ((mode == Mode::Fast || mode == Mode::Slow || mode == Mode::Replace) && id != Transform::id && !staged[id])
            staged[id] = reg.poolById(id)->newStage();

        for (uint64_t k = 0; k < b.count; ++k)
        {
            uint32_t index = 0;
            r.read(index);
            if (!r.ok()) return fail(what + ": " + r.error());
            const std::string where = what + ", entity index " + std::to_string(index);
            if (mode == Mode::DropEntity)
            {
                if (index == 0 || index >= p.slots.size() || !alive(p.slots[index]))
                    return fail(where + ": not an alive slot");
                drop[index] = true;
                json11::Json ignored;
                std::string e;
                if (!decoder.decode(savedType, r, nullptr, ignored, e)) return fail(where + ": " + e);
                continue;
            }
            if (!claim(id, index, what)) return false;
            if (mode == Mode::Fast)
            {
                if (id == Transform::id)
                {
                    Transform t;
                    r.read(t);
                    transforms.indices.push_back(index);
                    transforms.values.push_back(std::move(t));
                }
                else
                    reg.poolById(id)->stageBinary(*staged[id], index, r);
                if (!r.ok()) return fail(where + ": " + r.error());
                continue;
            }
            json11::Json json;
            std::string e;
            // Slow: migrated to the registered version; Replace: the removed type's own migrations, then by name
            if (!decoder.decode(savedType, r, mode == Mode::Slow ? targetType : nullptr, json, e))
                return fail(where + ": " + e);
            if (mode == Mode::Replace)
            {
                FieldMap fm;
                fm.fields = json.object_items();
                for (auto it = fm.fields.begin(); it != fm.fields.end();)
                {
                    bool known = false;
                    for (const FieldInfo& f : targetType->fields) known = known || f.name == it->first;
                    it = known ? std::next(it) : fm.fields.erase(it);
                }
                json = json11::Json(fm.fields);
            }
            if (id == Transform::id)
            {
                Transform t;
                if (!fromJson(json, t, &e, false)) return fail(where + ": " + e);
                transforms.indices.push_back(index);
                transforms.values.push_back(std::move(t));
            }
            else if (!reg.poolById(id)->stageJson(*staged[id], index, json, e))
                return fail(where + ": " + e);
        }
        if (!r.atEnd()) return fail(what + ": bytes left over");
    }
    // Every alive entity has a Transform
    for (size_t i = 1; i < p.slots.size(); ++i)
        if (alive(p.slots[i]) && (!transformSeen || seen[Transform::id].empty() || !seen[Transform::id][i]))
            return fail("the alive entity index " + std::to_string(i) + " has no Transform");
    return true;
}

bool WorldLoader::stageEvents(const std::vector<uint8_t>& data)
{
    SaveDecoder decoder(p.schema, &current, &w.migrationList);
    stagedEvents.resize(w.events.all().size());
    for (const ParsedSave::Block& b : p.events)
    {
        const SchemaType& savedType = p.schema.types[b.schemaIndex];
        BinaryReader r(data.data() + b.begin, b.end - b.begin);
        EventQueueBase* q = w.events.find(savedType.name);
        if (!q)
        {
            // Events live one step: a type that no longer exists is dropped (logged after the commit), after its
            // values were checked
            for (uint64_t k = 0; k < b.count; ++k)
            {
                std::string e;
                if (!decoder.skip(savedType, r, e)) return fail("event " + savedType.name + ": " + e);
            }
            if (!r.atEnd()) return fail("event " + savedType.name + ": bytes left over");
            droppedEvents.push_back(savedType.name + " (" + std::to_string(b.count) + ")");
            continue;
        }
        const SchemaType* targetType = current.find(SchemaKind::Event, savedType.name);
        std::string reason;
        const SchemaMatch m = compareTypes(p.schema, savedType, current, *targetType, reason);
        if (m == SchemaMatch::Incompatible) return fail(reason);
        auto& stage = stagedEvents[q->id];
        if (!stage) stage = q->newStage();
        for (uint64_t k = 0; k < b.count; ++k)
        {
            const std::string where = "event " + savedType.name;
            if (m == SchemaMatch::Same)
            {
                if (!q->stageBinary(*stage, r)) return fail(where + ": " + r.error());
                continue;
            }
            json11::Json json;
            std::string e;
            if (!decoder.decode(savedType, r, targetType, json, e) || !q->stageJson(*stage, json, e))
                return fail(where + ": " + e);
        }
        if (!r.atEnd()) return fail("event " + savedType.name + ": bytes left over");
    }
    return true;
}

template <class T>
void WorldLoader::filterStage(StagedOf<T>& s, const std::vector<bool>& keep)
{
    size_t n = 0;
    for (size_t i = 0; i < s.values.size(); ++i)
    {
        if (!keep[i]) continue;
        if (n != i)
        {
            s.values[n]  = std::move(s.values[i]);
            s.indices[n] = s.indices[i];
        }
        n++;
    }
    s.values.resize(n);
    s.indices.resize(n);
}

bool WorldLoader::mapClasses()
{
    if (!staged[Behaviour::id]) return true;
    auto& s = static_cast<StagedOf<Behaviour>&>(*staged[Behaviour::id]);
    std::vector<bool> keep(s.values.size(), true);
    for (size_t i = 0; i < s.values.size(); ++i)
    {
        Behaviour& b = s.values[i];
        if (s.indices[i] < p.slots.size() &&
            (p.slots[s.indices[i]].flags & (Registry::kWorldState | Registry::kSpaceState)))
            return fail("entity index " + std::to_string(s.indices[i]) + ": a world-state entity has a game object");
        const std::string* name = p.schema.className(b.classId);
        const std::string where = "entity index " + std::to_string(s.indices[i]);
        if (!name) return fail(where + ": the class ID " + std::to_string(b.classId) + " is not in the saved schema");
        const ClassList::ClassInfo* c = w.classes.find(*name);
        if (!c)
        {
            const MigrationList::Removed* removed = w.migrationList.findRemoved(*name);
            if (!removed)
                return fail(where + ": the class " + *name +
                            " is not registered and has no Removed entry (MigrationList::addRemoved)");
            if (removed->policy == RemovedPolicy::DropEntity)
                drop[s.indices[i]] = true;
            else if (removed->policy == RemovedPolicy::DropComponent)
                keep[i] = false;
            else
            {
                c = w.classes.find(removed->replacement);
                if (!c)
                    return fail("the replacement class " + removed->replacement + " of " + *name +
                                " is not registered");
            }
        }
        if (c) b.classId = c->id;
    }
    filterStage(s, keep);
    return true;
}

bool WorldLoader::dropEntities()
{
    // The descendants of dropped entities are dropped too (breadth first over the staged Parents)
    std::vector<std::vector<uint32_t>> childrenOf;
    if (staged[Parent::id])
    {
        const auto& s = static_cast<StagedOf<Parent>&>(*staged[Parent::id]);
        childrenOf.resize(p.slots.size());
        for (size_t i = 0; i < s.values.size(); ++i)
            if (s.values[i].entity.index < p.slots.size()) childrenOf[s.values[i].entity.index].push_back(s.indices[i]);
    }
    std::vector<uint32_t> queue;
    for (uint32_t i = 1; i < uint32_t(drop.size()); ++i)
        if (drop[i]) queue.push_back(i);
    for (size_t k = 0; k < queue.size(); ++k)
    {
        const uint32_t i = queue[k];
        if (p.slots[i].flags & (Registry::kWorldState | Registry::kSpaceState))
            return fail("the world-state entity " + std::to_string(i) + " would be dropped by a removed type");
        if (childrenOf.empty()) continue;
        for (uint32_t c : childrenOf[i])
            if (!drop[c])
            {
                drop[c] = true;
                queue.push_back(c);
            }
    }
    // Dropped slots become free like destroyed ones: the next generation, appended to the free list in index order
    // (a wrapped generation retires the slot)
    std::vector<uint32_t> dropped;
    for (uint32_t i = 1; i < uint32_t(drop.size()); ++i)
        if (drop[i]) dropped.push_back(i);
    for (uint32_t i : dropped)
    {
        auto& s      = p.slots[i];
        s.state      = uint8_t(SlotState::Free);
        s.flags      = 0;
        s.generation = s.generation + 1;
        if (s.generation != 0) p.freeList.push_back(i);
    }
    generation.assign(p.slots.size(), -1);
    for (size_t i = 1; i < p.slots.size(); ++i)
        if (alive(p.slots[i])) generation[i] = int64_t(p.slots[i].generation);
    return true;
}

bool WorldLoader::checkHierarchy()
{
    if (!staged[Parent::id]) return true;
    const auto& s = static_cast<StagedOf<Parent>&>(*staged[Parent::id]);
    std::vector<int64_t> parentOf(p.slots.size(), -1);
    for (size_t i = 0; i < s.values.size(); ++i)
    {
        const uint32_t child = s.indices[i];
        if (generation[child] < 0) continue;
        const Entity parent     = s.values[i].entity;
        const std::string where = "entity index " + std::to_string(child);
        if (parent.index == 0 || parent.index >= generation.size() ||
            generation[parent.index] != int64_t(parent.generation))
            return fail(where + ": its Parent names an entity that is not alive");
        if (p.slots[child].flags & (Registry::kWorldState | Registry::kSpaceState))
            return fail(where + ": a world-state entity has a Parent");
        parentOf[child] = parent.index;
    }
    // No cycles: 0 unvisited, 1 on the current path, 2 done
    std::vector<uint8_t> state(p.slots.size(), 0);
    std::vector<uint32_t> path;
    for (uint32_t start = 1; start < uint32_t(parentOf.size()); ++start)
    {
        if (state[start] != 0) continue;
        path.clear();
        int64_t x = start;
        while (x >= 0 && state[size_t(x)] == 0)
        {
            state[size_t(x)] = 1;
            path.push_back(uint32_t(x));
            x = parentOf[size_t(x)];
        }
        if (x >= 0 && state[size_t(x)] == 1)
            return fail("the Parents form a cycle (entity index " + std::to_string(x) + ")");
        for (uint32_t y : path) state[y] = 2;
    }
    return true;
}

bool WorldLoader::checkPersistent()
{
    if (!staged[Persistent::id]) return true;
    const auto& s = static_cast<StagedOf<Persistent>&>(*staged[Persistent::id]);
    std::unordered_set<uint64_t> ids;
    for (size_t i = 0; i < s.values.size(); ++i)
    {
        if (generation[s.indices[i]] < 0) continue;
        const uint64_t v = s.values[i].persistentId.value;
        if (v == 0) return fail("entity index " + std::to_string(s.indices[i]) + ": an invalid PersistentId");
        if (!ids.insert(v).second) return fail("the PersistentId " + std::to_string(v) + " is used twice");
        if (v > p.persistentCounter)
            return fail("the PersistentId " + std::to_string(v) + " is above the saved counter");
    }
    return true;
}

bool WorldLoader::checkFreedStateEntities()
{
    // A state entity whose components are no longer registered as world or space state is freed by the load; it must
    // not have children, a game object or a PersistentId (they would be left dangling)
    bool anyGlobal = false, anySpace = false;
    for (size_t id = 0; id < kMaxComponents; ++id)
    {
        anyGlobal = anyGlobal || reg.isGlobal(ComponentId(id));
        anySpace  = anySpace || reg.isSpaceState(ComponentId(id));
    }
    for (uint32_t i = 1; i < uint32_t(p.slots.size()); ++i)
    {
        if (generation[i] < 0) continue;
        const uint8_t f  = p.slots[i].flags;
        const bool freed = ((f & Registry::kWorldState) && !anyGlobal) || ((f & Registry::kSpaceState) && !anySpace);
        if (!freed) continue;
        auto has = [&](ComponentId id)
        {
            const auto& s = seen[id];
            return !s.empty() && s[i];
        };
        bool hasChildren = false;
        if (staged[Parent::id])
        {
            const auto& sp = static_cast<StagedOf<Parent>&>(*staged[Parent::id]);
            for (size_t k = 0; k < sp.values.size() && !hasChildren; ++k)
                hasChildren = generation[sp.indices[k]] >= 0 && sp.values[k].entity.index == i;
        }
        if (hasChildren || has(Persistent::id) || has(Behaviour::id))
            return fail("the state entity " + std::to_string(i) +
                        " would be freed (no state components are registered any more), but it has children, a "
                        "PersistentId or a game object");
    }
    return true;
}

void WorldLoader::commit()
{
    // The old entities leave (records as for a destroy: Transform first, then the components in ID order); no hooks
    for (uint32_t i = 1; i < reg.slotCount(); ++i)
    {
        const Entity e = reg.aliveAt(i);
        if (!e.valid()) continue;
        const ComponentMask m = reg.entityMask(e);
        size_t n              = 0;
        for (size_t id = 0; id < kMaxComponents; ++id) n += m.test(ComponentId(id)) ? 1 : 0;
        auto prepared = w.log.prepare(n);
        for (size_t id = 0; id < kMaxComponents; ++id)
            if (m.test(ComponentId(id))) prepared.append(e, ComponentId(id), false);
    }
    reg.clearForLoad();

    // The members that are not saved are reset (rebuilt below)
    w.commands.clear();
    w.pendingReserved.clear();
    w.children.clear();
    w.changed.clear();
    w.hierarchyOrder.clear();
    w.hierarchyDirty = true;
    w.pendingChildTeleports.clear();
    w.persistentIndex.clear();
    w.pendingPersistent.clear();
    w.retired.clear();
    w.destroyQueue.clear();
    w.stepChanged.clear();
    w.tickPlayerCells.clear();
    w.worldState = Entity{};
    w.spaceStates.clear();

    std::vector<Registry::SlotRecord> table(p.slots.size());
    for (size_t i = 0; i < p.slots.size(); ++i)
        table[i] = Registry::SlotRecord{p.slots[i].generation, SlotState(p.slots[i].state), p.slots[i].flags};
    reg.restoreSlots(table, p.freeList, transforms);
    for (size_t id = 1; id < kMaxComponents; ++id)
        if (staged[id]) reg.commitStage(ComponentId(id), *staged[id], generation);

    // The loaded entities arrive (records as for a create and the adds)
    for (uint32_t i = 1; i < reg.slotCount(); ++i)
    {
        const Entity e = reg.aliveAt(i);
        if (!e.valid()) continue;
        const ComponentMask m = reg.entityMask(e);
        size_t n              = 0;
        for (size_t id = 0; id < kMaxComponents; ++id) n += m.test(ComponentId(id)) ? 1 : 0;
        auto prepared = w.log.prepare(n);
        for (size_t id = 0; id < kMaxComponents; ++id)
            if (m.test(ComponentId(id))) prepared.append(e, ComponentId(id), true);
    }

    const auto& queues = w.events.all();
    for (size_t id = 0; id < queues.size(); ++id)
        if (queues[id]) queues[id]->restore(id < stagedEvents.size() ? stagedEvents[id].get() : nullptr);
    w.actions.clear();
    // As in the live world: a queued action stays also if its player is gone (deliverActions emits it with the stale
    // handle), so a loaded world delivers the same events
    for (const ParsedSave::Action& a : p.actions)
        w.actions.push_back(World::QueuedAction{a.peer, a.sequence, PlayerAction{a.player, a.action, a.aim}});
    w.actionSequence = p.actionSequence;
    w.rng.setState(Rng::State{p.rngState, p.rngIncrement});
    w.persistentCounter = p.persistentCounter;
    w.steps             = p.steps;
    // The same step rate and a time that is exactly steps * dt: time() stays steps * dt (a loaded world continues with
    // the same time values bit for bit); otherwise the saved time is the base
    if (p.stepsPerSecond == w.config.stepsPerSecond && p.time == double(p.steps) * w.stepDt)
    {
        w.stepBase = 0;
        w.timeBase = 0;
    }
    else
    {
        w.stepBase = p.steps;
        w.timeBase = p.time;
    }

    // Derived state
    const bool hierarchyOk = w.rebuildHierarchyState();
    SAIGA_ASSERT(hierarchyOk, "load: the hierarchy was checked before the commit");
    if (reg.registered(Persistent::id))
        reg.view<const Persistent>().each([&](Entity e, const Persistent& x)
                                          { w.persistentIndex.emplace(x.persistentId.value, e); });
    restoreStateEntities();
    reg.markAllTransformsChanged();
    rebuildSpatialGrid(w);
}

void WorldLoader::restoreStateEntities()
{
    for (uint32_t i = 1; i < reg.slotCount(); ++i)
    {
        const Entity e = reg.aliveAt(i);
        if (!e.valid()) continue;
        if (reg.flags(e) & Registry::kWorldState) w.worldState = e;
        if (reg.flags(e) & Registry::kSpaceState) w.spaceStates.push_back(e);
    }
    // Components registered as world or space state since the save get their state entity (the first free slot) and
    // their defaults; a state entity without any registered state component is freed
    auto restore = [&](Entity& e, uint8_t flag, bool (Registry::*is)(ComponentId) const)
    {
        bool any = false;
        for (size_t id = 0; id < kMaxComponents; ++id) any = any || (reg.*is)(ComponentId(id));
        if (!any)
        {
            if (e.valid())
            {
                const ComponentMask m = reg.entityMask(e);
                auto prepared         = w.log.prepare(kMaxComponents);
                reg.destroy(e);
                for (size_t id = 0; id < kMaxComponents; ++id)
                    if (m.test(ComponentId(id))) prepared.append(e, ComponentId(id), false);
                e = Entity{};
            }
            return;
        }
        if (!e.valid())
        {
            auto record = w.log.prepare(1);
            e           = reg.create();
            record.append(e, Transform::id, true);
            reg.setFlags(e, uint8_t(reg.flags(e) | flag));
        }
        for (size_t id = 0; id < kMaxComponents; ++id)
        {
            if (!(reg.*is)(ComponentId(id)) || reg.entityMask(e).test(ComponentId(id))) continue;
            auto record = w.log.prepare(1);
            reg.addDefault(e, ComponentId(id));
            record.append(e, ComponentId(id), true);
        }
    };
    restore(w.worldState, Registry::kWorldState, &Registry::isGlobal);
    Entity space = w.spaceStates.empty() ? Entity{} : w.spaceStates[0];
    restore(space, Registry::kSpaceState, &Registry::isSpaceState);
    w.spaceStates.clear();
    if (space.valid()) w.spaceStates.push_back(space);
}

void WorldLoader::startObjects(LoadResult& result)
{
    if (!reg.registered(Behaviour::id)) return;
    std::vector<Entity> objects;
    for (uint32_t i = 1; i < reg.slotCount(); ++i)
    {
        const Entity e = reg.aliveAt(i);
        if (e.valid() && reg.has<Behaviour>(e)) objects.push_back(e);
    }
    std::vector<Entity> failed;
    auto failure = [&](Entity e, const std::string& message)
    {
        const Behaviour* b            = reg.alive(e) ? reg.tryRead<Behaviour>(e) : nullptr;
        const ClassList::ClassInfo* c = b ? w.classes.find(b->classId) : nullptr;
        result.hookErrors.push_back(LoadResult::HookError{e, c ? c->name : "?", message});
        failed.push_back(e);
    };
    // A failed object is removed at once with its descendants, without hooks (their later hooks do not run); the
    // descendants are reported as removed
    auto removeFailed = [&](Entity e)
    {
        if (!reg.alive(e)) return;
        for (Entity x : w.collectDoomed(e))
            if (x != e) result.removed.push_back(x);
        w.abandonSpawn(e);
    };
    // Pass 1: every object exists before any hook runs (a hook may look at other objects)
    for (Entity e : objects)
    {
        Behaviour& b                  = reg.get<Behaviour>(e);
        const ClassList::ClassInfo* c = w.classes.find(b.classId);
        try
        {
            b.object        = c->factory();
            b.object->owner = &w;
            b.object->self  = e;
        }
        catch (const std::exception& x)
        {
            failure(e, std::string("the factory threw: ") + x.what());
        }
        catch (...)
        {
            failure(e, "the factory threw");
        }
    }
    for (Entity e : failed) removeFailed(e);
    // Pass 2: configure; pass 3: onLoad
    for (int phase = 1; phase <= 2; ++phase)
    {
        w.loadHookPhase = phase;
        for (Entity e : objects)
        {
            if (!reg.alive(e) || !reg.has<Behaviour>(e) || !reg.read<Behaviour>(e).object) continue;
            GameObject* o    = reg.get<Behaviour>(e).object.get();
            w.loadHookEntity = e;
            w.hookDepth++;
            bool threw = false;
            try
            {
                if (phase == 1)
                    o->configure();
                else
                    o->onLoad();
            }
            catch (const std::exception& x)
            {
                threw = true;
                failure(e, std::string(phase == 1 ? "configure" : "onLoad") + " threw: " + x.what());
            }
            catch (...)
            {
                threw = true;
                failure(e, std::string(phase == 1 ? "configure" : "onLoad") + " threw");
            }
            w.hookDepth--;
            if (threw)
            {
                w.loadHookPhase = 0;
                removeFailed(e);
                w.loadHookPhase = phase;
            }
        }
    }
    w.loadHookPhase  = 0;
    w.loadHookEntity = Entity{};
    w.releaseRetired();
}

LoadResult WorldLoader::run(const std::vector<uint8_t>& data)
{
    LoadResult result;
    // Refused in every build (the running game object and the views of the step would be destroyed under them)
    if (w.inStep() || !w.commands.empty() || w.hookDepth != 0 || w.loadHookPhase != 0 || w.reg.iterating() ||
        w.loadInProgress)
    {
        result.error = "a load only between steps, not inside a game hook, a view or another load (its listeners)";
        return result;
    }
    // Held until the listeners of this load ran (a nested load would hand the observers its records first)
    struct LoadGuard
    {
        bool& flag;
        ~LoadGuard() { flag = false; }
    } guard{w.loadInProgress = true};
    if (!parseSave(data, p, error) || !checkSlots())
    {
        result.error = error;
        return result;
    }
    current = SchemaTable::fromWorld(w);
    if (!stageComponents(data) || !stageEvents(data) || !mapClasses() || !dropEntities() || !checkHierarchy() ||
        !checkPersistent() || !checkFreedStateEntities())
    {
        result.error = error;
        return result;
    }
    // The defaults of the state components the commit may add are constructed once before the world changes (a
    // throwing constructor rejects the load instead of leaving half a world)
    for (size_t id = 1; id < kMaxComponents; ++id)
    {
        if (!reg.isGlobal(ComponentId(id)) && !reg.isSpaceState(ComponentId(id))) continue;
        try
        {
            reg.poolById(ComponentId(id))->probeDefault();
        }
        catch (const std::exception& x)
        {
            result.error =
                std::string("the default of ") + reg.componentInfo(ComponentId(id)).name + " threw: " + x.what();
            return result;
        }
    }

    const uint64_t logBegin = w.log.appendedTotal();
    const uint64_t fromStep = w.steps;
    commit();
    result.committed = true;
    w.loads++;
    for (const std::string& e : droppedEvents)
        SAIGA_LOG("save", "load: dropped the pending events of the unregistered type %s", e.c_str());
    startObjects(result);

    LoadInfo info;
    info.loadCount = w.loads;
    info.logBegin  = logBegin;
    info.logEnd    = w.log.appendedTotal();
    info.fromStep  = fromStep;
    info.toStep    = w.steps;
    // A copy: a listener may add or remove listeners
    auto listeners = w.loadListeners;
    for (auto& l : listeners)
    {
        // A listener removed by an earlier one (e.g. a world API rebound to another world) is not called
        bool registered = false;
        for (const auto& now : w.loadListeners) registered = registered || now.first == l.first;
        if (!registered) continue;
        // A throwing listener must not hide the committed load from the others (the world API's journal)
        try
        {
            l.second(info);
        }
        catch (const std::exception& x)
        {
            result.listenerErrors.push_back(x.what());
        }
        catch (...)
        {
            result.listenerErrors.push_back("a load listener threw");
        }
    }
    return result;
}

LoadResult World::tryLoad(const std::vector<uint8_t>& data)
{
    WorldLoader loader(*this);
    return loader.run(data);
}

void World::load(const std::vector<uint8_t>& data)
{
    const LoadResult r = tryLoad(data);
    if (!r.committed) SAIGA_EXIT_ERROR("World::load: " + r.error);
    if (!r.listenerErrors.empty()) SAIGA_EXIT_ERROR("World::load: a load listener threw: " + r.listenerErrors[0]);
    if (!r.hookErrors.empty())
        SAIGA_EXIT_ERROR("World::load: the game object of entity " + std::to_string(r.hookErrors[0].entity.index) +
                         " (" + r.hookErrors[0].className + ") failed: " + r.hookErrors[0].message);
}

}  // namespace Engine
}  // namespace Saiga
