/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// World: registration, commands, the step loop, advance and the hash.

#include "saiga/engine/world/World.h"

#include "saiga/engine/components/EngineComponents.h"
#include "saiga/engine/components/LocalTransform.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/core/Hash.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/gameplay/Systems.h"
#include "saiga/engine/world/Events.h"
#include "saiga/engine/world/TransformSystem.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>

namespace Saiga
{
namespace Engine
{

World::World(const WorldSetup& setup)
    : events(reg),
      pool(std::max(0, setup.workerThreads - 1)),
      rng(setup.seed),
      config(setup.config),
      stepDt(1.0 / setup.config.stepsPerSecond)
{
    SAIGA_ASSERT(setup.workerThreads >= 1, "WorldSetup::workerThreads must be at least 1 (the calling thread)");
    SAIGA_ASSERT(std::isfinite(setup.config.stepsPerSecond) && setup.config.stepsPerSecond > 0 &&
                     setup.config.maxStepsPerAdvance >= 1,
                 "invalid WorldConfig");
    reg.setJobPool(&pool);
    // Before any entity exists: every create reaches the grid through the log
    gridConsumer = log.addConsumer();

    // Engine lists first, then the game's (§7.9, §18.2)
    registerEngineComponents(reg);
    if (setup.components) setup.components(reg);
    reg.validateRegistration();

    EventRegistry eventRegistry(events);
    registerEngineEvents(eventRegistry);
    if (setup.events) setup.events(eventRegistry);
    events.validateRegistration();

    registerEngineSystems(systems);
    if (setup.systems) setup.systems(systems);

    if (setup.classes) setup.classes(classes);
    classes.validateRegistration();

    prof.enabled = setup.profiling;
    setupProfiler();

    createStateEntities();
}

void World::setupProfiler()
{
    stepScope = prof.scope("step");
    for (int p = kFirstPhase; p <= kLastPhase; ++p)
    {
        const std::string phasePath = std::string("step/phase:") + phaseName(Phase(p));
        phaseScopes[size_t(p)]      = prof.scope(phasePath);
        for (const SystemList::System& s : systems.systems(Phase(p)))
            systemScopes[size_t(p)].push_back(prof.scope(phasePath + "/system:" + s.name));
        flushScopes[size_t(p)] = prof.scope(phasePath + "/commands");
    }
    entitiesCounter = prof.counter("entities");
    objectsCounter  = prof.counter("game_objects");
    updatedCounter  = prof.counter("objects_updated");  // behaviourSystem adds to it
    commandsCounter = prof.counter("commands_applied");
    recordsCounter  = prof.counter("structural_records");
    gridCounter     = prof.counter("grid_items");
}

void World::createStateEntities()
{
    // The world-state entity and the space-state entity of space 0 exist when components are registered for them
    // (§7.8); before S21 there is only space 0
    bool anyGlobal = false, anySpace = false;
    for (size_t id = 0; id < kMaxComponents; ++id)
    {
        anyGlobal = anyGlobal || reg.isGlobal(ComponentId(id));
        anySpace  = anySpace || reg.isSpaceState(ComponentId(id));
    }
    auto make = [&](uint8_t flag, bool (Registry::*is)(ComponentId) const)
    {
        Entity e = create();
        reg.setFlags(e, uint8_t(reg.flags(e) | flag));
        for (size_t id = 0; id < kMaxComponents; ++id)
        {
            if (!(reg.*is)(ComponentId(id))) continue;
            auto record = log.prepare(1);
            reg.addDefault(e, ComponentId(id));
            record.append(e, ComponentId(id), true);
        }
        return e;
    };
    if (anyGlobal) worldState = make(Registry::kWorldState, &Registry::isGlobal);
    if (anySpace) spaceStates.push_back(make(Registry::kSpaceState, &Registry::isSpaceState));
}

// ---- game objects and gameplay
// -------------------------------------------------------------------------------------------

void World::addDefaultNow(Entity e, ComponentId id)
{
    SAIGA_ASSERT(!inStep(), "addDefaultNow: between steps");
    SAIGA_ASSERT(id != Transform::id && id != Behaviour::id && id != Persistent::id && id != Parent::id,
                 "addDefaultNow: this component has its own path");
    auto record = log.prepare(1);
    reg.addDefault(e, id);
    record.append(e, id, true);
}

void World::removeByIdNow(Entity e, ComponentId id)
{
    SAIGA_ASSERT(!inStep(), "removeByIdNow: between steps");
    if (id == Parent::id) return removeNow<Parent>(e);
    if (id == Persistent::id) return removeNow<Persistent>(e);
    if (id == Behaviour::id) return removeNow<Behaviour>(e);  // the object may be running: retired
    auto record = log.prepare(1);
    reg.removeById(e, id);
    record.append(e, id, false);
}

Entity World::spawnByClass(const ClassList::ClassInfo& info, SpaceId space, const Vec3& position,
                           const std::function<void(Entity)>& initial)
{
    SAIGA_ASSERT(!inStep(), "spawnByClass: between steps");
    const Entity e = beginSpawn(space);
    try
    {
        finishSpawn(e, info, position);
        if (initial) initial(e);
    }
    catch (...)
    {
        abandonSpawn(e);
        throw;
    }
    startObject(e);
    return e;
}

void World::runStepListeners()
{
    // Listeners may add listeners (kept) and remove themselves (return false). A throwing listener is removed; the
    // others are still notified of this (committed) step, then the first exception is rethrown.
    std::vector<std::pair<int, std::function<bool()>>> current;
    current.swap(stepListeners);
    removedStepListeners.clear();
    std::exception_ptr first;
    for (auto& l : current)
    {
        if (std::find(removedStepListeners.begin(), removedStepListeners.end(), l.first) != removedStepListeners.end())
            continue;
        try
        {
            if (l.second()) stepListeners.push_back(std::move(l));
        }
        catch (...)
        {
            if (!first) first = std::current_exception();
        }
    }
    removedStepListeners.clear();
    if (first) std::rethrow_exception(first);
}

Entity World::beginSpawn(SpaceId space)
{
    checkNotParallel("spawn");
    SAIGA_ASSERT(space == 0, "spawnIn: only space 0 exists before S21");
    return create();
}

void World::finishSpawn(Entity e, const ClassList::ClassInfo& info, const Vec3& position)
{
    reg.get<Transform>(e).setPosition(position);
    Behaviour b;
    b.object        = info.factory();
    b.object->owner = this;
    b.object->self  = e;
    b.classId       = info.id;
    addNow<Behaviour>(e, std::move(b));
}

void World::startObject(Entity e)
{
    // configure or onSpawn may destroy the object (a command during the flush applies at once)
    if (!reg.alive(e)) return;
    Behaviour& b = reg.get<Behaviour>(e);
    assignTickInterval(*this, e, b, true);
    GameObject* o = b.object.get();
    // The object stays valid while its hooks run, also if they destroy its entity (released afterwards)
    hookDepth++;
    try
    {
        o->configure();
        if (reg.alive(e) && reg.has<Behaviour>(e)) o->onSpawn();
    }
    catch (...)
    {
        hookDepth--;
        // A half-started object must not tick: the entity is destroyed (without its onDestroy)
        abandonSpawn(e);
        throw;
    }
    hookDepth--;
    releaseRetired();
}

void World::abandonSpawn(Entity e)
{
    if (!reg.alive(e)) return;
    // The whole subtree without hooks (a hook of a child created by the failed configure must not stop the cleanup)
    for (Entity x : collectDoomed(e))
        if (Behaviour* b = reg.tryGet<Behaviour>(x)) b->destroying = true;
    // At once, also inside a destroy (the queue would be dropped by the exception that is propagating)
    destroyOne(e);
}

void World::queueAction(const PlayerAction& action)
{
    checkNotParallel("queueAction");
    if (!reg.alive(action.player) || !reg.has<Player>(action.player))
    {
        // Refused in every build
        SAIGA_ASSERT(false, "queueAction: the player is not alive or has no Player component");
        return;
    }
    actions.push_back(QueuedAction{reg.read<Player>(action.player).peer, actionSequence++, action});
}

void World::deliverActions()
{
    // (peer, sequence) order, exactly once: the queue becomes pending ActionEvents, readable in this step
    std::stable_sort(actions.begin(), actions.end(), [](const QueuedAction& a, const QueuedAction& b)
                     { return a.peer != b.peer ? a.peer < b.peer : a.sequence < b.sequence; });
    for (const QueuedAction& a : actions) events.emit(ActionEvent{a.action.player, a.action.action, a.action.aim});
    actions.clear();
}

PersistentId World::makePersistent(Entity e)
{
    checkNotParallel("makePersistent");
    if (!reg.alive(e) && !reg.reserved(e))
    {
        // Refused in every build
        SAIGA_ASSERT(false, "makePersistent: the entity is not alive");
        return PersistentId{};
    }
    if (reg.alive(e))
        if (const Persistent* p = reg.tryRead<Persistent>(e)) return p->persistentId;
    // Assigned earlier in this phase (the component is added at the flush): the same ID (idempotent)
    for (const auto& pending : pendingPersistent)
        if (pending.first == e) return pending.second;
    const PersistentId id{++persistentCounter};
    Persistent value;
    value.persistentId = id;
    if (deferring() || (flushing && reg.reserved(e)))
    {
        pendingPersistent.push_back({e, id});
        record(makeCommand([e, value](World& w) mutable { w.addNow<Persistent>(e, std::move(value)); }));
    }
    else
        addNow<Persistent>(e, std::move(value));
    return id;
}

Entity World::resolve(PersistentId id) const
{
    auto it = persistentIndex.find(id.value);
    if (it == persistentIndex.end() || !reg.alive(it->second)) return Entity{};
    return it->second;
}

World::~World()
{
    // Observers (the world API, waiting tools) stop using the world
    for (auto& l : destroyListeners) l.second();
}

// ---- spatial queries
// ----------------------------------------------------------------------------------------------

std::vector<Entity> World::queryRadius(SpaceId space, const Vec3& center, double radius,
                                       const ComponentMask& mask) const
{
    std::vector<Entity> out;
    queryRadius(space, center, radius, mask, out);
    return out;
}

void World::queryRadius(SpaceId space, const Vec3& center, double radius, const ComponentMask& mask,
                        std::vector<Entity>& out) const
{
    out.clear();
    forEachWithin(space, center, radius, mask, [&](Entity e, const Vec3&) { out.push_back(e); });
    std::sort(out.begin(), out.end());
}

// ---- structural changes
// ----------------------------------------------------------------------------------------------

Entity World::create()
{
    checkNotParallel("create");
    if (deferring())
    {
        // Reserved now (views skip it), alive when the command is applied; reserving never moves a Transform
        // Everything the abort cleanup needs is allocated now (it must not allocate while an exception propagates)
        // (the registry keeps room for releasing every reserved entity)
        // Geometric growth: reserving exactly size + 1 reallocated on every create (quadratic in creates per phase)
        if (pendingReserved.capacity() == pendingReserved.size())
            pendingReserved.reserve(std::max<size_t>(16, 2 * pendingReserved.capacity()));
        Entity e = reg.reserve();
        pendingReserved.push_back(e);
        record(makeCommand([e](World& w) { w.activateNow(e); }));
        return e;
    }
    auto record = log.prepare(1);  // the record cannot be lost after the change (append does not allocate)
    Entity e    = reg.create();
    record.append(e, Transform::id, true);
    return e;
}

void World::destroy(Entity e)
{
    checkNotParallel("destroy");
    if (reg.alive(e) && (reg.flags(e) & (Registry::kWorldState | Registry::kSpaceState)))
    {
        // Refused in every build (the world state must stay; asserts in builds with asserts)
        SAIGA_ASSERT(false, "destroy: world-state and space-state entities cannot be destroyed");
        return;
    }
    // During a flush an entity reserved later in the phase is not active yet: the destroy follows its activation
    if (deferring() || (flushing && reg.reserved(e)))
    {
        SAIGA_ASSERT(reg.alive(e) || reg.reserved(e), "destroy: the entity is not alive");
        // Dropped if the entity is gone when the command is applied (e.g. destroyed with its parent)
        record(makeCommand(
            [e](World& w)
            {
                if (w.reg.alive(e)) w.destroyNow(e);
            }));
        return;
    }
    destroyNow(e);
}

void World::activateNow(Entity e)
{
    auto record = log.prepare(1);
    reg.activate(e);
    record.append(e, Transform::id, true);
}

std::vector<Entity> World::collectDoomed(Entity e) const
{
    // The entity and its descendants (§7.4), parents before children. One pass over the Parent pool builds the
    // (parent index, child) list, sorted for a deterministic order; the visited set stops at cycles (add<Parent>
    // rejects them, this keeps destroy finite regardless).
    std::vector<Entity> doomed{e};
    if (childCount(e) == 0) return doomed;
    std::vector<std::pair<uint32_t, Entity>> links;
    const_cast<Registry&>(reg).view<const Parent>().each([&](Entity child, const Parent& p)
                                                         { links.push_back({p.entity.index, child}); });
    std::sort(links.begin(), links.end(), [](const std::pair<uint32_t, Entity>& a, const std::pair<uint32_t, Entity>& b)
              { return a.first != b.first ? a.first < b.first : a.second.index < b.second.index; });
    std::vector<bool> visited(reg.slotCount(), false);
    visited[e.index] = true;
    for (size_t i = 0; i < doomed.size(); ++i)
    {
        const Entity parent = doomed[i];
        auto it =
            std::lower_bound(links.begin(), links.end(), parent.index,
                             [](const std::pair<uint32_t, Entity>& l, uint32_t index) { return l.first < index; });
        for (; it != links.end() && it->first == parent.index; ++it)
        {
            const Entity child = it->second;
            if (childCount(parent) == 0) break;
            if (visited[child.index] || reg.read<Parent>(child).entity != parent) continue;
            visited[child.index] = true;
            doomed.push_back(child);
        }
    }
    return doomed;
}

void World::destroyNow(Entity e)
{
    // A destroy requested while a destroy runs (an onDestroy hook) is queued and applied after it, iteratively: no
    // recursion depth per chained destroy
    if (destroyDepth > 0)
    {
        destroyQueue.push_back(e);
        return;
    }
    destroyQueue.push_back(e);
    try
    {
        for (size_t i = 0; i < destroyQueue.size(); ++i)
            if (reg.alive(destroyQueue[i])) destroyOne(destroyQueue[i]);
    }
    catch (...)
    {
        destroyQueue.clear();
        throw;
    }
    destroyQueue.clear();
}

void World::destroyOne(Entity e)
{
    std::vector<Entity> doomed = collectDoomed(e);
    // The game objects first, children before parents, while all components are present. A hook may destroy other
    // entities (queued, applied after this destroy), spawn new ones or attach entities to a doomed parent: the
    // descendants are collected again after the hooks, and the new ones are destroyed without their onDestroy (a
    // hook that keeps adding children cannot loop).
    std::vector<Entity> hooked;
    destroyDepth++;
    hookDepth++;
    try
    {
        for (auto it = doomed.rbegin(); it != doomed.rend(); ++it)
        {
            if (!reg.alive(*it)) continue;
            Behaviour* b = reg.tryGet<Behaviour>(*it);
            if (!b || !b->object || b->destroying) continue;
            b->destroying = true;
            hooked.push_back(*it);
            b->object->onDestroy();
        }
        if (reg.alive(e)) doomed = collectDoomed(e);
        for (Entity x : doomed)
            if (Behaviour* b = reg.tryGet<Behaviour>(x)) b->destroying = true;
    }
    catch (...)
    {
        // Nothing was destroyed: the entities can be destroyed again later (their onDestroy runs again then)
        for (Entity x : hooked)
            if (Behaviour* b = reg.alive(x) ? reg.tryGet<Behaviour>(x) : nullptr) b->destroying = false;
        destroyDepth--;
        hookDepth--;
        throw;
    }
    destroyDepth--;
    hookDepth--;
    for (auto it = doomed.rbegin(); it != doomed.rend(); ++it)
    {
        const Entity x = *it;
        // A hook (onDestroy) may have destroyed it already
        if (!reg.alive(x)) continue;
        // Snapshot, reserve the log records and the room for the retired object, then destroy (the only step that
        // can throw, before any change); the bookkeeping is committed afterwards without allocation
        const ComponentMask m = reg.entityMask(x);
        size_t records        = 0;
        for (size_t id = 0; id < kMaxComponents; ++id) records += m.test(ComponentId(id)) ? 1 : 0;
        auto prepared = log.prepare(records);
        retired.reserve(retired.size() + 1);
        const Parent* p        = reg.tryRead<Parent>(x);
        const Entity oldParent = p ? p->entity : Entity{};
        const Persistent* pid  = reg.tryRead<Persistent>(x);
        const uint64_t oldId   = pid ? pid->persistentId.value : 0;
        // The game object may still be running (a hook that destroyed its own entity): it is released when no hook
        // runs any more
        if (Behaviour* b = reg.tryGet<Behaviour>(x); b && b->object) retired.push_back(std::move(b->object));
        reg.destroy(x);
        if (oldId) persistentIndex.erase(oldId);
        if (oldParent.valid()) changeChildCount(oldParent, -1);
        if (x.index < children.size()) children[x.index] = 0;
        // One removed record per component in ID order, Transform first
        for (size_t id = 0; id < kMaxComponents; ++id)
            if (m.test(ComponentId(id))) prepared.append(x, ComponentId(id), false);
        if (m.test(Parent::id) || doomed.size() > 1) hierarchyDirty = true;
    }
    // An entity whose onDestroy ran but that a hook released from the cascade (e.g. removed its Parent) lives on:
    // its flag is reset, so its real destroy later calls onDestroy again
    for (Entity x : hooked)
        if (Behaviour* b = reg.alive(x) ? reg.tryGet<Behaviour>(x) : nullptr) b->destroying = false;
    releaseRetired();
}

void World::releaseRetired()
{
    // Game objects of destroyed entities (or removed Behaviours) live until no hook runs any more
    if (hookDepth == 0) retired.clear();
}

bool World::rebuildHierarchyState()
{
    // Built separately and validated before it replaces the old counts: every Parent must name an alive entity (a
    // stale or out-of-range handle from a direct registry fill is rejected in every build)
    std::vector<uint32_t> counts(size_t(reg.slotCount()), 0);
    bool ok = true;
    if (reg.registered(Parent::id))
        reg.view<const Parent>().each(
            [&](Entity, const Parent& p)
            {
                if (!reg.alive(p.entity))
                    ok = false;
                else
                    counts[p.entity.index]++;
            });
    if (!ok) return false;
    // No cycles: every chain must reach a root within slotCount links
    if (reg.registered(Parent::id))
    {
        const uint32_t max = reg.slotCount();
        reg.view<const Parent>().each(
            [&](Entity, const Parent& p)
            {
                Entity a = p.entity;
                for (uint32_t depth = 0; ok; ++depth)
                {
                    const Parent* up = reg.tryRead<Parent>(a);
                    if (!up) break;
                    if (depth > max || !reg.alive(up->entity)) ok = false;
                    a = up->entity;
                }
            });
    }
    if (!ok) return false;
    children.swap(counts);
    hierarchyDirty = true;
    return true;
}

uint32_t World::childCount(Entity e) const
{
    return e.index < children.size() ? children[e.index] : 0;
}

void World::reserveChildSlot(Entity parent)
{
    if (parent.index >= children.size()) children.resize(size_t(reg.slotCount()), 0);
}

void World::changeChildCount(Entity parent, int delta)
{
    reserveChildSlot(parent);
    SAIGA_ASSERT(delta > 0 || children[parent.index] > 0, "child count underflow");
    children[parent.index] = uint32_t(int64_t(children[parent.index]) + delta);
}

bool World::acceptParent(Entity child, Entity parent)
{
    // World-state entities are never children (a parent's destroy would take them along); refused in every build
    if (reg.flags(child) & (Registry::kWorldState | Registry::kSpaceState))
    {
        SAIGA_ASSERT(false, "add<Parent>: world-state and space-state entities cannot have a parent");
        return false;
    }
    if (!reg.alive(parent))
    {
        // During a flush the parent may have been destroyed earlier in the same flush: the child follows it
        // Between steps a dead parent is an error (refused without asserts); during a flush the parent may have been
        // destroyed earlier in the same flush: the child follows it
        SAIGA_ASSERT(flushing, "add<Parent>: the parent is not alive");
        if (flushing) destroyNow(child);
        return false;
    }
    // No cycles: the child must not be an ancestor of the parent (or the parent itself). An acyclic chain has fewer
    // links than there are slots, so a longer walk means an existing cycle: refused as well (never accepted early).
    Entity a           = parent;
    const uint32_t max = reg.slotCount();
    for (uint32_t depth = 0;; ++depth)
    {
        SAIGA_ASSERT(a != child, "add<Parent>: the parent is the child or one of its descendants (a cycle)");
        if (a == child || depth > max) return false;
        const Parent* p = reg.tryRead<Parent>(a);
        if (!p) return true;
        a = p->entity;
    }
}

void World::abortStep()
{
    // The world leaves the step first (whatever happens below, the next step can run)
    phase              = Phase::None;
    flushing           = false;
    const bool restore = stepEventsBegun;
    stepEventsBegun    = false;
    // The commands of the interrupted phase are dropped, the entities they would have activated released, and the
    // events of the step are restored: the readable ones become pending again, the ones the aborted step emitted are
    // dropped like its commands. None of this allocates (the capacity for the releases was reserved when the entities
    // were reserved; the event restore is a swap), so the cleanup cannot fail while the exception propagates.
    commands.clear();
    if (restore) events.abortStep();
    pendingPersistent.clear();
    for (Entity e : pendingReserved)
        if (reg.reserved(e)) reg.release(e);
    pendingReserved.clear();
}

void World::record(std::unique_ptr<Command> c)
{
    commands.push_back(std::move(c));
}

void World::flushCommands()
{
    // Changes made by a command apply immediately (flushing is set, so nothing is recorded during a flush)
    flushing = true;
    for (size_t i = 0; i < commands.size(); ++i)
    {
        std::unique_ptr<Command> c = std::move(commands[i]);
        c->apply(*this);
    }
    commands.clear();
    pendingReserved.clear();
    pendingPersistent.clear();
    flushing = false;
    releaseRetired();
}

void World::checkNotParallel(const char* operation) const
{
    SAIGA_ASSERT(!reg.inParallelEach(), std::string(operation) + " inside parallelEach (R-PAR-1)");
}

// ---- steps
// -----------------------------------------------------------------------------------------------------------

Rng& World::random()
{
    checkNotParallel("random");
    return rng;
}

void World::runStep()
{
    // Timed by hand: an aborted step's time must not reach the next sample (a scope timer would add it after the
    // discard)
    const auto stepStart = prof.enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    recordsAtStepStart   = log.appendedTotal();
    // Queued player actions become ActionEvents; with the events of the last step they become readable
    deliverActions();
    events.beginStep();
    stepEventsBegun = true;
    try
    {
        for (int p = kFirstPhase; p <= kLastPhase; ++p)
        {
            phase = Phase(p);
            Profiler::Timer phaseTimer(prof, phaseScopes[size_t(p)]);
            // Teleports made in phase 7 reach the children before extract reads them (§7.4)
            if (phase == Phase::Extract) propagateTeleportsBeforeExtract(*this);
            const auto& list = systems.systems(phase);
            for (size_t i = 0; i < list.size(); ++i)
            {
                // Before S16 the world is the host: client-only systems do not run
                if (list[i].runOn == RunOn::Client) continue;
                Profiler::Timer systemTimer(prof, systemScopes[size_t(p)][i]);
                list[i].fn(*this, stepDt);
            }
            Profiler::Timer flushTimer(prof, flushScopes[size_t(p)]);
            prof.addCounter(commandsCounter, double(commands.size()));
            flushCommands();
        }
        // Children whose parent was teleported after their last update (phase 8) follow in the next phase 6
        collectLateChildTeleports(*this);
    }
    catch (...)
    {
        // The world stays usable: the step is incomplete (the changes of the finished phases remain), the pending
        // commands are dropped and their reserved entities released; the step index does not advance
        abortStep();
        prof.discardSample();
        throw;
    }
    phase           = Phase::None;
    stepEventsBegun = false;
    stepChanged.clear();
    reg.shiftTransformFlags(trackStepChanges > 0 ? &stepChanged : nullptr);
    // The records of this step (consumers that read after the step, e.g. the world API, keep older ones a step longer)
    prof.setCounter(recordsCounter, double(log.appendedTotal() - recordsAtStepStart));
    log.truncate();
    steps++;
    prof.setCounter(entitiesCounter, double(reg.aliveCount()));
    prof.setCounter(objectsCounter, double(reg.registered(Behaviour::id) ? reg.poolById(Behaviour::id)->size() : 0));
    prof.setCounter(gridCounter, double(grid.size()));
    if (prof.enabled)
        prof.add(stepScope,
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stepStart).count());
}

void World::step(int n)
{
    SAIGA_ASSERT(!inStep(), "World::step inside a step");
    for (int i = 0; i < n; ++i)
    {
        runStep();
        // After runStep: its step timer has stopped
        prof.endSample();
        runStepListeners();
    }
}

void World::advance(double frameDt)
{
    SAIGA_ASSERT(!inStep(), "World::advance inside a step");
    if (paused) return;
    // NaN would poison the accumulator for good; +inf is a very long frame (the backlog is dropped below)
    if (std::isnan(frameDt)) frameDt = 0;
    if (std::isinf(frameDt)) frameDt = frameDt > 0 ? stepDt * (config.maxStepsPerAdvance + 1) : 0;
    accumulator += std::max(frameDt, 0.0) * scale;
    int n = 0;
    while (accumulator >= stepDt && n < config.maxStepsPerAdvance)
    {
        {
            runStep();
            // After runStep: its step timer has stopped
            prof.endSample();
            runStepListeners();
        }
        accumulator -= stepDt;
        n++;
    }
    if (accumulator >= stepDt)
    {
        // The backlog beyond maxStepsPerAdvance is dropped (a slow frame does not cause a spiral of catch-up steps)
        // The whole steps that are discarded (0.999 dt of the backlog is kept)
        const double kept      = 0.999 * stepDt;
        const double discarded = std::floor((accumulator - kept) / stepDt);
        // Clamped before the conversion (a huge or infinite backlog)
        dropped += discarded >= 1e18 ? uint64_t(1e18) : uint64_t(discarded);
        accumulator = kept;
    }
}

void World::setTimeScale(double s)
{
    SAIGA_ASSERT(std::isfinite(s) && s >= 0, "World::setTimeScale: the scale must be finite and not negative");
    scale = s;
}

void World::setWorkerThreads(int n)
{
    SAIGA_ASSERT(n >= 1, "World::setWorkerThreads: at least 1 (the calling thread)");
    SAIGA_ASSERT(!inStep(), "World::setWorkerThreads inside a step");
    pool.setWorkerThreads(n - 1);
}

uint64_t World::hash() const
{
    SAIGA_ASSERT(!inStep() && commands.empty(), "World::hash inside a step");
    Hasher h;
    reg.hashInto(h);
    const Rng::State s = rng.state();
    h.addU64(s.state).addU64(s.increment);
    events.hashPending(h);
    // The queued actions (input that becomes events) and the Persistent counter (S10)
    h.addU64(uint64_t(actions.size()));
    for (const QueuedAction& a : actions)
    {
        // The queue order (the sequence counter itself is not state: only the order matters)
        h.addU64(a.peer).addU32(a.action.player.index).addU32(a.action.player.generation);
        h.addU32(a.action.action);
        for (int i = 0; i < 3; ++i) h.addDouble(a.action.aim[i]);
    }
    h.addU64(persistentCounter);
    h.addU64(steps);
    return h.value();
}

}  // namespace Engine
}  // namespace Saiga
