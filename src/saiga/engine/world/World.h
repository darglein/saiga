/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/engine/components/EngineIds.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/Persistent.h"
#include "saiga/engine/core/JobPool.h"
#include "saiga/engine/core/Profiler.h"
#include "saiga/engine/core/Rng.h"
#include "saiga/engine/ecs/Registry.h"
#include "saiga/engine/gameplay/ClassList.h"
#include "saiga/engine/gameplay/PlayerAction.h"
#include "saiga/engine/gameplay/TickPolicy.h"
#include "saiga/engine/world/Commands.h"
#include "saiga/engine/world/EventQueues.h"
#include "saiga/engine/world/EventRegistry.h"
#include "saiga/engine/world/Phase.h"
#include "saiga/engine/world/SpatialGrid.h"
#include "saiga/engine/world/StructuralLog.h"
#include "saiga/engine/world/SystemList.h"
#include "saiga/export.h"

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <type_traits>
#include <unordered_map>

namespace Saiga
{
namespace Engine
{
// Game policies of a world (R-MECH-1; docs/engine_design.md §7.9). Later stages add their fields.
struct WorldConfig
{
    double stepsPerSecond  = 60;
    int maxStepsPerAdvance = 4;
    TickPolicy ticking;  // §8.4 (S10)
};

struct WorldSetup
{
    uint64_t seed = 1;
    // Threads of the frame JobPool for parallelEach, including the calling thread (1: serial)
    int workerThreads = 1;
    WorldConfig config;
    // The game's registration lists; the engine's lists are always registered first
    void (*components)(Registry&)  = nullptr;
    void (*events)(EventRegistry&) = nullptr;
    void (*systems)(SystemList&)   = nullptr;
    void (*classes)(ClassList&)    = nullptr;  // game object classes (S10)
    // Times every step, phase, system and command flush and counts entities, objects and updates (Profiler, S10);
    // false: no clock calls
    bool profiling = true;
};

/**
 * The simulation (docs/engine_design.md §7.6-§7.10, §8.1-§8.3, §8.6). Owns the Registry and runs fixed steps through
 * the phases of §8.1; the systems of a phase run in registration order and the commands they record are applied after
 * the phase.
 *
 * Structural changes (create, destroy, add, remove):
 * - between steps (setup code, tests, loading, tool handlers) and while commands are applied: immediately;
 * - during a step (systems): as commands, applied at the end of the phase in recording order. create() reserves the
 *   entity and returns it at once; views skip it and get/read/has assert until the command is applied; add on it is
 *   allowed (also a command). Commands on an entity that was destroyed earlier in the same flush are dropped.
 * Every applied change is appended to the structural log. Destroying a parent destroys its descendants (§7.4).
 */
class SAIGA_ENGINE_API World
{
   public:
    explicit World(const WorldSetup& setup);
    ~World();
    World(const World&)            = delete;
    World& operator=(const World&) = delete;

    // ---- entities and components (§7.10) -------------------------------------------------------------------------

    Entity create();
    // A game object of class T at position (space 0; spawnIn: S21 adds more spaces): Transform, Behaviour and the
    // initial components, then configure() and onSpawn() (§9.2). During a step the entity is reserved and returned at
    // once, the rest happens at the flush; between steps at once. Defined in gameplay/Behaviour.h.
    template <class T, class... C>
    Entity spawn(const Vec3& position, C&&... initial);
    template <class T, class... C>
    Entity spawnIn(SpaceId space, const Vec3& position, C&&... initial);
    // The game object's onDestroy() runs first (children before parents in the cascade). World-state entities cannot
    // be destroyed.
    void destroy(Entity e);
    bool alive(Entity e) const { return reg.alive(e); }

    template <class T, class... A>
    void add(Entity e, A&&... args);
    template <class T>
    void remove(Entity e);
    // Parent is read-only through the World: change it with remove + add (the hierarchy order and the cycle check
    // depend on it)
    template <class T>
    T& get(Entity e)
    {
        static_assert(T::id != ComponentId(EngineComponent::Parent), "Parent: use remove<Parent> and add<Parent>");
        static_assert(T::id != ComponentId(EngineComponent::Persistent), "Persistent is read-only (makePersistent)");
        return reg.get<T>(e);
    }
    template <class T>
    T* tryGet(Entity e)
    {
        static_assert(T::id != ComponentId(EngineComponent::Parent), "Parent: use remove<Parent> and add<Parent>");
        static_assert(T::id != ComponentId(EngineComponent::Persistent), "Persistent is read-only (makePersistent)");
        return reg.tryGet<T>(e);
    }
    template <class T>
    const T& read(Entity e) const
    {
        return reg.read<T>(e);
    }
    template <class T>
    const T* tryRead(Entity e) const
    {
        return reg.tryRead<T>(e);
    }
    template <class T>
    bool has(Entity e) const
    {
        return reg.has<T>(e);
    }
    // In phases 2-5 only Simulated entities (§12.3); viewAll: all alive entities
    template <class... C>
    View<C...> view()
    {
        static_assert(
            parentReadOnly<C...>(),
            "Parent and Persistent in a view must be const (Parent: remove and add; Persistent: makePersistent)");
        return reg.view<C...>(simulatedOnly(phase) ? Registry::kSimulated : 0);
    }
    template <class... C>
    View<C...> viewAll()
    {
        static_assert(
            parentReadOnly<C...>(),
            "Parent and Persistent in a view must be const (Parent: remove and add; Persistent: makePersistent)");
        return reg.view<C...>(0);
    }
    template <class... C>
    static ComponentMask mask()
    {
        return Registry::mask<C...>();
    }

    EventQueues events;

    // ---- gameplay (§7.8, §9.3, §9.6) ------------------------------------------------------------------------------

    // A one-shot player action (before S15 a local queue): delivered as ActionEvent, readable in the next step, in
    // (peer, sequence) order, exactly once. The player must be alive and have a Player component.
    void queueAction(const PlayerAction& action);
    // Before S16 the world is the host and the only peer
    uint8_t localPeer() const { return 0; }
    bool isHost() const { return true; }
    // The entities a peer owns (R-NET-2); before S16 all entities
    template <class... C>
    View<C...> ownedBy(uint8_t /*peer*/)
    {
        return view<C...>();
    }
    // World-state components (registerGlobal) on the world-state entity, space-state components (registerSpaceState)
    // on the space-state entity of the space (space 0 before S21). A component that is not registered as global or
    // space state asserts.
    template <class T>
    T& global()
    {
        return reg.get<T>(globalEntity<T>());
    }
    template <class T>
    const T& readGlobal() const
    {
        return reg.read<T>(const_cast<World*>(this)->globalEntity<T>());
    }
    template <class T>
    T& spaceState(SpaceId space)
    {
        return reg.get<T>(spaceStateEntityOf<T>(space));
    }
    template <class T>
    const T& readSpaceState(SpaceId space) const
    {
        return reg.read<T>(const_cast<World*>(this)->spaceStateEntityOf<T>(space));
    }
    // The world-state entity (Entity{} if no global component is registered) and the space-state entity of a space
    Entity worldStateEntity() const { return worldState; }
    Entity spaceStateEntity(SpaceId space) const { return space < spaceStates.size() ? spaceStates[space] : Entity{}; }
    // A referable ID that survives saving (§9.6): the existing one, or the next value of the world's counter (the
    // Persistent component is added at once between steps, at the flush during a step)
    PersistentId makePersistent(Entity e);
    // The alive entity with this ID, Entity{} otherwise
    Entity resolve(PersistentId id) const;

    // ---- spatial queries (§9.5) ----------------------------------------------------------------------------------

    // The alive entities of the space within radius of center (Euclidean distance, boundary included) that have all
    // components of mask, sorted by entity index (deterministic). Positions as of the last phase 6: an entity moved
    // since is found at its old position, one created since is not found until then; destroyed entities are never
    // returned. Reserved entities are not found. Allowed in parallelEach (reads only).
    std::vector<Entity> queryRadius(SpaceId space, const Vec3& center, double radius,
                                    const ComponentMask& mask = ComponentMask()) const;
    // The same into out (cleared first; allocates only if out's capacity is smaller than the result)
    void queryRadius(SpaceId space, const Vec3& center, double radius, const ComponentMask& mask,
                     std::vector<Entity>& out) const;

    // ---- time and steps (§8.3, §8.6) -----------------------------------------------------------------------------

    // The world's random stream (PCG32 seeded from WorldSetup::seed; part of the hash and saves). Not in parallelEach.
    Rng& random();
    double dt() const { return stepDt; }
    double time() const { return double(steps) * stepDt; }
    uint64_t stepIndex() const { return steps; }
    // Interpolation factor between the last two steps, in [0, 1) (set by advance)
    double alpha() const { return accumulator / stepDt; }
    // accumulator += frameDt * timeScale; runs whole steps, at most maxStepsPerAdvance; the remaining backlog is
    // dropped (counted in droppedSteps). No steps while paused.
    void advance(double frameDt);
    // Runs exactly n steps (also while paused)
    void step(int n = 1);
    void setPaused(bool p) { paused = p; }
    bool isPaused() const { return paused; }
    void setTimeScale(double s);
    double timeScale() const { return scale; }
    uint64_t droppedSteps() const { return dropped; }
    // Threads of parallelEach including the caller (1: serial); between steps
    void setWorkerThreads(int n);

    // FNV-1a 64 over the saved components of the alive entities (index order), the free list, the RNG state, the
    // pending events and the step index (§17.4). Between steps.
    uint64_t hash() const;

    // ---- engine-internal -----------------------------------------------------------------------------------------

    Registry& registry() { return reg; }
    const Registry& registry() const { return reg; }
    const ClassList& classList() const { return classes; }
    // Step timings and counters (§17.1, S10); the app adds frame and GPU numbers from S13
    Profiler& profiler() { return prof; }
    int objectsUpdatedCounter() const { return updatedCounter; }
    const Profiler& profiler() const { return prof; }
    const WorldConfig& worldConfig() const { return config; }
    // Player cells of the last tick evaluation (space, x, z in TickPolicy::ringSize cells), sorted
    // (gameplay/TickSystem)
    std::vector<std::array<int64_t, 3>> tickPlayerCells;
    StructuralLog& structuralLog() { return log; }
    // Unsorted queryRadius for order-insensitive engine callers (interest sets, ticking): fn(Entity, const Vec3&
    // gridPosition) for each alive entity of the space within radius that has all components of mask, in an
    // unspecified order. gridPosition is the position the decision was made with (the last phase 6), so a caller that
    // needs the distance (hysteresis, nearest player) uses the same state as the query. fn must not change the grid,
    // directly or indirectly (no step(), no rebuildSpatialGrid while the query runs); copy gridPosition to keep it.
    // Nested read-only queries are fine.
    template <class F>
    void forEachWithin(SpaceId space, const Vec3& center, double radius, const ComponentMask& mask, F&& fn) const
    {
        grid.forEachWithin(space, center, radius,
                           [&](const SpatialGrid::Item& it)
                           {
                               // Destroyed since the last update (the slot may be reused: the generation differs)
                               if (!reg.alive(it.entity) || !reg.entityMask(it.entity).containsAll(mask)) return;
                               fn(it.entity, it.position);
                           });
    }
    // The grid of queryRadius (updated by spatialGridSystem in phase 6) and its structural log consumer
    SpatialGrid& spatialGrid() { return grid; }
    const SpatialGrid& spatialGrid() const { return grid; }
    int spatialGridConsumer() const { return gridConsumer; }
    Phase currentPhase() const { return phase; }
    bool inStep() const { return phase != Phase::None; }
    // Entities whose Transform changed in this or the previous step, sorted by index (built in phase 6)
    const std::vector<uint32_t>& changedTransforms() const { return changed; }
    std::vector<uint32_t>& changedTransformsForUpdate() { return changed; }
    // Set when Parent components or entities change: phase 6 rebuilds the depth-sorted child list
    bool hierarchyDirty = true;
    // The entities with a Parent, parents before children (depth, then index); rebuilt by phase 6 when dirty
    std::vector<Entity> hierarchyOrder;
    // Children whose parent was teleported after their last update in a step (phase 8): teleported in the next
    // phase 6, sorted by index
    std::vector<Entity> pendingChildTeleports;
    // ---- type-erased access for the world API (between steps; S10) ----
    // Adds a default component by ID with its log record (not Transform, Behaviour, Persistent or Parent)
    void addDefaultNow(Entity e, ComponentId id);
    // Removes a component by ID with its log record (Parent and Persistent keep their bookkeeping; not Transform)
    void removeByIdNow(Entity e, ComponentId id);
    // Spawns a class by its registration: Transform, Behaviour, then initial(e) adds the initial components, then the
    // tick interval, configure() and onSpawn()
    Entity spawnByClass(const ClassList::ClassInfo& info, SpaceId space, const Vec3& position,
                        const std::function<void(Entity)>& initial);
    // Runs fn after every completed step (also of step(n) and advance); fn returns false to be removed
    int addStepListener(std::function<bool()> fn)
    {
        stepListeners.push_back({nextListenerId, std::move(fn)});
        return nextListenerId++;
    }
    // Removes a step listener (also while the listeners run: it is not called any more)
    void removeStepListener(int id)
    {
        for (size_t i = 0; i < stepListeners.size(); ++i)
            if (stepListeners[i].first == id) stepListeners.erase(stepListeners.begin() + std::ptrdiff_t(i));
        removedStepListeners.push_back(id);
    }
    // Runs fn when the world is destroyed (observers such as the world API detach); returns an ID for removal
    int addDestroyListener(std::function<void()> fn)
    {
        destroyListeners.push_back({nextListenerId, std::move(fn)});
        return nextListenerId++;
    }
    void removeDestroyListener(int id)
    {
        for (size_t i = 0; i < destroyListeners.size(); ++i)
            if (destroyListeners[i].first == id) destroyListeners.erase(destroyListeners.begin() + std::ptrdiff_t(i));
    }
    // While at least one user asked for it: the alive entities whose Transform changed during the last completed step
    // (any phase), collected during the flag shift at the end of the step (the world API's change journal)
    void retainStepChanges() { trackStepChanges++; }
    void releaseStepChanges() { trackStepChanges = std::max(0, trackStepChanges - 1); }
    const std::vector<Entity>& changedDuringLastStep() const { return stepChanged; }

    // Recounts the children per entity from the Parent pool and marks the hierarchy order dirty (after the registry was
    // filled directly, e.g. by load in S11)
    // Returns false (and changes nothing, in every build: the caller can repair the fill) if a Parent names an entity
    // that is not alive or the Parents form a cycle
    bool rebuildHierarchyState();

   private:
    // Structural changes are deferred during a step, except while commands are applied
    bool deferring() const { return phase != Phase::None && !flushing; }
    void record(std::unique_ptr<Command> c);
    void flushCommands();
    void runStep();
    void checkNotParallel(const char* operation) const;

    void activateNow(Entity e);
    // spawn: the entity (reserved during a step), then at the flush Transform and Behaviour, the initial components,
    // and the start (tick interval, configure, onSpawn)
    Entity beginSpawn(SpaceId space);
    void finishSpawn(Entity e, const ClassList::ClassInfo& info, const Vec3& position);
    void startObject(Entity e);
    // A spawn that failed halfway (an exception in configure, onSpawn or an initial add): the entity is destroyed
    // without its onDestroy, so no half-started object ticks
    void abandonSpawn(Entity e);
    template <class T>
    Entity globalEntity()
    {
        SAIGA_ASSERT(reg.isGlobal(T::id), std::string(T::name) + " is not registered as a global (registerGlobal)");
        return worldState;
    }
    template <class T>
    Entity spaceStateEntityOf(SpaceId space)
    {
        SAIGA_ASSERT(reg.isSpaceState(T::id),
                     std::string(T::name) + " is not registered as space state (registerSpaceState)");
        SAIGA_ASSERT(space < spaceStates.size(), "spaceState: the space " + std::to_string(space) + " is not loaded");
        return spaceStates[space];
    }
    void createStateEntities();
    void deliverActions();
    // add<Parent>: false if the child was destroyed because its parent is gone (during a flush); asserts between steps
    // for a dead parent and always for a cycle
    bool acceptParent(Entity child, Entity parent);
    // On an exception in a step: drops the pending commands and releases the entities they would have activated
    void abortStep();
    // Children per entity index (destroy skips the cascade for entities without children)
    uint32_t childCount(Entity e) const;
    void changeChildCount(Entity parent, int delta);
    // Allocates the parent's count slot (before Parent is added, so the add cannot leave a count missing)
    void reserveChildSlot(Entity parent);
    void destroyNow(Entity e);
    void destroyOne(Entity e);
    std::vector<Entity> collectDoomed(Entity e) const;
    int destroyDepth = 0;              // destroyOne is running (nested destroys are queued)
    std::vector<Entity> destroyQueue;  // destroys requested by hooks during a destroy
    // Game objects of destroyed entities and removed Behaviours, released when no hook runs (a hook may destroy its
    // own entity and keep running)
    std::vector<std::unique_ptr<GameObject>> retired;
    int hookDepth = 0;
    void releaseRetired();
    template <class T>
    void addNow(Entity e, T&& value);
    template <class T>
    void removeNow(Entity e);

    template <class... C>
    static constexpr bool parentReadOnly()
    {
        return (((!std::is_same_v<std::remove_cv_t<C>, Parent> && !std::is_same_v<std::remove_cv_t<C>, Persistent>) ||
                 std::is_const_v<C>) &&
                ...);
    }

    template <class T, class... A>
    static T makeComponent(A&&... args)
    {
        if constexpr (std::is_constructible_v<T, A...>)
            return T(std::forward<A>(args)...);
        else
            return T{std::forward<A>(args)...};
    }

    Registry reg;
    ClassList classes;
    Profiler prof;
    // Scope IDs: the step, per phase, per system (in SystemList order), per command flush; counter IDs
    int stepScope = -1;
    std::array<int, size_t(kLastPhase) + 1> phaseScopes{}, flushScopes{};
    std::array<std::vector<int>, size_t(kLastPhase) + 1> systemScopes;
    int entitiesCounter = -1, objectsCounter = -1, commandsCounter = -1, recordsCounter = -1, gridCounter = -1,
        updatedCounter          = -1;
    uint64_t recordsAtStepStart = 0;
    void setupProfiler();
    StructuralLog log;
    SpatialGrid grid;
    int gridConsumer = -1;
    SystemList systems;
    JobPool pool;
    Rng rng;
    std::vector<std::unique_ptr<Command>> commands;
    std::vector<Entity> pendingReserved;  // reserved by create() since the last flush
    std::vector<uint32_t> children;       // number of Parent components naming each entity index
    bool stepEventsBegun = false;
    std::vector<uint32_t> changed;
    // Queued actions with their (peer, sequence) key
    struct QueuedAction
    {
        uint8_t peer;
        uint64_t sequence;
        PlayerAction action;
    };
    std::vector<QueuedAction> actions;
    uint64_t actionSequence = 0;
    Entity worldState;
    std::vector<Entity> spaceStates;  // by SpaceId
    std::vector<std::pair<int, std::function<bool()>>> stepListeners;
    std::vector<int> removedStepListeners;  // removed while the listeners run
    std::vector<std::pair<int, std::function<void()>>> destroyListeners;
    int nextListenerId   = 0;
    int trackStepChanges = 0;
    std::vector<Entity> stepChanged;
    void runStepListeners();
    uint64_t persistentCounter = 0;
    std::vector<std::pair<Entity, PersistentId>> pendingPersistent;  // assigned in this phase, added at the flush
    std::unordered_map<uint64_t, Entity> persistentIndex;

    WorldConfig config;
    double stepDt;
    double accumulator = 0;
    double scale       = 1;
    bool paused        = false;
    bool flushing      = false;
    uint64_t steps     = 0;
    uint64_t dropped   = 0;
    Phase phase        = Phase::None;
};

// ---- template definitions ------------------------------------------------------------------------------------------

template <class T, class... A>
void World::add(Entity e, A&&... args)
{
    static_assert(T::id != ComponentId(EngineComponent::Behaviour), "Behaviour is added by spawn<T>");
    static_assert(T::id != ComponentId(EngineComponent::Persistent), "Persistent is added by makePersistent");
    checkNotParallel("add");
    T value = makeComponent<T>(std::forward<A>(args)...);
    // During a flush an entity (or a parent) reserved later in the phase is not active yet: the add follows its
    // activation
    bool defer = deferring() || (flushing && reg.reserved(e));
    if constexpr (T::id == ComponentId(EngineComponent::Parent))
        defer = defer || (flushing && reg.reserved(value.entity));
    if (defer)
    {
        SAIGA_ASSERT(reg.alive(e) || reg.reserved(e), std::string("add<") + T::name + ">: the entity is not alive");
        // Whether e has T is checked when the command is applied (earlier commands of the phase may add or remove T:
        // remove + add re-parents, §7.4); the flush asserts with the component and the entity named
        if constexpr (T::id == ComponentId(EngineComponent::Parent))
        {
            // The parent must exist now (it may be destroyed before the flush; then the child follows it); refused in
            // every build
            if (!reg.alive(value.entity) && !reg.reserved(value.entity))
            {
                SAIGA_ASSERT(false, "add<Parent>: the parent is not alive");
                return;
            }
        }
        record(makeCommand([e, v = std::move(value)](World& w) mutable { w.addNow<T>(e, std::move(v)); }));
        return;
    }
    addNow<T>(e, std::move(value));
}

template <class T>
void World::remove(Entity e)
{
    checkNotParallel("remove");
    if (deferring() || (flushing && reg.reserved(e)))
    {
        SAIGA_ASSERT(reg.alive(e) || reg.reserved(e), std::string("remove<") + T::name + ">: the entity is not alive");
        record(makeCommand([e](World& w) { w.removeNow<T>(e); }));
        return;
    }
    removeNow<T>(e);
}

template <class T>
void World::addNow(Entity e, T&& value)
{
    // A command on an entity destroyed earlier in the same flush is dropped
    if (flushing && !reg.alive(e)) return;
    // Two deferred adds (hooks see the state before the flush): refused in every build (the pool must not get a
    // second entry)
    if (reg.alive(e) && reg.has<T>(e))
    {
        SAIGA_ASSERT(false, std::string("add<") + T::name + ">: the entity already has " + T::name);
        return;
    }
    if constexpr (T::id == ComponentId(EngineComponent::Parent))
    {
        // A parent that is gone: the child follows it (destroyed); a cycle asserts
        if (!acceptParent(e, value.entity)) return;
    }
    // Everything that can allocate first (the log record, the parent's count slot); after the add nothing can fail
    auto record = log.prepare(1);
    if constexpr (T::id == ComponentId(EngineComponent::Persistent))
    {
        SAIGA_ASSERT(value.persistentId.valid() && !persistentIndex.count(value.persistentId.value),
                     "Persistent: the ID is invalid or already used");
    }
    if constexpr (T::id == ComponentId(EngineComponent::Parent))
    {
        const Entity parent = value.entity;
        reserveChildSlot(parent);
        reg.add<T>(e, std::move(value));
        changeChildCount(parent, +1);
        hierarchyDirty = true;
    }
    else if constexpr (T::id == ComponentId(EngineComponent::Persistent))
    {
        // The index entry first (it allocates), rolled back if the add fails
        const uint64_t id = value.persistentId.value;
        persistentIndex.emplace(id, e);
        try
        {
            reg.add<T>(e, std::move(value));
        }
        catch (...)
        {
            persistentIndex.erase(id);
            throw;
        }
    }
    else
        reg.add<T>(e, std::move(value));
    record.append(e, T::id, true);
}

template <class T>
void World::removeNow(Entity e)
{
    if (flushing && !reg.alive(e)) return;
    // Two deferred removes, or none to remove: refused in every build
    if (!reg.alive(e) || !reg.has<T>(e))
    {
        SAIGA_ASSERT(false, std::string("remove<") + T::name + ">: the entity has no " + T::name);
        return;
    }
    if ((reg.flags(e) & (Registry::kWorldState | Registry::kSpaceState)) &&
        (reg.isGlobal(T::id) || reg.isSpaceState(T::id)))
    {
        // Refused in every build: global<T>() and spaceState<T>() rely on them
        SAIGA_ASSERT(false, std::string("remove<") + T::name + ">: world-state components cannot be removed");
        return;
    }
    auto record = log.prepare(1);
    if constexpr (T::id == ComponentId(EngineComponent::Behaviour))
    {
        // The object may be running (a hook removing its own Behaviour): released when no hook runs
        retired.reserve(retired.size() + 1);
        if (reg.read<T>(e).object) retired.push_back(std::move(reg.get<T>(e).object));
    }
    if constexpr (T::id == ComponentId(EngineComponent::Parent))
    {
        changeChildCount(reg.read<Parent>(e).entity, -1);
        hierarchyDirty = true;
    }
    if constexpr (T::id == ComponentId(EngineComponent::Persistent))
        persistentIndex.erase(reg.read<T>(e).persistentId.value);
    reg.remove<T>(e);
    record.append(e, T::id, false);
    if constexpr (T::id == ComponentId(EngineComponent::Behaviour)) releaseRetired();
}

}  // namespace Engine
}  // namespace Saiga
