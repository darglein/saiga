/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/engine/components/EngineIds.h"
#include "saiga/engine/core/JobPool.h"
#include "saiga/engine/core/Rng.h"
#include "saiga/engine/ecs/Registry.h"
#include "saiga/engine/world/Commands.h"
#include "saiga/engine/world/EventQueues.h"
#include "saiga/engine/world/EventRegistry.h"
#include "saiga/engine/world/Phase.h"
#include "saiga/engine/world/StructuralLog.h"
#include "saiga/engine/world/SystemList.h"
#include "saiga/export.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <type_traits>

namespace Saiga
{
namespace Engine
{
// Game policies of a world (R-MECH-1; docs/engine_design.md §7.9). Later stages add their fields.
struct WorldConfig
{
    double stepsPerSecond  = 60;
    int maxStepsPerAdvance = 4;
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
        return reg.get<T>(e);
    }
    template <class T>
    T* tryGet(Entity e)
    {
        static_assert(T::id != ComponentId(EngineComponent::Parent), "Parent: use remove<Parent> and add<Parent>");
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
        return reg.view<C...>(simulatedOnly(phase) ? Registry::kSimulated : 0);
    }
    template <class... C>
    View<C...> viewAll()
    {
        return reg.view<C...>(0);
    }
    template <class... C>
    static ComponentMask mask()
    {
        return Registry::mask<C...>();
    }

    EventQueues events;

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
    StructuralLog& structuralLog() { return log; }
    Phase currentPhase() const { return phase; }
    bool inStep() const { return phase != Phase::None; }
    // Entities whose Transform changed in this or the previous step, sorted by index (built in phase 6)
    const std::vector<uint32_t>& changedTransforms() const { return changed; }
    std::vector<uint32_t>& changedTransformsForUpdate() { return changed; }
    // Set when Parent components or entities change: phase 6 rebuilds the depth-sorted child list
    bool hierarchyDirty = true;
    // The entities with a Parent, parents before children (depth, then index); rebuilt by phase 6 when dirty
    std::vector<Entity> hierarchyOrder;

   private:
    // Structural changes are deferred during a step, except while commands are applied
    bool deferring() const { return phase != Phase::None && !flushing; }
    void record(std::unique_ptr<Command> c);
    void flushCommands();
    void runStep();
    void checkNotParallel(const char* operation) const;

    void activateNow(Entity e);
    // add<Parent>: false if the child was destroyed because its parent is gone (during a flush); asserts between steps
    // for a dead parent and always for a cycle
    bool acceptParent(Entity child, Entity parent);
    // On an exception in a step: drops the pending commands and releases the entities they would have activated
    void abortStep();
    // After phase 8: children of parents teleported in phases 7 and 8 (after the hierarchy update) are teleported too
    void propagateLateTeleports();
    void destroyNow(Entity e);
    template <class T>
    void addNow(Entity e, T&& value);
    template <class T>
    void removeNow(Entity e);

    template <class T, class... A>
    static T makeComponent(A&&... args)
    {
        if constexpr (std::is_constructible_v<T, A...>)
            return T(std::forward<A>(args)...);
        else
            return T{std::forward<A>(args)...};
    }

    Registry reg;
    StructuralLog log;
    SystemList systems;
    JobPool pool;
    Rng rng;
    std::vector<std::unique_ptr<Command>> commands;
    std::vector<Entity> pendingReserved;  // reserved by create() since the last flush
    std::vector<uint32_t> changed;

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
    checkNotParallel("add");
    if (deferring())
    {
        SAIGA_ASSERT(reg.alive(e) || reg.reserved(e), std::string("add<") + T::name + ">: the entity is not alive");
        // Checked when recorded, so the failure names the system's call (an alive entity can still lose T before the
        // flush; then add applies normally)
        if (reg.alive(e)) SAIGA_ASSERT(!reg.has<T>(e), std::string("add<") + T::name + ">: the entity already has it");
        record(makeCommand([e, v = makeComponent<T>(std::forward<A>(args)...)](World& w) mutable
                           { w.addNow<T>(e, std::move(v)); }));
        return;
    }
    addNow<T>(e, makeComponent<T>(std::forward<A>(args)...));
}

template <class T>
void World::remove(Entity e)
{
    checkNotParallel("remove");
    if (deferring())
    {
        SAIGA_ASSERT(reg.alive(e) || reg.reserved(e), std::string("remove<") + T::name + ">: the entity is not alive");
        if (reg.alive(e))
            SAIGA_ASSERT(reg.has<T>(e), std::string("remove<") + T::name + ">: the entity does not have it");
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
    if constexpr (T::id == ComponentId(EngineComponent::Parent))
    {
        // A parent that is gone: the child follows it (destroyed); a cycle asserts
        if (!acceptParent(e, value.entity)) return;
    }
    reg.add<T>(e, std::move(value));
    log.append(e, T::id, true);
    if constexpr (T::id == ComponentId(EngineComponent::Parent)) hierarchyDirty = true;
}

template <class T>
void World::removeNow(Entity e)
{
    if (flushing && !reg.alive(e)) return;
    reg.remove<T>(e);
    log.append(e, T::id, false);
    if constexpr (T::id == ComponentId(EngineComponent::Parent)) hierarchyDirty = true;
}

}  // namespace Engine
}  // namespace Saiga
