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
#include "saiga/engine/world/Events.h"
#include "saiga/engine/world/TransformSystem.h"

#include <algorithm>
#include <cmath>

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
    SAIGA_ASSERT(setup.config.stepsPerSecond > 0 && setup.config.maxStepsPerAdvance >= 1, "invalid WorldConfig");
    reg.setJobPool(&pool);

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
}

World::~World() = default;

// ---- structural changes
// ----------------------------------------------------------------------------------------------

Entity World::create()
{
    checkNotParallel("create");
    if (deferring())
    {
        // Reserved now (views skip it), alive when the command is applied; reserving never moves a Transform
        Entity e = reg.reserve();
        pendingReserved.push_back(e);
        record(makeCommand([e](World& w) { w.activateNow(e); }));
        return e;
    }
    Entity e = reg.create();
    log.append(e, Transform::id, true);
    return e;
}

void World::destroy(Entity e)
{
    checkNotParallel("destroy");
    if (deferring())
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
    reg.activate(e);
    log.append(e, Transform::id, true);
}

void World::destroyNow(Entity e)
{
    SAIGA_ASSERT(reg.alive(e), "destroy: the entity is not alive");
    // The entity and its descendants (§7.4), children before parents. One pass over the Parent pool builds the
    // (parent index, child) list, sorted for a deterministic order; the visited set stops at cycles (add<Parent>
    // rejects them, this keeps destroy finite regardless).
    std::vector<Entity> doomed{e};
    if (reg.registered(Parent::id) && reg.view<const Parent>().count() > 0)
    {
        std::vector<std::pair<uint32_t, Entity>> links;
        reg.view<const Parent>().each([&](Entity child, const Parent& p) { links.push_back({p.entity.index, child}); });
        std::sort(links.begin(), links.end(),
                  [](const std::pair<uint32_t, Entity>& a, const std::pair<uint32_t, Entity>& b)
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
                if (visited[child.index] || reg.read<Parent>(child).entity != parent) continue;
                visited[child.index] = true;
                doomed.push_back(child);
            }
        }
    }
    for (auto it = doomed.rbegin(); it != doomed.rend(); ++it)
    {
        const Entity x = *it;
        // One removed record per component in ID order, Transform first
        const ComponentMask& m = reg.entityMask(x);
        for (size_t id = 0; id < kMaxComponents; ++id)
            if (m.test(ComponentId(id))) log.append(x, ComponentId(id), false);
        if (m.test(Parent::id) || doomed.size() > 1) hierarchyDirty = true;
        reg.destroy(x);
    }
}

bool World::acceptParent(Entity child, Entity parent)
{
    if (!reg.alive(parent))
    {
        // During a flush the parent may have been destroyed earlier in the same flush: the child follows it
        SAIGA_ASSERT(flushing, "add<Parent>: the parent is not alive");
        destroyNow(child);
        return false;
    }
    // No cycles: the child must not be an ancestor of the parent (or the parent itself)
    Entity a  = parent;
    int depth = 0;
    while (true)
    {
        SAIGA_ASSERT(a != child, "add<Parent>: the parent is the child or one of its descendants (a cycle)");
        if (a == child) return false;  // without asserts: refuse the cycle
        const Parent* p = reg.tryRead<Parent>(a);
        if (!p) break;
        a = p->entity;
        if (++depth > 100000) break;  // defensive: an existing cycle (cannot happen through add<Parent>)
    }
    return true;
}

void World::abortStep()
{
    // The commands of the interrupted phase are dropped; the entities they would have activated are released
    commands.clear();
    for (Entity e : pendingReserved)
        if (reg.reserved(e)) reg.release(e);
    pendingReserved.clear();
    phase    = Phase::None;
    flushing = false;
}

void World::propagateLateTeleports()
{
    // Phase 6 propagated teleports made up to then; a parent teleported in phases 7 or 8 (streaming, replication) still
    // has its flag here: its children are teleported now, so they are not interpolated across the jump
    if (hierarchyDirty) return;  // structural changes after phase 6: the next phase 6 handles them
    for (Entity child : hierarchyOrder)
    {
        if (!reg.alive(child)) continue;
        const Entity parentEntity = reg.read<Parent>(child).entity;
        if (!reg.alive(parentEntity)) continue;
        const Transform& parent = reg.read<Transform>(parentEntity);
        Transform& t            = reg.get<Transform>(child);
        if (!parent.wasTeleported() || t.wasTeleported()) continue;
        LocalTransform identity;
        const LocalTransform* local = reg.tryRead<LocalTransform>(child);
        if (!local) local = &identity;
        t.teleport(parent.position() + parent.rotation() * parent.scale().cwiseProduct(local->position),
                   parent.rotation() * local->rotation);
        t.setScale(parent.scale().cwiseProduct(local->scale));
    }
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
    flushing = false;
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
    // Events emitted in the last step or between steps become readable
    events.beginStep();
    try
    {
        for (int p = kFirstPhase; p <= kLastPhase; ++p)
        {
            phase = Phase(p);
            for (const SystemList::System& s : systems.systems(phase))
            {
                // Before S16 the world is the host: client-only systems do not run
                if (s.runOn == RunOn::Client) continue;
                s.fn(*this, stepDt);
            }
            flushCommands();
        }
        propagateLateTeleports();
    }
    catch (...)
    {
        // The world stays usable: the step is incomplete (the changes of the finished phases remain), the pending
        // commands are dropped and their reserved entities released; the step index does not advance
        abortStep();
        throw;
    }
    phase = Phase::None;
    reg.shiftTransformFlags();
    log.truncate();
    steps++;
}

void World::step(int n)
{
    SAIGA_ASSERT(!inStep(), "World::step inside a step");
    for (int i = 0; i < n; ++i) runStep();
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
        runStep();
        accumulator -= stepDt;
        n++;
    }
    if (accumulator >= stepDt)
    {
        // The backlog beyond maxStepsPerAdvance is dropped (a slow frame does not cause a spiral of catch-up steps)
        // The whole steps that are discarded (0.999 dt of the backlog is kept)
        const double kept = 0.999 * stepDt;
        dropped += uint64_t(std::floor((accumulator - kept) / stepDt));
        accumulator = kept;
    }
}

void World::setTimeScale(double s)
{
    SAIGA_ASSERT(s >= 0, "World::setTimeScale: negative scale");
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
    h.addU64(steps);
    return h.value();
}

}  // namespace Engine
}  // namespace Saiga
