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
    SAIGA_ASSERT(std::isfinite(setup.config.stepsPerSecond) && setup.config.stepsPerSecond > 0 &&
                     setup.config.maxStepsPerAdvance >= 1,
                 "invalid WorldConfig");
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
    auto record = log.prepare(1);
    reg.activate(e);
    record.append(e, Transform::id, true);
}

void World::destroyNow(Entity e)
{
    SAIGA_ASSERT(reg.alive(e), "destroy: the entity is not alive");
    // The entity and its descendants (§7.4), children before parents. One pass over the Parent pool builds the
    // (parent index, child) list, sorted for a deterministic order; the visited set stops at cycles (add<Parent>
    // rejects them, this keeps destroy finite regardless).
    std::vector<Entity> doomed{e};
    if (childCount(e) > 0)
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
                if (childCount(parent) == 0) break;
                if (visited[child.index] || reg.read<Parent>(child).entity != parent) continue;
                visited[child.index] = true;
                doomed.push_back(child);
            }
        }
    }
    for (auto it = doomed.rbegin(); it != doomed.rend(); ++it)
    {
        const Entity x = *it;
        // A callback (S10 onDestroy) may have destroyed it already
        if (!reg.alive(x)) continue;
        // Snapshot, reserve the log records, then destroy (the only step that can throw, before any change); the
        // bookkeeping is committed afterwards without allocation
        const ComponentMask m = reg.entityMask(x);
        size_t records        = 0;
        for (size_t id = 0; id < kMaxComponents; ++id) records += m.test(ComponentId(id)) ? 1 : 0;
        auto prepared          = log.prepare(records);
        const Parent* p        = reg.tryRead<Parent>(x);
        const Entity oldParent = p ? p->entity : Entity{};
        reg.destroy(x);
        if (oldParent.valid()) changeChildCount(oldParent, -1);
        if (x.index < children.size()) children[x.index] = 0;
        // One removed record per component in ID order, Transform first
        for (size_t id = 0; id < kMaxComponents; ++id)
            if (m.test(ComponentId(id))) prepared.append(x, ComponentId(id), false);
        if (m.test(Parent::id) || doomed.size() > 1) hierarchyDirty = true;
    }
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
    stepEventsBegun = true;
    try
    {
        for (int p = kFirstPhase; p <= kLastPhase; ++p)
        {
            phase = Phase(p);
            // Teleports made in phase 7 reach the children before extract reads them (§7.4)
            if (phase == Phase::Extract) propagateTeleportsBeforeExtract(*this);
            for (const SystemList::System& s : systems.systems(phase))
            {
                // Before S16 the world is the host: client-only systems do not run
                if (s.runOn == RunOn::Client) continue;
                s.fn(*this, stepDt);
            }
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
        throw;
    }
    phase           = Phase::None;
    stepEventsBegun = false;
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
    h.addU64(steps);
    return h.value();
}

}  // namespace Engine
}  // namespace Saiga
