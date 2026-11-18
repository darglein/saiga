/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// World: registration, commands, the step loop, advance and the hash.

#include "saiga/engine/world/World.h"

#include "saiga/engine/components/EngineComponents.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/world/Events.h"
#include "saiga/engine/world/TransformSystem.h"

#include <algorithm>
#include <cmath>

namespace Saiga
{
namespace Engine
{
namespace
{
// Resets the phase and the flush flag if a system or a command throws
struct PhaseGuard
{
    Phase& phase;
    bool& flushing;
    ~PhaseGuard()
    {
        phase    = Phase::None;
        flushing = false;
    }
};
}  // namespace

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
    // The entity and its descendants (§7.4), children before parents
    std::vector<Entity> doomed{e};
    if (reg.registered(Parent::id))
    {
        for (size_t i = 0; i < doomed.size(); ++i)
        {
            const Entity parent = doomed[i];
            reg.view<const Parent>().each(
                [&](Entity child, const Parent& p)
                {
                    if (p.entity == parent) doomed.push_back(child);
                });
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

void World::record(std::unique_ptr<Command> c)
{
    commands.push_back(std::move(c));
}

void World::flushCommands()
{
    // Changes made by a command apply immediately (flushing); commands recorded meanwhile run in this flush
    flushing = true;
    for (size_t i = 0; i < commands.size(); ++i)
    {
        std::unique_ptr<Command> c = std::move(commands[i]);
        c->apply(*this);
    }
    commands.clear();
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
    PhaseGuard guard{phase, flushing};
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
        dropped += uint64_t(std::floor(accumulator / stepDt));
        accumulator = std::min(accumulator, 0.999 * stepDt);
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
    SAIGA_ASSERT(!inStep(), "World::hash inside a step");
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
