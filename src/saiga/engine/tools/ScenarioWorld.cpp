/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// ScenarioWorld: the world of a scenario with its helpers, the step observation and the save/load swap.

#include "saiga/engine/tools/ScenarioWorld.h"

#include "saiga/engine/core/Log.h"
#include "saiga/engine/save/SaveDiff.h"

#include <algorithm>
#include <cmath>

namespace Saiga
{
namespace Engine
{
namespace
{
WorldSetup scenarioSetup(WorldSetup s)
{
    // No clock calls: a scenario is reproducible (R-DET-1)
    s.profiling = false;
    return s;
}
}  // namespace

ScenarioWorld::ScenarioWorld(const WorldSetup& s, const ScenarioOptions& o)
    : setup(scenarioSetup(s)), options(o), w(std::make_unique<World>(setup))
{
    logStart = engineLog().lastSequence();
    // A replacement world shares the data (also after reload_data)
    setup.sharedData = w->sharedDataStore();
    w->setTickEveryStep(true);
    attach();
}

ScenarioWorld::~ScenarioWorld()
{
    // The API detaches from the world before the world goes away
    apiPtr.reset();
    if (w && stepListener >= 0) w->removeStepListener(stepListener);
    if (w && countListener >= 0) w->removeStepListener(countListener);
    for (int id : ownListeners)
        if (w) w->removeStepListener(id);
}

void ScenarioWorld::attachCounter()
{
    countListener = w->addStepListener(
        [this]()
        {
            observedSteps++;
            return true;
        });
}

void ScenarioWorld::moveObserverLast()
{
    // Not while the listeners run: a listener added now runs from the next step on, and the observer must not miss
    // this one
    if (w->dispatchingListeners()) return;
    w->removeStepListener(stepListener);
    attachObserver();
}

void ScenarioWorld::attachObserver()
{
    stepListener = w->addStepListener(
        [this]()
        {
            try
            {
                onStep();
            }
            catch (...)
            {
                // The world removes a throwing listener: the observation continues in a new registration (adding
                // listeners while they run is allowed), so later steps are still counted and still end the scenario
                attachObserver();
                throw;
            }
            return true;
        });
}

void ScenarioWorld::attach()
{
    attachCounter();
    ownListeners.clear();
    for (auto& fn : listeners)
        ownListeners.push_back(w->addStepListener(
            [fn]()
            {
                (*fn)();
                return true;
            }));
    attachObserver();
}

void ScenarioWorld::addStepListener(std::function<void()> fn)
{
    auto shared = std::make_shared<std::function<void()>>(std::move(fn));
    listeners.push_back(shared);
    ownListeners.push_back(w->addStepListener(
        [shared]()
        {
            (*shared)();
            return true;
        }));
    moveObserverLast();
}

void ScenarioWorld::onStep()
{
    // After the scenario's listeners (the counter already counted this step). Throws ScenarioAbort past maxSteps (the
    // observer registers itself again, see attachObserver)
    if (control.hashes) control.hashes->push_back(w->hash());
    for (Trace& t : traces)
    {
        t.values.push_back({observedSteps, t.read(*w)});
        if (t.values.size() > 20) t.values.pop_front();
    }
    if (observedSteps == control.stopAt)
    {
        if (control.stopSave) *control.stopSave = w->save();
        if (abortReason.empty()) abortReason = "stopped by the runner at step " + std::to_string(observedSteps);
    }
    if (observedSteps > options.maxSteps && abortReason.empty())
        abortReason = "more than " + std::to_string(options.maxSteps) + " steps";
    if (options.deadline && abortReason.empty() && std::chrono::steady_clock::now() >= *options.deadline)
        abortReason = "stopped: tool time budget";
    // Ends the scenario also when it steps outside the helpers (world().step(), advance()): the exception leaves the
    // world's step; the world API turns it into an error result, after which call() ends the scenario
    if (!abortReason.empty()) throw ScenarioAbort{abortReason};
}

void ScenarioWorld::checkAbort() const
{
    if (!abortReason.empty()) throw ScenarioAbort{abortReason};
}

WorldApi& ScenarioWorld::api()
{
    if (!apiPtr)
    {
        apiPtr = std::make_unique<WorldApi>(*w);
        // No wall-clock limit on step and apply (a scenario must not depend on the machine's speed); maxSteps bounds
        // the run
        apiPtr->setTimeBudget(1e7);
    }
    return *apiPtr;
}

ApiResult ScenarioWorld::call(const std::string& operation, const json11::Json& arguments)
{
    checkAbort();
    ApiResult r = api().call(operation, arguments);
    checkAbort();
    // The API's step counts like a step helper for the save/load variant's swap (after the call, between steps)
    if (!control.saveLoadDone && observedSteps >= control.saveLoadAt && stepDepth == 0 && !w->dispatchingListeners())
        saveLoad();
    return r;
}

void ScenarioWorld::involve(Entity e)
{
    if (involved.size() < 10 && std::find(involved.begin(), involved.end(), e) == involved.end()) involved.push_back(e);
}

Entity ScenarioWorld::spawnPlayer(const Vec3& position, uint8_t peer)
{
    const Entity e = w->create();
    w->get<Transform>(e).setPosition(position);
    Player p;
    p.peer        = peer;
    p.characterId = nextCharacterId++;
    w->add<Player>(e, p);
    w->add<PlayerInput>(e);
    w->add<CharacterIntent>(e);
    involve(e);
    return e;
}

void ScenarioWorld::move(Entity player, const Vec3& direction)
{
    if (!check(w->alive(player) && w->has<PlayerInput>(player), "move: the player is not alive or has no PlayerInput"))
        return;
    const double n                   = direction.norm();
    w->get<PlayerInput>(player).move = n > 1 ? Vec3(direction / n) : direction;
}

void ScenarioWorld::input(Entity player, uint32_t action, const Vec3& aim)
{
    if (!check(w->alive(player) && w->has<Player>(player), "input: the player is not alive")) return;
    w->queueAction(PlayerAction{player, action, aim});
}

void ScenarioWorld::damage(Entity target, double amount, Entity source)
{
    w->events.emit(DamageEvent{target, source, amount});
}

void ScenarioWorld::enableDistanceTicking()
{
    tickEverything = false;
    w->setTickEveryStep(false);
}

void ScenarioWorld::step(int n)
{
    if (w->dispatchingListeners())
    {
        failed.push_back("step " + std::to_string(observedSteps) +
                         ": a step helper was called from a step listener (not supported: step in the scenario "
                         "function)");
        abortReason = "stepped inside a step listener";
        throw ScenarioAbort{abortReason};
    }
    for (int i = 0; i < n; ++i)
    {
        checkAbort();
        // Not inside a step listener (a listener that steps): the old world must not be replaced while it runs them
        if (!control.saveLoadDone && observedSteps >= control.saveLoadAt && stepDepth == 0) saveLoad();
        stepDepth++;
        try
        {
            w->step();
        }
        catch (...)
        {
            stepDepth--;
            throw;
        }
        stepDepth--;
    }
    checkAbort();
}

void ScenarioWorld::stepSeconds(double seconds)
{
    step(int(std::llround(seconds / w->dt())));
}

bool ScenarioWorld::stepUntil(const std::function<bool()>& predicate, int maxSteps)
{
    for (int i = 0; i < maxSteps; ++i)
    {
        if (predicate()) return true;
        step();
    }
    return predicate();
}

bool ScenarioWorld::check(bool condition, const std::string& message)
{
    if (!condition) failed.push_back("step " + std::to_string(observedSteps) + ": " + message);
    return condition;
}

double ScenarioWorld::distance(Entity a, Entity b)
{
    if (!check(w->alive(a) && w->alive(b), "distance: an entity is not alive"))
        return std::numeric_limits<double>::infinity();
    return (w->read<Transform>(a).position() - w->read<Transform>(b).position()).norm();
}

void ScenarioWorld::saveLoad()
{
    // The old world must not be replaced while it runs its listeners (the rest of them would see the old world)
    if (w->dispatchingListeners() || w->inStep())
    {
        failed.push_back("step " + std::to_string(observedSteps) + ": saveLoad was called from " +
                         (w->inStep() ? "a system or game object during a step" : "a step listener") +
                         " (not supported: call it in the scenario function)");
        abortReason = "save/load inside a step";
        throw ScenarioAbort{abortReason};
    }
    control.saveLoadDone             = true;
    control.swappedAtStep            = observedSteps;
    const std::vector<uint8_t> saved = w->save();
    auto fresh                       = std::make_unique<World>(setup);
    // Before the load: configure() and onLoad() of the loaded game objects see the scenario's tick setting
    fresh->setTickEveryStep(w->tickEveryStep());
    const LoadResult r = fresh->tryLoad(saved);
    if (!r.committed)
    {
        failed.push_back("step " + std::to_string(observedSteps) + ": save/load: the load was rejected: " + r.error);
        throw ScenarioAbort{"the save/load round trip failed"};
    }
    for (const LoadResult::HookError& e : r.hookErrors)
        failed.push_back("step " + std::to_string(observedSteps) + ": save/load: the game object of entity " +
                         std::to_string(e.entity.index) + ":" + std::to_string(e.entity.generation) + " (" +
                         e.className + ") failed: " + e.message);
    for (const std::string& e : r.listenerErrors)
        failed.push_back("step " + std::to_string(observedSteps) + ": save/load: a load listener failed: " + e);
    const std::vector<uint8_t> again = fresh->save();
    if (again != saved)
    {
        DiffFilter f;
        f.limit          = 20;
        const SaveDiff d = diffSaves(saved, again, f);
        std::string text;
        for (const SaveDifference& x : d.differences) text += "\n  " + x.describe();
        failed.push_back("step " + std::to_string(observedSteps) +
                         ": save/load: the loaded world saves differently (state outside the saved components, "
                         "R-STATE-1):" +
                         text);
    }
    // Step listeners the runner does not know would be lost with the old world: fail instead of losing them silently
    // (by ID: a scenario listener the world removed after it threw must not hide a direct one)
    bool unknown = false;
    for (int id : w->stepListenerIds())
        unknown = unknown || (id != stepListener && id != countListener &&
                              std::find(ownListeners.begin(), ownListeners.end(), id) == ownListeners.end() &&
                              !(apiPtr && id == apiPtr->stepListenerId()));
    if (unknown)
        failed.push_back("step " + std::to_string(observedSteps) +
                         ": save/load: the world has step listeners that are lost with it (add them with "
                         "ScenarioWorld::addStepListener)");
    // What is not world state carries over
    fresh->setPaused(w->isPaused());
    fresh->setTimeScale(w->timeScale());
    fresh->setAccumulatedTime(w->accumulatedTime());
    fresh->setWorkerThreads(w->workerThreads());
    w->removeStepListener(stepListener);
    w->removeStepListener(countListener);
    for (int id : ownListeners) w->removeStepListener(id);
    if (apiPtr) apiPtr->rebind(*fresh);
    std::swap(w, fresh);
    attach();
    retiredWorlds.push_back(std::move(fresh));  // the old world lives until the scenario ends
}

std::string ScenarioWorld::report() const
{
    // The context of the failures (the failures themselves are listed by the runner)
    std::string out;
    const std::vector<LogRing::Line> lines = engineLog().since(logStart);
    if (!lines.empty())
    {
        out += "  log:\n";
        const size_t first = lines.size() > 200 ? lines.size() - 200 : 0;
        for (size_t i = first; i < lines.size(); ++i)
            out += "    [" + lines[i].category + "] " + lines[i].message + "\n";
    }
    for (Entity e : involved)
        out += "  entity " + std::to_string(e.index) + ":" + std::to_string(e.generation) + ": " +
               (w->alive(e) ? w->dumpJson(e) : std::string("not alive")) + "\n";
    for (const Trace& t : traces)
    {
        out += "  trace " + std::string(t.component) + " of " + std::to_string(t.entity.index) + ":\n";
        for (const auto& v : t.values) out += "    step " + std::to_string(v.first) + ": " + v.second.dump() + "\n";
    }
    return out;
}

}  // namespace Engine
}  // namespace Saiga
