/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// ScenarioWorld: the world of a scenario with its helpers, the step observation and the save/load swap.

#include "saiga/engine/tools/ScenarioWorld.h"

#include "saiga/engine/core/Log.h"
#include "saiga/engine/save/SaveDiff.h"

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
    w->setTickEveryStep(true);
    attach();
}

ScenarioWorld::~ScenarioWorld()
{
    // The API detaches from the world before the world goes away
    apiPtr.reset();
    if (w && stepListener >= 0) w->removeStepListener(stepListener);
}

void ScenarioWorld::attach()
{
    stepListener = w->addStepListener(
        [this]()
        {
            onStep();
            return true;
        });
}

void ScenarioWorld::onStep()
{
    // Never throws (a throwing listener would be removed; the helpers end the scenario)
    observedSteps++;
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
}

void ScenarioWorld::checkAbort() const
{
    if (!abortReason.empty()) throw ScenarioAbort{abortReason};
}

WorldApi& ScenarioWorld::api()
{
    if (!apiPtr) apiPtr = std::make_unique<WorldApi>(*w);
    return *apiPtr;
}

ApiResult ScenarioWorld::call(const std::string& operation, const json11::Json& arguments)
{
    checkAbort();
    ApiResult r = api().call(operation, arguments);
    checkAbort();
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
    for (int i = 0; i < n; ++i)
    {
        checkAbort();
        if (!control.saveLoadDone && observedSteps >= control.saveLoadAt) saveLoad();
        w->step();
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
    control.saveLoadDone             = true;
    const std::vector<uint8_t> saved = w->save();
    auto fresh                       = std::make_unique<World>(setup);
    const LoadResult r               = fresh->tryLoad(saved);
    if (!r.committed)
    {
        failed.push_back("step " + std::to_string(observedSteps) + ": save/load: the load was rejected: " + r.error);
        throw ScenarioAbort{"the save/load round trip failed"};
    }
    for (const LoadResult::HookError& e : r.hookErrors)
        failed.push_back("step " + std::to_string(observedSteps) + ": save/load: the game object of entity " +
                         std::to_string(e.entity.index) + ":" + std::to_string(e.entity.generation) + " (" +
                         e.className + ") failed: " + e.message);
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
    // What is not world state carries over
    fresh->setPaused(w->isPaused());
    fresh->setTimeScale(w->timeScale());
    fresh->setTickEveryStep(tickEverything);
    w->removeStepListener(stepListener);
    if (apiPtr) apiPtr->rebind(*fresh);
    std::swap(w, fresh);
    attach();
    fresh.reset();  // the old world
}

std::string ScenarioWorld::report() const
{
    std::string out;
    for (const std::string& f : failed) out += "  failed: " + f + "\n";
    if (!abortReason.empty()) out += "  aborted: " + abortReason + "\n";
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
