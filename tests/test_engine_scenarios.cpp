/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Engine scenarios with test-only game objects, run in the plain, repeat and save/load variants, and the tests of the
// scenario runner itself (docs/engine_design.md §17.2, §17.4; stage S12)

#include "saiga/engine/components/CharacterIntent.h"
#include "saiga/engine/components/Health.h"
#include "saiga/engine/components/LocalTransform.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/PlayerInput.h"
#include "saiga/engine/components/RenderPose.h"
#include "saiga/engine/components/SpaceLink.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/gameplay/GameObject.h"
#include "saiga/engine/gameplay/Systems.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/terrain/TerrainBrush.h"
#include "saiga/engine/terrain/TerrainEditStore.h"
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/engine/tools/ScenarioRunner.h"
#include "saiga/engine/world/World.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>

#include "terrain_test_hooks.h"

using namespace Saiga;
using namespace Saiga::Engine;
using Json = json11::Json;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);

// ---- test-only components, classes and systems
// ------------------------------------------------------------------------

enum class TestComponent : uint8_t
{
    WandererState = kFirstGameComponent,
    ActionLog,
    TickCount,
    Marker,
    DestroyLog,
    Ref,
    WorkerFlag,
    TransitionLog,
    TerraformerState,
    Count
};
enum class TestClass : uint16_t
{
    Wanderer = kFirstGameClass,
    Counter,
    Recorder,
    Lazy,
    Diver,
    Terraformer,
    Count
};

// The state of a Wanderer (R-STATE-1): who hurt it and how fast it flees
struct WandererState
{
    static constexpr const char* name = "WandererState";
    static constexpr ComponentId id   = ComponentId(TestComponent::WandererState);
    static constexpr uint32_t version = 1;
    Entity threat;
    double fleeSpeed = 8;
    template <class V>
    void visit(V& v)
    {
        v("threat", threat);
        v("fleeSpeed", fleeSpeed);
    }
};

// The actions the test system saw (a global: system state lives in components too)
struct ActionLog
{
    static constexpr const char* name = "ActionLog";
    static constexpr ComponentId id   = ComponentId(TestComponent::ActionLog);
    static constexpr uint32_t version = 1;
    std::vector<uint32_t> actions;
    template <class V>
    void visit(V& v)
    {
        v("actions", actions);
    }
};

struct TickCount
{
    static constexpr const char* name = "TickCount";
    static constexpr ComponentId id   = ComponentId(TestComponent::TickCount);
    static constexpr uint32_t version = 1;
    int32_t updates                   = 0;
    template <class V>
    void visit(V& v)
    {
        v("updates", updates);
    }
};

struct Marker
{
    static constexpr const char* name = "Marker";
    static constexpr ComponentId id   = ComponentId(TestComponent::Marker);
    static constexpr uint32_t version = 1;
    int32_t value                     = 0;
    template <class V>
    void visit(V& v)
    {
        v("value", value);
    }
};

struct DestroyLog
{
    static constexpr const char* name = "DestroyLog";
    static constexpr ComponentId id   = ComponentId(TestComponent::DestroyLog);
    static constexpr uint32_t version = 1;
    std::vector<int32_t> order;
    template <class V>
    void visit(V& v)
    {
        v("order", order);
    }
};

struct Ref
{
    static constexpr const char* name = "Ref";
    static constexpr ComponentId id   = ComponentId(TestComponent::Ref);
    static constexpr uint32_t version = 1;
    PersistentId target;
    template <class V>
    void visit(V& v)
    {
        v("target", target);
    }
};

struct WorkerFlag
{
    static constexpr const char* name = "WorkerFlag";
    static constexpr ComponentId id   = ComponentId(TestComponent::WorkerFlag);
    static constexpr uint32_t version = 1;
    bool onWorker                     = false;
    template <class V>
    void visit(V& v)
    {
        v("onWorker", onWorker);
    }
};

// Flees from whoever damaged it (§17.2): the engine's character mover moves it
class Wanderer final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(TestClass::Wanderer);
    static constexpr const char* name = "Wanderer";
    void onSpawn() override
    {
        if (!has<WandererState>()) add<WandererState>();
        if (!has<Health>()) add<Health>();
        if (!has<CharacterIntent>()) add<CharacterIntent>();
    }
    void onDamage(const DamageEvent& e) override
    {
        write<Health>().hp -= e.amount;
        write<WandererState>().threat = e.source;
    }
    void update(double) override
    {
        const WandererState& s = read<WandererState>();
        Vec3 velocity          = Vec3::Zero();
        if (world().alive(s.threat))
        {
            Vec3 away = transform().position() - world().read<Transform>(s.threat).position();
            away.y()  = 0;
            if (away.norm() > 1e-9) velocity = away.normalized() * s.fleeSpeed;
        }
        write<CharacterIntent>().desiredVelocity = velocity;
    }
};

// Counts its updates in a component
class Counter final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(TestClass::Counter);
    static constexpr const char* name = "Counter";
    void onSpawn() override
    {
        if (!has<TickCount>()) add<TickCount>();
    }
    void update(double) override { write<TickCount>().updates++; }
};

// Records its Marker in the global DestroyLog when destroyed
class Recorder final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(TestClass::Recorder);
    static constexpr const char* name = "Recorder";
    void onDestroy() override { world().global<DestroyLog>().order.push_back(read<Marker>().value); }
};

// A class with its own (very slow) ticking policy: ScenarioWorld still ticks it every step
bool g_lastConfigureTick    = false;  // the world's tick setting when a Lazy was last configured
bool g_throwingLoadListener = false;  // Lazy::onLoad adds a load listener that throws

class Lazy final : public GameObject
{
   public:
    void configure() override { g_lastConfigureTick = world().tickEveryStep(); }
    void onLoad() override
    {
        if (g_throwingLoadListener)
            world().addLoadListener([](const LoadInfo&) { throw std::runtime_error("listener boom"); });
    }
    static constexpr ClassId id       = ClassId(TestClass::Lazy);
    static constexpr const char* name = "Lazy";
    void onSpawn() override
    {
        if (!has<TickCount>()) add<TickCount>();
    }
    void update(double) override { write<TickCount>().updates++; }
    const TickPolicy* tickPolicy() const override
    {
        static const TickPolicy slow{32, -1, -1, 100, 100};
        return &slow;
    }
};

void actionLogSystem(World& w, double)
{
    for (const ActionEvent& a : w.events.read<ActionEvent>()) w.global<ActionLog>().actions.push_back(a.action);
}

std::thread::id g_mainThread;
// Writes whether the world has worker threads: the result depends on the thread count (caught by varyThreads).
// Regression (CodeRabbit, S12): the flag came from the thread that ran a chunk, so the test waited for a worker with a
// timeout and depended on the scheduling; a serial pass over the configured count is deterministic.
void workerFlagSystem(World& w, double)
{
    const bool workers = w.workerThreads() > 1;
    w.view<WorkerFlag>().each([workers](Entity, WorkerFlag& f) { f.onWorker = workers; });
}

WorldSetup scenarioSetup()
{
    WorldSetup s;
    s.components = [](Registry& r)
    {
        r.registerComponent<WandererState>();
        r.registerGlobal<ActionLog>();
        r.registerComponent<TickCount>();
        r.registerComponent<Marker>();
        r.registerGlobal<DestroyLog>();
        r.registerComponent<Ref>();
        r.registerComponent<WorkerFlag>();
    };
    s.classes = [](ClassList& c)
    {
        c.add<Wanderer>();
        c.add<Counter>();
        c.add<Recorder>();
        c.add<Lazy>();
    };
    s.systems = [](SystemList& l)
    {
        l.add(Phase::Gameplay, "actionLog", actionLogSystem);
        l.add(Phase::Gameplay, "workerFlag", workerFlagSystem);
    };
    return s;
}

// The setup of the terrain scenarios (S20): scenarioSetup with a 16-chunk space of slopeRidgeHook at 2 m
WorldSetup terrainScenarioSetup()
{
    WorldSetup s = scenarioSetup();
    s.config.spaceTemplates.push_back(TerrainTestHooks::testSpaceTemplate(16, 2, TerrainTestHooks::slopeRidgeHook, 0));
    return s;
}

// The scenarios that need their own setup
WorldSetup setupFor(const std::string& scenario)
{
    return scenario == "walker_follows_terrain" ? terrainScenarioSetup() : scenarioSetup();
}

// ---- the engine scenarios (one per engine gameplay mechanism, R-TEST-1)
// ----------------------------------------------

void registerEngineScenarios(ScenarioList& s)
{
    s.add("wanderer_flees_when_damaged",
          [](ScenarioWorld& w)
          {
              auto wanderer = w.spawn<Wanderer>({0, 0, 0});
              auto player   = w.spawnPlayer({5, 0, 0});
              w.stepSeconds(1.0);
              w.check(w.distance(wanderer, player) < 6, "stays while not threatened");
              w.damage(wanderer, 10.0, player);  // emits DamageEvent: the real path
              w.trace<CharacterIntent>(wanderer);
              w.check(w.stepUntil([&] { return w.distance(wanderer, player) > 20.0; }, 600), "flees from the attacker");
              w.check(w.world().read<Health>(wanderer).hp == Health().hp - 10, "took the damage");
          });
    s.add("player_walks_with_input",
          [](ScenarioWorld& w)
          {
              auto player = w.spawnPlayer({0, 0, 0});
              w.move(player, Vec3(2, 0, 0));  // shortened to length 1
              w.stepSeconds(2.0);
              const double speed = w.world().read<Player>(player).moveSpeed;
              const double x     = w.world().read<Transform>(player).position().x();
              // Host + client (S16): the client's player arrives, walks on the client and its pose reaches the host a
              // few steps later (at most 2 * delay + async offset + 3 steps behind; extrapolated up to 6 steps)
              const double lag = w.networked() ? 12 * speed / 60 : 0;
              w.check(x > 2 * speed - 0.2 - lag && x < 2 * speed + 0.2,
                      "walked 2 s at moveSpeed: x = " + std::to_string(x));
              w.move(player, Vec3::Zero());
              auto position = [&] { return w.world().read<Transform>(player).position().x(); };
              // Stops: the position stays the same over a few steps (the stop reaches the host with the latency)
              double last = position();
              int still   = 0;
              w.check(w.stepUntil(
                          [&]
                          {
                              const double now = position();
                              still            = now == last ? still + 1 : 0;
                              last             = now;
                              return still >= 5;
                          },
                          60),
                      "stops");
              // At most the step that was already under way (plus the latency on the host)
              w.check(std::abs(position() - x) <= speed / 60 + 1e-9 + 2 * lag, "stops where it was");
          });
    s.add("action_reaches_listener",
          [](ScenarioWorld& w)
          {
              auto player = w.spawnPlayer({0, 0, 0});
              w.input(player, 7);
              w.input(player, 8);
              // Single world: delivered in the next step; host + client (S16): the PlayerAction message arrives
              // 1 + delay steps later on the host, then the next step delivers it
              const auto& log = [&]() -> const std::vector<uint32_t>&
              { return w.world().readGlobal<ActionLog>().actions; };
              w.check(w.stepUntil([&] { return !log().empty(); }, 20), "delivered");
              w.check(log() == std::vector<uint32_t>{7, 8}, "delivered together in queue order");
              if (!w.networked()) w.check(w.steps() == 1, "in the next step");
              w.step(3);
              w.check(w.world().readGlobal<ActionLog>().actions.size() == 2, "exactly once");
          });
    s.add("far_objects_tick_slower",
          [](ScenarioWorld& w)
          {
              w.enableDistanceTicking();
              w.spawnPlayer({0, 0, 0});
              auto near = w.spawn<Counter>({5, 0, 5});
              auto mid  = w.spawn<Counter>({100, 0, 0});   // ring 3: every midInterval
              auto far  = w.spawn<Counter>({1000, 0, 0});  // ring 31: every farInterval
              w.stepSeconds(2.0);
              const TickPolicy& p = w.world().worldConfig().ticking;
              auto updates        = [&](Entity e) { return w.world().read<TickCount>(e).updates; };
              w.check(updates(near) == 120, "near: every step (" + std::to_string(updates(near)) + ")");
              w.check(std::abs(updates(mid) - 2.0 / p.midInterval) <= 1,
                      "mid: every midInterval (" + std::to_string(updates(mid)) + ")");
              w.check(std::abs(updates(far) - 2.0 / p.farInterval) <= 1,
                      "far: every farInterval (" + std::to_string(updates(far)) + ")");
          });
    s.add("objects_tick_every_step_by_default",
          [](ScenarioWorld& w)
          {
              // No player and a class with its own slow policy: ScenarioWorld ticks every object every step
              auto lazy    = w.spawn<Lazy>({1000, 0, 0});
              auto counter = w.spawn<Counter>({-5000, 0, 0});
              w.step(30);
              w.check(w.world().read<TickCount>(lazy).updates == 30, "Lazy ticks every step");
              w.check(w.world().read<TickCount>(counter).updates == 30, "Counter ticks every step");
          });
    s.add("destroy_runs_on_destroy_children_first",
          [](ScenarioWorld& w)
          {
              auto parent = w.spawn<Recorder>({0, 0, 0}, Marker{1});
              auto child  = w.spawn<Recorder>({0, 0, 0}, Marker{2});
              w.world().add<Parent>(child, Parent{parent});
              w.step(5);
              w.world().destroy(parent);
              w.step();
              w.check(w.world().readGlobal<DestroyLog>().order == std::vector<int32_t>{2, 1}, "children first");
              w.check(!w.world().alive(child), "the child is gone");
          });
    s.add("persistent_reference_survives",
          [](ScenarioWorld& w)
          {
              auto target = w.spawn<Counter>({3, 0, 3});
              auto holder = w.world().create();
              w.world().add<Ref>(holder, Ref{w.world().makePersistent(target)});
              w.step(30);  // the save/load variant replaces the world halfway
              w.check(w.world().resolve(w.world().read<Ref>(holder).target) == target, "resolves after a load");
          });
    // S20: the kinematic mover follows the terrain (terrainScenarioSetup). The y is checked on the owner's world (the
    // client's in the host + client variants: the host extrapolates received poses)
    s.add("walker_follows_terrain",
          [](ScenarioWorld& w)
          {
              auto player = w.spawnPlayer({200, 0, 300});
              w.move(player, Vec3(1, 0, 1));
              auto owner = [&]() -> std::pair<World*, Entity>
              {
                  if (!w.networked()) return {&w.world(), player};
                  return {&w.client(w.defaultPeer()), w.onClient(w.defaultPeer(), player)};
              };
              const Vec3 start = Vec3(200, 0, 300);
              double walked    = 0;
              int mismatches   = 0;
              w.check(w.stepUntil(
                          [&]
                          {
                              auto [world, e] = owner();
                              if (!e.valid() || !world->alive(e)) return false;
                              const Vec3 p = world->read<Transform>(e).position();
                              if (walked > 0 && p.y() != world->terrainHeight(0, p.x(), p.z())) mismatches++;
                              walked = std::hypot(p.x() - start.x(), p.z() - start.z());
                              return walked >= 500;
                          },
                          9000),
                      "walked 500 m");
              w.check(mismatches == 0, "y equals terrainHeight every step (" + std::to_string(mismatches) + " off)");
              w.move(player, Vec3::Zero());
              w.step(20);
              auto [world, e] = owner();
              const Vec3 p    = world->read<Transform>(e).position();
              w.check(p.y() == world->terrainHeight(0, p.x(), p.z()), "rests on the terrain");
              w.check(p.y() > 1, "the slope rose along x: y = " + std::to_string(p.y()));
          });
    s.add(
        "api_edits_reach_gameplay",
        [](ScenarioWorld& w)
        {
            auto wanderer = w.spawn<Wanderer>({0, 0, 0});
            auto player   = w.spawnPlayer({4, 0, 0});
            const Json we = WorldApi::entityJson(wanderer);
            w.check(w.call("set_component",
                           Json::object{{"entity", we}, {"component", "Health"}, {"values", Json::object{{"hp", 50}}}})
                        .ok,
                    "set_component");
            w.check(w.call("emit_event", Json::object{{"event", "DamageEvent"},
                                                      {"values", Json::object{{"target", we},
                                                                              {"source", WorldApi::entityJson(player)},
                                                                              {"amount", 10}}}})
                        .ok,
                    "emit_event");
            w.step(60);
            w.check(w.world().read<Health>(wanderer).hp == 40, "onDamage ran");
            w.check(w.distance(wanderer, player) > 8, "it flees from the agent's attacker");
        });
}

// ---- the spaces scenarios (S21; spacesScenarioSetup)
// ------------------------------------------------------------------------------------------

// The transitions of a world (a world-state test component, saved with the world: nothing leaks between the runs of
// a variant): per transition the entity, the link, the spaces and the step that moved it; the steps at which
// interactSystem saw action 20
struct TransitionLog
{
    static constexpr const char* name = "TransitionLog";
    static constexpr ComponentId id   = ComponentId(TestComponent::TransitionLog);
    static constexpr uint32_t version = 1;
    std::vector<Entity> entities, links;
    std::vector<uint16_t> from, to;
    std::vector<uint64_t> steps, interactSteps;
    template <class V>
    void visit(V& v)
    {
        v("entities", entities);
        v("links", links);
        v("from", from);
        v("to", to);
        v("steps", steps);
        v("interactSteps", interactSteps);
    }
};

constexpr uint32_t kInteract = 20;

// Moves itself to the cellar (space 1) in its update after 30 updates (during a step)
class Diver final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(TestClass::Diver);
    static constexpr const char* name = "Diver";
    void onSpawn() override
    {
        if (!has<TickCount>()) add<TickCount>();
    }
    void update(double) override
    {
        if (++write<TickCount>().updates == 30 && transform().space() == 0)
            world().moveToSpace(entity(), 1, Vec3(650, 0, 650), Quat::Identity());
    }
};

// Phase 4 (host): an ActionEvent with kInteract emits an InteractEvent towards the nearest SpaceLink within 2 m in the
// actor's space (ties by entity index) and records the step
void interactSystem(World& w, double)
{
    for (const ActionEvent& a : w.events.read<ActionEvent>())
    {
        if (a.action != kInteract || !w.alive(a.player)) continue;
        w.global<TransitionLog>().interactSteps.push_back(w.stepIndex());
        const Transform& actor = w.read<Transform>(a.player);
        Entity best;
        double bestDistance = 2;
        w.viewAll<const SpaceLink, const Transform>().each(
            [&](Entity link, const SpaceLink&, const Transform& t)
            {
                if (t.space() != actor.space()) return;
                const double d = (t.position() - actor.position()).norm();
                if (d < bestDistance || (d == bestDistance && best.valid() && link.index < best.index))
                {
                    best         = link;
                    bestDistance = d;
                }
            });
        if (best.valid()) w.events.emit(InteractEvent{best, a.player, kInteract});
    }
}

// Phase 4 (host): records the completed transitions (the event is readable one step after the move)
void transitionLogSystem(World& w, double)
{
    for (const SpaceTransitionEvent& ev : w.events.read<SpaceTransitionEvent>())
    {
        TransitionLog& log = w.global<TransitionLog>();
        log.entities.push_back(ev.entity);
        log.links.push_back(ev.link);
        log.from.push_back(ev.from);
        log.to.push_back(ev.to);
        log.steps.push_back(w.stepIndex() - 1);
    }
}

// scenarioSetup with space 0 (the flat default) and the enclosed cellar (space 1, site origin (640, 0, 640)), the
// transition log, the test interact system and the engine's spaceLinkSystem
WorldSetup spacesScenarioSetup()
{
    WorldSetup s            = scenarioSetup();
    s.config.spaceTemplates = {SpaceTemplate(), SpaceTemplate::enclosed("cellar", 4)};
    s.components            = [](Registry& r)
    {
        r.registerComponent<WandererState>();
        r.registerGlobal<ActionLog>();
        r.registerComponent<TickCount>();
        r.registerComponent<Marker>();
        r.registerGlobal<DestroyLog>();
        r.registerComponent<Ref>();
        r.registerComponent<WorkerFlag>();
        r.registerGlobal<TransitionLog>();
    };
    s.classes = [](ClassList& c)
    {
        c.add<Wanderer>();
        c.add<Counter>();
        c.add<Recorder>();
        c.add<Lazy>();
        c.add<Diver>();
    };
    s.systems = [](SystemList& l)
    {
        l.add(Phase::Gameplay, "actionLog", actionLogSystem);
        l.add(Phase::Gameplay, "workerFlag", workerFlagSystem);
        l.add(Phase::Gameplay, "interact", interactSystem, RunOn::Host);
        l.add(Phase::Gameplay, "space_link", spaceLinkSystem, RunOn::Host);
        l.add(Phase::Gameplay, "transition_log", transitionLogSystem, RunOn::Host);
    };
    return s;
}

Quat yawDegrees(double degrees)
{
    const double a = degrees * 3.14159265358979323846 / 360.0;
    return Quat(std::cos(a), 0, std::sin(a), 0);
}

Entity spaceLink(ScenarioWorld& w, SpaceId space, const Vec3& p, SpaceId target, const Vec3& arrival,
                 const Quat& rotation)
{
    const Entity e = w.world().create();
    if (space != 0) w.world().moveToSpace(e, space, p, Quat::Identity());
    w.world().get<Transform>(e).setPosition(p);
    w.world().add<SpaceLink>(e, SpaceLink{target, arrival, rotation});
    return e;
}

// The angle between two rotations; infinite for a non-finite or non-unit a (the clamp would hide it; PR review)
double angleBetween(const Quat& a, const Quat& b)
{
    const double n = a.norm();
    if (!std::isfinite(n) || std::abs(n - 1) > 1e-9) return std::numeric_limits<double>::infinity();
    return 2 * std::acos(std::min(1.0, std::abs(a.dot(b))));
}

void registerSpacesScenarios(ScenarioList& s)
{
    // §22 acceptance (S21): a player uses an entrance to the cellar and back (the interact chain of Scope 6)
    s.add("uses_a_space_link",
          [](ScenarioWorld& w)
          {
              const Vec3 arrival(640, 0, 640), back(103, 0, 102);
              const Entity link   = spaceLink(w, 0, Vec3(103, 0, 100), 1, arrival, yawDegrees(90));
              const Entity exit   = spaceLink(w, 1, Vec3(645, 0, 640), 0, back, yawDegrees(180));
              const Entity player = w.spawnPlayer({102, 0, 100});
              w.step(10);
              w.input(player, kInteract);
              const Transform* t = nullptr;
              auto space         = [&] { return w.world().read<Transform>(player).space(); };
              w.check(w.stepUntil([&] { return space() == 1; }, 30), "in the cellar");
              t = &w.world().read<Transform>(player);
              w.check((t->position() - arrival).norm() < 1e-3, "at the arrival position");
              w.check(angleBetween(t->rotation(), yawDegrees(90)) < 1e-4, "with the arrival rotation");
              const TransitionLog& log = w.world().readGlobal<TransitionLog>();
              w.check(!log.interactSteps.empty(), "the interact system saw the action");
              const int net   = w.networked() ? 1 : 0;
              const int delay = w.networkConfig().delaySteps;
              if (net)
              {
                  auto ownerSpace = [&]
                  {
                      const Entity c = w.onClient(w.defaultPeer(), player);
                      return c.valid() ? int(w.client(w.defaultPeer()).read<Transform>(c).space()) : -1;
                  };
                  const int bound = 2 * delay + 5 + (w.networkConfig().async ? 6 : 0);
                  w.check(w.stepUntil([&] { return ownerSpace() == 1; }, bound), "the owner's copy in the cellar");
              }
              w.step();
              {
                  const std::vector<Entity> in1 = w.world().queryRadius(1, arrival, 1);
                  w.check(std::find(in1.begin(), in1.end(), player) != in1.end(), "found in the cellar");
                  const std::vector<Entity> in0 = w.world().queryRadius(0, Vec3(102, 0, 100), 5);
                  w.check(std::find(in0.begin(), in0.end(), player) == in0.end(), "not found outside");
              }
              // Walks to the exit (the owner's mover) and interacts with it
              w.move(player, Vec3(1, 0, 0));
              w.check(w.stepUntil(
                          [&]
                          {
                              return (w.world().read<Transform>(player).position() -
                                      w.world().read<Transform>(exit).position())
                                         .norm() < 2;
                          },
                          300),
                      "walked to the exit");
              w.move(player, Vec3::Zero());
              w.input(player, kInteract);
              w.check(w.stepUntil([&] { return space() == 0; }, 30), "back outside");
              w.check((w.world().read<Transform>(player).position() - back).norm() < 1e-3,
                      "at the exit's arrival position");
              w.step(2);
              const TransitionLog& l = w.world().readGlobal<TransitionLog>();
              w.check(l.entities.size() == 2, "two transitions logged");
              w.check(l.entities.size() == 2 && l.entities[0] == player && l.entities[1] == player,
                      "both transitions of the player");
              // Regression (CodeRabbit, S21): a missing interaction record skipped the link, space and timing checks
              w.check(l.interactSteps.size() >= 2, "both interactions logged");
              if (l.entities.size() == 2)
              {
                  w.check(l.links[0] == link && l.from[0] == 0 && l.to[0] == 1, "the first through the link");
                  w.check(l.links[1] == exit && l.from[1] == 1 && l.to[1] == 0, "the second through the exit");
              }
              if (l.entities.size() == 2 && l.interactSteps.size() >= 2)
              {
                  w.check(l.steps[0] - l.interactSteps[0] == 2, "moved 2 steps after the interaction (" +
                                                                    std::to_string(l.steps[0]) + " - " +
                                                                    std::to_string(l.interactSteps[0]) + ")");
                  w.check(l.steps[1] - l.interactSteps.back() == 2, "the exit 2 steps after its interaction");
              }
          });
    // moveToSpace of host-owned objects: between steps with a child, and from GameObject::update
    s.add("objects_change_space",
          [](ScenarioWorld& w)
          {
              const Entity wanderer = w.spawn<Wanderer>({10, 0, 10});
              const Entity marker   = w.world().create();
              w.world().add<LocalTransform>(marker, LocalTransform{Vec3(0, 2, 0), Quat::Identity(), Vec3::Ones()});
              w.world().add<Parent>(marker, Parent{wanderer});
              w.world().add<Marker>(marker, Marker{7});
              const Entity diver = w.spawn<Diver>({20, 0, 20});
              w.step(5);
              w.world().moveToSpace(wanderer, 1, Vec3(640, 0, 640), Quat::Identity());
              w.check(w.world().read<Transform>(marker).space() == 1, "the child moves at once");
              w.step();
              w.check(w.world().read<Transform>(marker).position() == Vec3(640, 2, 640), "the child's pose follows");
              w.check(w.stepUntil([&] { return w.world().read<Transform>(diver).space() == 1; }, 60),
                      "the diver moved itself");
              const int32_t updates = w.world().read<TickCount>(diver).updates;
              w.step(10);
              w.check(w.world().read<TickCount>(diver).updates == updates + 10, "keeps ticking in the cellar");
              w.check(
                  w.world().read<Transform>(wanderer).space() == 1 && w.world().read<Transform>(marker).space() == 1,
                  "the wanderer and its child stay in the cellar");
              if (!w.networked()) return;
              const int bound = w.world().worldConfig().interpolationDelaySteps + w.networkConfig().delaySteps + 3 +
                                (w.networkConfig().async ? 6 : 0);
              for (Entity e : {wanderer, marker, diver})
              {
                  auto replicaIn1 = [&]
                  {
                      const Entity c = w.onClient(w.defaultPeer(), e);
                      if (!c.valid()) return false;
                      const World& client = w.client(w.defaultPeer());
                      return client.read<Transform>(c).space() == 1 && client.has<RenderPose>(c) &&
                             client.read<RenderPose>(c).space == 1;
                  };
                  w.check(w.stepUntil(replicaIn1, bound), "the replica and its RenderPose in the cellar");
              }
          });
}

// ---- the terraforming scenarios (S22; terraformScenarioSetup)
// ------------------------------------------------------------------------------------------

// The brush a Terraformer applies and how many more times (R-STATE-1: the state is a component)
struct TerraformerState
{
    static constexpr const char* name = "TerraformerState";
    static constexpr ComponentId id   = ComponentId(TestComponent::TerraformerState);
    static constexpr uint32_t version = 1;
    TerrainBrush brush;
    uint32_t remaining = 0;
    template <class V>
    void visit(V& v)
    {
        v("brush", brush);
        v("remaining", remaining);
    }
};

// Edits the terrain in its update (phase 3: a command, applied at the phase-3 flush) until remaining is 0
class Terraformer final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(TestClass::Terraformer);
    static constexpr const char* name = "Terraformer";
    void update(double) override
    {
        if (!has<TerraformerState>() || read<TerraformerState>().remaining == 0) return;
        TerraformerState& s = write<TerraformerState>();
        world().editTerrain(0, s.brush);
        s.remaining--;
    }
};

// spacesScenarioSetup's components and classes plus the Terraformer, on terraformTestTemplate(spacing)
WorldSetup terraformScenarioSetup(int spacing)
{
    WorldSetup s            = spacesScenarioSetup();
    s.config.spaceTemplates = {TerrainTestHooks::terraformTestTemplate(spacing)};
    s.components            = [](Registry& r)
    {
        r.registerComponent<WandererState>();
        r.registerGlobal<ActionLog>();
        r.registerComponent<TickCount>();
        r.registerComponent<Marker>();
        r.registerGlobal<DestroyLog>();
        r.registerComponent<Ref>();
        r.registerComponent<WorkerFlag>();
        r.registerGlobal<TransitionLog>();
        r.registerComponent<TerraformerState>();
    };
    s.classes = [](ClassList& c)
    {
        c.add<Wanderer>();
        c.add<Counter>();
        c.add<Recorder>();
        c.add<Lazy>();
        c.add<Diver>();
        c.add<Terraformer>();
    };
    s.systems = [](SystemList& l)
    {
        l.add(Phase::Gameplay, "actionLog", actionLogSystem);
        l.add(Phase::Gameplay, "workerFlag", workerFlagSystem);
    };
    return s;
}

TerrainBrush terrainBrush(TerrainBrush::Op op, double x, double z, double radius, double strength, double y = 0)
{
    TerrainBrush b;
    b.op       = op;
    b.center   = Vec3(x, y, z);
    b.radius   = radius;
    b.strength = strength;
    return b;
}

ChunkId chunkAt(int32_t x, int32_t z)
{
    ChunkId c;
    c.x = x;
    c.z = z;
    return c;
}

// The base surface (without edits) at (x, z): the §12.2 rule over TerrainSystem::sample
double baseHeight(const World& w, double x, double z)
{
    const double sp = w.spaceTemplate(0).baseSpacing;
    const double gx = x / sp, gz = z / sp;
    const int64_t ci = int64_t(std::floor(gx)), cj = int64_t(std::floor(gz));
    const double fx = gx - double(ci), fz = gz - double(cj);
    const TerrainSystem& t = w.terrain();
    return terrainSurfaceDouble(t.sample(0, ci, cj), t.sample(0, ci + 1, cj), t.sample(0, ci, cj + 1),
                                t.sample(0, ci + 1, cj + 1), fx, fz) *
           kHeightUnit;
}

void registerTerraformScenarios(ScenarioList& s)
{
    s.add("terraform_dig_and_fill",
          [](ScenarioWorld& w)
          {
              w.step(10);
              // A mound of 5 x 1 m at the sample (1000, 1000) in steps 10-14 (the command path, phase 3)
              TerraformerState state;
              state.brush     = terrainBrush(TerrainBrush::Add, 1000, 1000, 6, 1);
              state.remaining = 5;
              w.spawn<Terraformer>(Vec3(1000, 0, 1000), state);
              w.step(10);
              // Step 20: a 2 m pit with radius 12 m on the chunk corner (256, 256) through the world API (immediate)
              const ApiResult pit = w.call(
                  "edit_terrain",
                  Json::object{
                      {"op", "subtract"}, {"center", Json::array{256, 0, 256}}, {"radius", 12}, {"strength", 2}});
              w.check(pit.ok && pit.value["chunks"].array_items().size() == 4, "edit_terrain: " + pit.error);
              w.step(10);
              // Step 30: flatten (2000, 600) to its base + 3 m; checked right after the immediate edit
              const int32_t base = w.world().terrain().sample(0, 2000, 600);
              w.editTerrain(terrainBrush(TerrainBrush::Flatten, 2000, 600, 8, 1, double(base + 3072) * kHeightUnit));
              w.check(w.terrainSample(2000, 600) == base + 3072, "the flatten reached its target exactly");
              w.step(1);
              // Step 31: smooth the same spot (changes the bump the flatten left)
              w.editTerrain(terrainBrush(TerrainBrush::Smooth, 2000, 600, 8, 1));
              w.step(5);
              // The final state only (wherever the save/load swap fell)
              const World& h             = w.world();
              const TerrainEditStore& st = h.terrainEdits();
              w.check(h.terrainSample(0, 1000, 1000) == h.terrain().sample(0, 1000, 1000) + 5 * 1024,
                      "the mound's center is base + 5 m");
              w.check(st.sequence(chunkAt(3, 3)) == 5,
                      "the mound's chunk has sequence 5, is " + std::to_string(st.sequence(chunkAt(3, 3))));
              w.check(h.terrainSample(0, 256, 256) == h.terrain().sample(0, 256, 256) - 2048,
                      "the pit's center is base - 2 m");
              w.check(h.terrainSample(0, 268, 256) == h.terrain().sample(0, 268, 256) &&
                          h.terrainSample(0, 256, 244) == h.terrain().sample(0, 256, 244),
                      "the samples 12 m from the pit's center are unchanged");
              for (const ChunkId& c : {chunkAt(0, 0), chunkAt(1, 0), chunkAt(0, 1), chunkAt(1, 1)})
                  w.check(st.sequence(c) == 1, "a corner chunk has sequence 1");
              w.check(st.sequence(chunkAt(7, 2)) == 2,
                      "the flatten spot's chunk has sequence 2, is " + std::to_string(st.sequence(chunkAt(7, 2))));
              w.check(st.chunkCount() == 6, "six chunks have state, " + std::to_string(st.chunkCount()));
          });
    s.add("terraform_player_walks_into_crater",
          [](ScenarioWorld& w)
          {
              auto player = w.spawnPlayer({500, 0, 500});
              w.step(5);
              w.editTerrain(terrainBrush(TerrainBrush::Subtract, 520, 500, 10, 3));
              w.move(player, Vec3(1, 0, 0));
              auto owner = [&]() -> std::pair<World*, Entity>
              {
                  if (!w.networked()) return {&w.world(), player};
                  return {&w.client(w.defaultPeer()), w.onClient(w.defaultPeer(), player)};
              };
              double y = 0, ground = 0;
              w.check(w.stepUntil(
                          [&]
                          {
                              auto [world, e] = owner();
                              if (!e.valid() || !world->alive(e)) return false;
                              const Vec3 p = world->read<Transform>(e).position();
                              y            = p.y();
                              ground       = baseHeight(w.world(), p.x(), p.z());
                              return std::abs(p.y() - w.world().terrainHeight(0, p.x(), p.z())) <= 1e-4 &&
                                     p.y() <= ground - 2;
                          },
                          600),
                      "the player walked into the crater (y " + std::to_string(y) + ", base ground " +
                          std::to_string(ground) + ")");
              w.move(player, Vec3::Zero());
          });
}

// A scenario with the setup it runs on and the registration it comes from
struct Case
{
    std::string scenario;
    ScenarioVariant variant;
    WorldSetup (*setup)(const std::string& scenario);
    void (*registerScenarios)(ScenarioList&);
};
WorldSetup spacesSetupFor(const std::string&)
{
    return spacesScenarioSetup();
}
WorldSetup terraformSetupFor(const std::string& scenario)
{
    return terraformScenarioSetup(scenario == "terraform_dig_and_fill" ? 1 : 2);
}
std::vector<Case> allCases()
{
    std::vector<Case> cases;
    ScenarioOptions options;
    options.networkVariants = true;
    auto collect            = [&](void (*registerScenarios)(ScenarioList&), WorldSetup (*setup)(const std::string&))
    {
        ScenarioList list;
        registerScenarios(list);
        for (const auto& e : list.entries())
            for (ScenarioVariant v : variantsFor(e, options))
                cases.push_back(Case{e.name, v, setup, registerScenarios});
    };
    collect(registerEngineScenarios, setupFor);
    collect(registerSpacesScenarios, spacesSetupFor);
    collect(registerTerraformScenarios, terraformSetupFor);
    return cases;
}
}  // namespace

class EngineScenario : public ::testing::TestWithParam<Case>
{
};

TEST_P(EngineScenario, Passes)
{
    g_mainThread = std::this_thread::get_id();
    ScenarioList list;
    GetParam().registerScenarios(list);
    const ScenarioList::Entry* entry = list.find(GetParam().scenario);
    ASSERT_TRUE(entry);
    ScenarioOptions options;
    // The engine's own systems must not depend on the thread count, and continue identically after a load
    options.varyThreads        = true;
    options.strictContinuation = true;
    options.networkVariants    = true;
    const ScenarioResult r     = runScenario(GetParam().setup(entry->name), *entry, GetParam().variant, options);
    EXPECT_TRUE(r.passed) << r.report;
}

INSTANTIATE_TEST_SUITE_P(Engine, EngineScenario, ::testing::ValuesIn(allCases()),
                         [](const ::testing::TestParamInfo<Case>& info)
                         { return info.param.scenario + "_" + variantName(info.param.variant); });

// ---------------------------------------------------------------------------------------------------------------------
// The runner detects each kind of failure

namespace
{
ScenarioList::Entry entry(const char* name, ScenarioFn fn)
{
    ScenarioList l;
    l.add(name, std::move(fn));
    return l.entries()[0];
}

int g_globalCounter = 0;
// Reads a process-global counter: different in every run (nondeterministic)
class Rando final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(TestClass::Wanderer);
    static constexpr const char* name = "Rando";
    void onSpawn() override { add<TickCount>(); }
    void update(double) override { write<TickCount>().updates = g_globalCounter++; }
};
// Keeps its count in a member (R-STATE-1 violated): a load resets it
class Forgetful final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(TestClass::Wanderer);
    static constexpr const char* name = "Forgetful";
    void onSpawn() override { add<TickCount>(); }
    void update(double) override { write<TickCount>().updates = ++n; }
    int n = 0;
};
WorldSetup withClass(void (*classes)(ClassList&))
{
    WorldSetup s = scenarioSetup();
    s.classes    = classes;
    return s;
}
bool contains(const ScenarioResult& r, const std::string& text)
{
    return r.report.find(text) != std::string::npos;
}
}  // namespace

TEST(ScenarioRunner, AFailingCheckIsReportedWithItsStepLogsAndDumps)
{
    auto e                 = entry("fails",
                                   [](ScenarioWorld& w)
                                   {
                       auto c = w.spawn<Counter>({1, 2, 3});
                       w.step(3);
                       SAIGA_LOG("scenario", "a log line of the scenario");
                       w.check(false, "always fails");
                       (void)c;
                   });
    const ScenarioResult r = runScenario(scenarioSetup(), e, ScenarioVariant::Plain);
    EXPECT_FALSE(r.passed);
    ASSERT_EQ(r.failures.size(), 1u);
    EXPECT_EQ(r.failures[0], "step 3: always fails");
    EXPECT_TRUE(contains(r, "a log line of the scenario")) << r.report;
    EXPECT_TRUE(contains(r, "TickCount")) << r.report;  // the dump of the involved entity
}

TEST(ScenarioRunner, RepeatCatchesNondeterminism)
{
    auto e             = entry("random",
                               [](ScenarioWorld& w)
                               {
                       w.spawn<Rando>({0, 0, 0});
                       w.step(10);
                   });
    const WorldSetup s = withClass([](ClassList& c) { c.add<Rando>(); });
    EXPECT_TRUE(runScenario(s, e, ScenarioVariant::Plain).passed);
    const ScenarioResult r = runScenario(s, e, ScenarioVariant::Repeat);
    EXPECT_FALSE(r.passed);
    EXPECT_TRUE(contains(r, "TickCount.updates")) << r.report;
}

TEST(ScenarioRunner, SaveLoadCatchesStateInMembers)
{
    auto withCheck     = entry("member_state",
                               [](ScenarioWorld& w)
                               {
                               auto f = w.spawn<Forgetful>({0, 0, 0});
                               w.step(20);
                               w.check(w.world().read<TickCount>(f).updates == 20, "counted every update");
                           });
    const WorldSetup s = withClass([](ClassList& c) { c.add<Forgetful>(); });
    EXPECT_TRUE(runScenario(s, withCheck, ScenarioVariant::Plain).passed);
    EXPECT_FALSE(runScenario(s, withCheck, ScenarioVariant::SaveLoad).passed);
    // Without a check that looks at it, strictContinuation catches it
    auto noCheck = entry("member_state_unchecked",
                         [](ScenarioWorld& w)
                         {
                             w.spawn<Forgetful>({0, 0, 0});
                             w.step(20);
                         });
    EXPECT_TRUE(runScenario(s, noCheck, ScenarioVariant::SaveLoad).passed);
    ScenarioOptions strict;
    strict.strictContinuation = true;
    const ScenarioResult r    = runScenario(s, noCheck, ScenarioVariant::SaveLoad, strict);
    EXPECT_FALSE(r.passed);
    EXPECT_TRUE(contains(r, "R-STATE-1")) << r.report;
}

TEST(ScenarioRunner, ThreadDependenceOnlyWithVaryThreads)
{
    g_mainThread = std::this_thread::get_id();
    auto e       = entry("threads",
                         [](ScenarioWorld& w)
                         {
                       for (int i = 0; i < 4000; ++i) w.world().add<WorkerFlag>(w.world().create());
                       w.step();
                   });
    EXPECT_TRUE(runScenario(scenarioSetup(), e, ScenarioVariant::Repeat).passed);
    ScenarioOptions vary;
    vary.varyThreads       = true;
    const ScenarioResult r = runScenario(scenarioSetup(), e, ScenarioVariant::Repeat, vary);
    EXPECT_FALSE(r.passed);
    EXPECT_TRUE(contains(r, "WorkerFlag.onWorker")) << r.report;
}

// Regression (CodeRabbit, S14): the deadline was only checked after steps, so a run that finished after it (one step,
// then a long final check) started the Repeat rerun and the SaveLoad run anyway; and a SaveLoad baseline stopped by the
// deadline lost its steps and report
TEST(ScenarioRunner, NoFurtherRunAfterTheDeadline)
{
    static int calls = 0;
    auto late        = entry("late",
                             [](ScenarioWorld& w)
                             {
                          calls++;
                          w.step(1);
                          std::this_thread::sleep_for(std::chrono::milliseconds(300));
                      });
    for (ScenarioVariant v : {ScenarioVariant::Repeat, ScenarioVariant::SaveLoad})
    {
        ScenarioOptions o;
        o.deadline             = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
        calls                  = 0;
        const ScenarioResult r = runScenario(scenarioSetup(), late, v, o);
        EXPECT_FALSE(r.passed) << variantName(v);
        EXPECT_TRUE(contains(r, "tool time budget")) << r.report;
        EXPECT_EQ(r.steps, 1u) << r.report;
        EXPECT_EQ(calls, 1) << variantName(v);  // one run only
    }
    auto slow = entry("slow",
                      [](ScenarioWorld& w)
                      {
                          SAIGA_LOG("scenario", "a log line of the slow scenario");
                          w.addStepListener([]() { std::this_thread::sleep_for(std::chrono::milliseconds(20)); });
                          w.step(1000);
                      });
    ScenarioOptions o;
    o.deadline             = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
    const ScenarioResult r = runScenario(scenarioSetup(), slow, ScenarioVariant::SaveLoad, o);
    EXPECT_FALSE(r.passed);
    EXPECT_GT(r.steps, 0u) << r.report;
    EXPECT_TRUE(contains(r, "a log line of the slow scenario")) << r.report;
}

TEST(ScenarioRunner, ExceptionsStepLimitsAndOtherStepPaths)
{
    auto throws            = entry("throws", [](ScenarioWorld&) { throw std::runtime_error("boom"); });
    const ScenarioResult t = runScenario(scenarioSetup(), throws, ScenarioVariant::Plain);
    EXPECT_FALSE(t.passed);
    EXPECT_TRUE(contains(t, "exception: boom")) << t.report;

    ScenarioOptions small;
    small.maxSteps = 10;
    auto helper    = entry("too_long", [](ScenarioWorld& w) { w.step(20); });
    EXPECT_TRUE(contains(runScenario(scenarioSetup(), helper, ScenarioVariant::Plain, small), "more than 10 steps"));
    // Through the world API: the API keeps stepping, the next call ends the scenario
    auto api               = entry("too_long_api",
                                   [](ScenarioWorld& w)
                                   {
                         w.call("pause", Json::object{});
                         w.call("step", Json::object{{"n", 20}});
                         w.check(false, "not reached");
                     });
    const ScenarioResult a = runScenario(scenarioSetup(), api, ScenarioVariant::Plain, small);
    EXPECT_TRUE(contains(a, "more than 10 steps")) << a.report;
    EXPECT_FALSE(contains(a, "not reached")) << a.report;

    // Steps outside the helpers are observed (hashes, count)
    auto direct = entry("direct_steps",
                        [](ScenarioWorld& w)
                        {
                            w.spawn<Counter>({0, 0, 0});
                            w.world().step(5);
                            w.check(w.steps() == 5, "5 steps observed");
                        });
    for (ScenarioVariant v : {ScenarioVariant::Plain, ScenarioVariant::Repeat})
    {
        const ScenarioResult r = runScenario(scenarioSetup(), direct, v);
        EXPECT_TRUE(r.passed) << r.report;
    }
    // Regression (CodeRabbit, S12): without a step helper or call() after the middle the round trip ran only at the end
    // and the variant passed although nothing ran on the loaded world
    const ScenarioResult d = runScenario(scenarioSetup(), direct, ScenarioVariant::SaveLoad);
    EXPECT_FALSE(d.passed);
    EXPECT_TRUE(contains(d, "nothing ran on the loaded world")) << d.report;
    // Regression (CodeRabbit, S12): one call() that ran every step swapped the world only after them
    auto oneCall           = entry("one_call",
                                   [](ScenarioWorld& w)
                                   {
                             w.spawn<Counter>({0, 0, 0});
                             w.call("pause", Json::object{});
                             w.call("step", Json::object{{"n", 5}});
                         });
    const ScenarioResult o = runScenario(scenarioSetup(), oneCall, ScenarioVariant::SaveLoad);
    EXPECT_FALSE(o.passed);
    EXPECT_TRUE(contains(o, "nothing ran on the loaded world")) << o.report;
    // The API's step through call() swaps the world like a step helper
    auto called = entry("called_steps",
                        [](ScenarioWorld& w)
                        {
                            w.spawn<Counter>({0, 0, 0});
                            w.call("pause", Json::object{});
                            w.call("step", Json::object{{"n", 5}});
                            w.call("step", Json::object{{"n", 5}});
                            w.check(w.steps() == 10, "10 steps observed");
                        });
    ScenarioOptions strict;
    strict.strictContinuation = true;
    const ScenarioResult c    = runScenario(scenarioSetup(), called, ScenarioVariant::SaveLoad, strict);
    EXPECT_TRUE(c.passed) << c.report;
    // No steps at all: the save/load round trip happens at the end
    auto none = entry("no_steps", [](ScenarioWorld& w) { w.spawn<Counter>({0, 0, 0}); });
    EXPECT_TRUE(runScenario(scenarioSetup(), none, ScenarioVariant::SaveLoad).passed);
}

// The world API survives the save/load swap: cursors, snapshots and the paused state
TEST(ScenarioRunner, SaveLoadKeepsTheApi)
{
    auto e = entry("api_across_swap",
                   [](ScenarioWorld& w)
                   {
                       w.spawn<Counter>({0, 0, 0});
                       w.check(w.call("save_snapshot", Json::object{{"name", "x"}}).ok, "save_snapshot");
                       const Json cursor = w.call("changes_since", Json::object{{"cursor", "now"}}).value["cursor"];
                       w.step(10);
                       w.check(w.call("changes_since", Json::object{{"cursor", cursor}}).ok, "the cursor is valid");
                       w.check(w.call("load_snapshot", Json::object{{"name", "x"}}).ok, "the snapshot survived");
                       w.call("pause", Json::object{});
                       w.step(4);
                       w.check(w.world().isPaused(), "still paused");
                   });
    for (ScenarioVariant v : {ScenarioVariant::Plain, ScenarioVariant::SaveLoad})
    {
        const ScenarioResult r = runScenario(scenarioSetup(), e, v);
        EXPECT_TRUE(r.passed) << r.report;
    }
}

TEST(ScenarioRunner, ListAndRunAll)
{
    ScenarioList list;
    registerEngineScenarios(list);
    EXPECT_EQ(list.match("wanderer").size(), 1u);
    EXPECT_EQ(list.match("").size(), list.entries().size());
    EXPECT_TRUE(list.match("nothing_matches").empty());
    EXPECT_EQ(runScenarios(scenarioSetup(), list, "persistent").size(), 3u);
    // With the host + client variants (S16): six per scenario that is not local-only
    ScenarioOptions options;
    options.networkVariants = true;
    const auto results      = runScenarios(scenarioSetup(), list, "persistent", options);
    ASSERT_EQ(results.size(), 6u);
    for (const ScenarioResult& r : results) EXPECT_TRUE(r.passed) << r.report;
}

// Refused in every build
TEST(ScenarioRunnerDeathTest, NamesAreUniqueAndSnakeCase)
{
    ScenarioList list;
    list.add("one", [](ScenarioWorld&) {});
    EXPECT_DEATH(list.add("one", [](ScenarioWorld&) {}), "already used");
    EXPECT_DEATH(list.add("Not Snake", [](ScenarioWorld&) {}), "snake_case");
}

// Review regressions (S12 round 1)

// Listeners added through ScenarioWorld survive the save/load swap; ones added to world() directly fail it loudly
// (they were lost silently); the advance accumulator carries over
TEST(ScenarioRunner, ListenersAndTheAccumulatorSurviveTheSwap)
{
    auto own = entry("own_listener",
                     [](ScenarioWorld& w)
                     {
                         auto n = std::make_shared<int>(0);
                         w.addStepListener([n]() { (*n)++; });
                         w.step(20);
                         w.check(*n == 20, "the listener saw every step (" + std::to_string(*n) + ")");
                     });
    EXPECT_TRUE(runScenario(scenarioSetup(), own, ScenarioVariant::SaveLoad).passed);
    auto direct            = entry("direct_listener",
                                   [](ScenarioWorld& w)
                                   {
                            w.world().addStepListener([]() { return true; });
                            w.step(20);
                        });
    const ScenarioResult r = runScenario(scenarioSetup(), direct, ScenarioVariant::SaveLoad);
    EXPECT_FALSE(r.passed);
    EXPECT_TRUE(contains(r, "addStepListener")) << r.report;
    auto advance = entry("advance",
                         [](ScenarioWorld& w)
                         {
                             w.spawn<Counter>({0, 0, 0});
                             for (int i = 0; i < 30; ++i)
                             {
                                 w.world().advance(1.3 / 60);
                                 w.step();
                             }
                         });
    ScenarioOptions strict;
    strict.strictContinuation = true;
    const ScenarioResult a    = runScenario(scenarioSetup(), advance, ScenarioVariant::SaveLoad, strict);
    EXPECT_TRUE(a.passed) << a.report;
}

// A step helper inside a step listener (the world's listener dispatch is not re-entrant): a scenario failure, not a
// crash or a swap of the running world
TEST(ScenarioRunner, SteppingInsideAStepListenerFails)
{
    auto e = entry("stepping_listener",
                   [](ScenarioWorld& w)
                   {
                       w.spawn<Counter>({0, 0, 0});
                       w.addStepListener([&w]() { w.step(); });
                       w.world().step(6);
                   });
    for (ScenarioVariant v : {ScenarioVariant::Plain, ScenarioVariant::SaveLoad})
    {
        const ScenarioResult r = runScenario(scenarioSetup(), e, v);
        EXPECT_FALSE(r.passed);
        EXPECT_TRUE(contains(r, "from a step listener")) << r.report;
    }
    // maxSteps ends a loop of world().step() too
    ScenarioOptions small;
    small.maxSteps = 10;
    auto loop      = entry("endless",
                           [](ScenarioWorld& w)
                           {
                          for (;;) w.world().step();
                      });
    EXPECT_TRUE(contains(runScenario(scenarioSetup(), loop, ScenarioVariant::Plain, small), "more than 10 steps"));
}

// Review regressions (CodeRabbit, S12)

// A failure of the plain baseline run fails the save/load variant (it was dropped when the second run passed)
TEST(ScenarioRunner, SaveLoadReportsAFailingBaseline)
{
    auto runs              = std::make_shared<int>(0);
    auto e                 = entry("fails_once",
                                   [runs](ScenarioWorld& w)
                                   {
                       w.step(4);
                       if ((*runs)++ == 0) throw std::runtime_error("first run only");
                   });
    const ScenarioResult r = runScenario(scenarioSetup(), e, ScenarioVariant::SaveLoad);
    EXPECT_FALSE(r.passed);
    EXPECT_TRUE(contains(r, "baseline run: ")) << r.report;
    EXPECT_TRUE(contains(r, "first run only")) << r.report;
}

// A listener's captured state survives the swap (each registration copied the callable, the copy in the replacement
// started from the registration-time state)
TEST(ScenarioRunner, ListenerStateSurvivesTheSwap)
{
    auto e                 = entry("stateful_listener",
                                   [](ScenarioWorld& w)
                                   {
                       w.addStepListener(
                           [&w, n = uint64_t(0)]() mutable
                           {
                               n++;
                               w.check(n == w.steps(), "the listener counted " + std::to_string(n) + " steps");
                           });
                       w.step(20);
                   });
    const ScenarioResult r = runScenario(scenarioSetup(), e, ScenarioVariant::SaveLoad);
    EXPECT_TRUE(r.passed) << r.report;
}

// saveLoad inside a step listener would replace the world while it runs its listeners
TEST(ScenarioRunner, SaveLoadInsideAStepListenerFails)
{
    auto e                 = entry("swapping_listener",
                                   [](ScenarioWorld& w)
                                   {
                       w.addStepListener([&w]() { w.saveLoad(); });
                       w.step(2);
                   });
    const ScenarioResult r = runScenario(scenarioSetup(), e, ScenarioVariant::Plain);
    EXPECT_FALSE(r.passed);
    EXPECT_TRUE(contains(r, "saveLoad was called from a step listener")) << r.report;
}

// The tick setting carries over from the world (also when set on world() directly)
TEST(ScenarioRunner, TheTickSettingOfTheWorldCarriesOver)
{
    auto e = entry("tick_setting",
                   [](ScenarioWorld& w)
                   {
                       w.world().setTickEveryStep(false);
                       w.spawnPlayer({0, 0, 0});
                       auto far = w.spawn<Counter>({1000, 0, 0});
                       w.step(60);
                       w.check(w.world().read<TickCount>(far).updates <= 3, "far objects tick slowly");
                   });
    ScenarioOptions strict;
    strict.strictContinuation = true;
    const ScenarioResult r    = runScenario(scenarioSetup(), e, ScenarioVariant::SaveLoad, strict);
    EXPECT_TRUE(r.passed) << r.report;
}

// count<Transform> counted the world-state entity; stepUntil returns false when not reached; traces are reported
TEST(ScenarioRunner, CountsStepUntilAndTraces)
{
    auto e                 = entry("helpers",
                                   [](ScenarioWorld& w)
                                   {
                       w.check(w.count<Transform>() == 0, "an empty world");
                       auto c = w.spawn<Counter>({0, 0, 0});
                       w.check(w.count<Transform>() == 1 && w.count<TickCount>() == 1, "one object");
                       w.check(!w.stepUntil([] { return false; }, 5), "not reached");
                       w.check(w.steps() == 5, "stepped 5 times");
                       w.trace<TickCount>(c);
                       w.step(2);
                       w.check(false, "show the trace");
                   });
    const ScenarioResult r = runScenario(scenarioSetup(), e, ScenarioVariant::Plain);
    ASSERT_EQ(r.failures.size(), 1u) << r.report;
    EXPECT_TRUE(contains(r, "trace TickCount")) << r.report;
    // Each failure once
    EXPECT_EQ(r.report.find("show the trace"), r.report.rfind("show the trace")) << r.report;
}

TEST(ScenarioRunner, FindIsExact)
{
    ScenarioList list;
    list.add("walk_fast", [](ScenarioWorld&) {});
    list.add("walk", [](ScenarioWorld&) {});
    ASSERT_TRUE(list.find("walk"));
    EXPECT_EQ(list.find("walk")->name, "walk");
    EXPECT_FALSE(list.find("wal"));
}

// Review regressions (S12 round 3)

// After the first abort the observer was removed by the world: later API steps were neither counted nor stopped
TEST(ScenarioRunner, TheObserverSurvivesAnAbort)
{
    ScenarioOptions small;
    small.maxSteps         = 10;
    auto e                 = entry("api_loop",
                                   [](ScenarioWorld& w)
                                   {
                       w.api().call("pause", Json::object{});
                       for (int i = 0; i < 50; ++i) w.api().call("step", Json::object{{"n", 1}});
                       w.check(w.steps() == 50, "every step observed (" + std::to_string(w.steps()) + ")");
                   });
    const ScenarioResult r = runScenario(scenarioSetup(), e, ScenarioVariant::Plain, small);
    EXPECT_FALSE(r.passed);
    EXPECT_TRUE(contains(r, "more than 10 steps")) << r.report;
    EXPECT_FALSE(contains(r, "every step observed")) << r.report;
}

// The world API refuses edits (a step) from inside a step listener (it aborted the process)
TEST(ScenarioRunner, ApiStepInsideAListenerIsAnError)
{
    World w(scenarioSetup());
    WorldApi api(w);
    api.call("pause", Json::object{});
    ApiResult inner;
    w.addStepListener(
        [&]()
        {
            inner = api.call("step", Json::object{{"n", 1}});
            return false;
        });
    w.step();
    EXPECT_FALSE(inner.ok);
    EXPECT_NE(inner.error.find("not inside a step listener"), std::string::npos) << inner.error;
}

// Review regressions (CodeRabbit, S12, second review)

// Regression (CodeRabbit, S12): the Repeat hash was taken before the scenario's step listeners ran, so a listener that
// wrote different saved values on step 1 (reset by the scenario later) was not noticed
TEST(ScenarioRunner, RepeatHashesAfterTheStepListeners)
{
    auto runs              = std::make_shared<int>(0);
    auto e                 = entry("listener_marker",
                                   [runs](ScenarioWorld& w)
                                   {
                       const Entity m = w.world().create();
                       w.world().add<Marker>(m);
                       const int run = (*runs)++;
                       w.addStepListener(
                           [&w, m, run]()
                           {
                               if (w.steps() == 1) w.world().get<Marker>(m).value = 1 + run;
                           });
                       w.step();
                       w.world().get<Marker>(m).value = 0;
                       w.step();
                   });
    const ScenarioResult r = runScenario(scenarioSetup(), e, ScenarioVariant::Repeat);
    EXPECT_FALSE(r.passed);
    EXPECT_TRUE(contains(r, "differ after step 1")) << r.report;
}

// Regression (CodeRabbit, S12): a diagnostic rerun that ended before the differing step left an empty save, and the
// report said the difference was not reproducible
TEST(ScenarioRunner, ARerunThatEndsEarlyIsReported)
{
    auto runs              = std::make_shared<int>(0);
    auto e                 = entry("rerun_fails",
                                   [runs](ScenarioWorld& w)
                                   {
                       const int run = (*runs)++;
                       if (run >= 2) throw std::runtime_error("rerun fails");
                       const Entity m = w.world().create();
                       w.world().add<Marker>(m);
                       w.world().get<Marker>(m).value = run;
                       w.step();
                   });
    const ScenarioResult r = runScenario(scenarioSetup(), e, ScenarioVariant::Repeat);
    EXPECT_FALSE(r.passed);
    EXPECT_TRUE(contains(r, "rerun ended before step 1")) << r.report;
    EXPECT_TRUE(contains(r, "rerun fails")) << r.report;
    EXPECT_FALSE(contains(r, "not reproducible")) << r.report;
}

// Regression (CodeRabbit, S12): an exception of a WorldSetup callback left runOnce and ended the whole run
TEST(ScenarioRunner, AFailingWorldSetupFailsTheVariant)
{
    WorldSetup s = scenarioSetup();
    s.components = [](Registry&) { throw std::runtime_error("setup boom"); };
    auto e       = entry("any", [](ScenarioWorld& w) { w.step(); });
    for (ScenarioVariant v : {ScenarioVariant::Plain, ScenarioVariant::Repeat, ScenarioVariant::SaveLoad})
    {
        const ScenarioResult r = runScenario(s, e, v);
        EXPECT_FALSE(r.passed);
        EXPECT_TRUE(contains(r, "could not be created: setup boom")) << r.report;
    }
}

Json obj(const std::string& text)
{
    std::string err;
    Json j = Json::parse(text, err);
    EXPECT_TRUE(err.empty()) << err;
    return j;
}

ScenarioWorld* g_scenario = nullptr;
bool g_swapInSystem       = false;

// Regression (CodeRabbit, S12): saveLoad from a system (during the step) reached World::save, which asserts
TEST(ScenarioRunner, SaveLoadDuringAStepFails)
{
    WorldSetup s = scenarioSetup();
    s.systems    = [](SystemList& l)
    {
        l.add(Phase::Gameplay, "swapper",
              [](World&, double)
              {
                  if (!g_swapInSystem || !g_scenario) return;
                  g_swapInSystem = false;
                  g_scenario->saveLoad();
              });
    };
    auto e                 = entry("swap_in_system",
                                   [](ScenarioWorld& w)
                                   {
                       g_scenario     = &w;
                       g_swapInSystem = true;
                       w.step();
                   });
    const ScenarioResult r = runScenario(s, e, ScenarioVariant::Plain);
    g_scenario             = nullptr;
    EXPECT_FALSE(r.passed);
    EXPECT_TRUE(contains(r, "during a step")) << r.report;
}

// Regression (CodeRabbit, S12): the replacement got the scenario's tick setting after the load, so configure() of the
// loaded game objects saw the default
TEST(ScenarioRunner, TheReplacementHasTheTickSettingBeforeTheLoad)
{
    auto e                 = entry("tick_setting_on_load",
                                   [](ScenarioWorld& w)
                                   {
                       w.spawn<Lazy>({0, 0, 0});
                       w.step(4);
                       w.step(4);
                       w.check(g_lastConfigureTick, "configure saw tickEveryStep");
                   });
    const ScenarioResult r = runScenario(scenarioSetup(), e, ScenarioVariant::SaveLoad);
    EXPECT_TRUE(r.passed) << r.report;
}

// Regression (CodeRabbit, S12): a failing load listener of the swap was not reported
TEST(ScenarioRunner, LoadListenerErrorsOfTheSwapFail)
{
    auto e                 = entry("throwing_load_listener",
                                   [](ScenarioWorld& w)
                                   {
                       w.spawn<Lazy>({0, 0, 0});
                       w.step(4);
                       w.step(4);
                   });
    g_throwingLoadListener = true;
    const ScenarioResult r = runScenario(scenarioSetup(), e, ScenarioVariant::SaveLoad);
    g_throwingLoadListener = false;
    EXPECT_FALSE(r.passed);
    EXPECT_TRUE(contains(r, "a load listener failed: listener boom")) << r.report;
}

// Regression (CodeRabbit, S12): the swap compared listener counts; a scenario listener the world dropped after it threw
// hid a direct listener, which was lost silently
TEST(ScenarioRunner, ADirectListenerIsFoundByItsId)
{
    auto e                 = entry("dropped_and_direct",
                                   [](ScenarioWorld& w)
                                   {
                       auto thrown = std::make_shared<bool>(false);
                       w.addStepListener(
                           [thrown]()
                           {
                               if (*thrown) return;
                               *thrown = true;
                               throw std::runtime_error("once");
                           });
                       try
                       {
                           w.step();
                       }
                       catch (const std::runtime_error&)
                       {
                       }
                       w.world().addStepListener([]() { return true; });
                       w.step(20);
                   });
    const ScenarioResult r = runScenario(scenarioSetup(), e, ScenarioVariant::SaveLoad);
    EXPECT_FALSE(r.passed);
    EXPECT_TRUE(contains(r, "addStepListener")) << r.report;
}

// Regression (CodeRabbit, S12): a read-only apply batch inside a step listener was refused; one with an edit is refused
// before any of its operations runs
TEST(ScenarioRunner, ReadOnlyApplyInsideAStepListener)
{
    auto e                 = entry("apply_in_listener",
                                   [](ScenarioWorld& w)
                                   {
                       w.api();
                       auto done = std::make_shared<bool>(false);
                       w.addStepListener(
                           [&w, done]()
                           {
                               if (*done) return;
                               *done = true;
                               const ApiResult read =
                                   w.api().call("apply", obj(R"({"operations": [{"op": "world_status"}]})"));
                               w.check(read.ok, "read-only apply: " + read.error);
                               const ApiResult edit = w.api().call(
                                   "apply", obj(R"({"operations": [{"op": "world_status"}, {"op": "pause"}]})"));
                               w.check(!edit.ok && edit.error.find("nothing of the batch") != std::string::npos,
                                                       "edit batch refused: " + edit.error);
                               w.check(!w.world().isPaused(), "not paused");
                           });
                       w.step();
                   });
    const ScenarioResult r = runScenario(scenarioSetup(), e, ScenarioVariant::Plain);
    EXPECT_TRUE(r.passed) << r.report;
}

// ---------------------------------------------------------------------------------------------------------------------
// The host + client variants (S16, §17.3)

namespace
{
// Writes the PlayerInput of every player on the host each step: fine in one world, a write to a client's
// owner-authoritative component in the host + client variant (R-NET-2)
void hostInputWriterSystem(World& w, double)
{
    w.view<PlayerInput>().each([](Entity, PlayerInput& in) { in.heldButtons++; });
}
}  // namespace

TEST(ScenarioRunner, HostClientRunsTheClients)
{
    auto e = entry("networked",
                   [](ScenarioWorld& w)
                   {
                       auto player   = w.spawnPlayer({0, 0, 0});
                       auto wanderer = w.spawn<Wanderer>({3, 0, 0});
                       w.check(w.networked(), "networked");
                       w.check(w.world().ownerOf(player) == 1, "the player belongs to the client");
                       w.check(w.stepUntil([&] { return w.onClient(1, player).valid(); }, 10), "the player arrives");
                       w.check(w.onClient(1, wanderer).valid(), "the wanderer arrives");
                       w.check(w.client(1).has<CharacterIntent>(w.onClient(1, player)), "the presence list");
                       // An owner-authoritative edit on the owner reaches the host
                       w.edit(player, [](World& world, Entity p) { world.get<PlayerInput>(p).heldButtons = 9; });
                       w.check(w.stepUntil([&] { return w.world().read<PlayerInput>(player).heldButtons == 9; }, 10),
                               "the owner's input reaches the host");
                   });
    for (ScenarioVariant v :
         {ScenarioVariant::HostClient, ScenarioVariant::HostClientDelay, ScenarioVariant::HostClientAsync})
    {
        const ScenarioResult r = runScenario(scenarioSetup(), e, v);
        EXPECT_TRUE(r.passed) << r.report;
    }
}

// The write detector: a write to a component that belongs to a remote authority fails the variant
TEST(ScenarioRunner, TheWriteDetectorFindsHostWritesToOwnedComponents)
{
    // The scenario writes a client-owned player's input on the host after the spawn (it should use edit)
    auto direct = entry("host_writes_input",
                        [](ScenarioWorld& w)
                        {
                            auto player = w.spawnPlayer({0, 0, 0});
                            w.step(10);
                            w.world().get<PlayerInput>(player).heldButtons = 5;
                            w.step(10);
                        });
    EXPECT_TRUE(runScenario(scenarioSetup(), direct, ScenarioVariant::Plain).passed);
    ScenarioResult r = runScenario(scenarioSetup(), direct, ScenarioVariant::HostClient);
    EXPECT_FALSE(r.passed);
    EXPECT_TRUE(contains(r, "PlayerInput")) << r.report;
    EXPECT_TRUE(contains(r, "R-NET-2")) << r.report;
    // A host system that writes it every step
    WorldSetup s = scenarioSetup();
    s.systems    = [](SystemList& l)
    {
        l.add(Phase::Gameplay, "actionLog", actionLogSystem);
        l.add(Phase::Gameplay, "inputWriter", hostInputWriterSystem);
    };
    auto walk = entry("system_writes_input",
                      [](ScenarioWorld& w)
                      {
                          w.spawnPlayer({0, 0, 0});
                          w.step(20);
                      });
    EXPECT_TRUE(runScenario(s, walk, ScenarioVariant::Plain).passed);
    r = runScenario(s, walk, ScenarioVariant::HostClientDelay);
    EXPECT_FALSE(r.passed);
    EXPECT_TRUE(contains(r, "R-NET-2")) << r.report;
}

// The asynchronous variant is reproducible: two runs give equal host hashes and equal client states
TEST(ScenarioRunner, TheAsyncVariantIsReproducible)
{
    auto run = []()
    {
        ScenarioNetwork net;
        net.enabled = true;
        net.async   = true;
        ScenarioOptions o;
        o.clients = 2;
        ScenarioWorld w(scenarioSetup(), o, net);
        std::vector<uint64_t> hashes;
        w.control.hashes = &hashes;
        auto a           = w.spawnPlayer({0, 0, 0}, 1);
        auto b           = w.spawnPlayer({5, 0, 0}, 2);
        w.spawn<Wanderer>({2, 0, 2});
        w.move(a, Vec3(1, 0, 0));
        w.move(b, Vec3(0, 0, -1));
        w.step(60);
        w.finishNetwork();
        EXPECT_TRUE(w.failures().empty()) << (w.failures().empty() ? "" : w.failures()[0]);
        hashes.push_back(w.client(1).hash());
        hashes.push_back(w.client(2).hash());
        return hashes;
    };
    const auto first = run();
    EXPECT_EQ(first, run());
    EXPECT_GT(first.size(), 60u);
}

// Regression (CodeRabbit, S16): a client-owned player spawned after the last step was not in the final sync's list of
// owners, so the replicas were compared before the host received its owner state
TEST(ScenarioRunner, TheFinalSyncWaitsForTheOwnerStateOfALateSpawn)
{
    for (int delay : {0, 2})
    {
        ScenarioNetwork net;
        net.enabled    = true;
        net.delaySteps = delay;
        ScenarioWorld w(scenarioSetup(), ScenarioOptions{}, net);
        w.step(5);
        const Entity player = w.spawnPlayer({1, 0, 0});
        w.finishNetwork();
        EXPECT_TRUE(w.failures().empty()) << (w.failures().empty() ? "" : w.failures()[0]);
        EXPECT_EQ(w.world().ownerOf(player), 1);
        EXPECT_GE(w.world().replication().ownerStateSequence(1), 1u) << "delay " << delay;
    }
}

// Regression (CodeRabbit, S16): saveLoad in a host + client variant replaced the host world under connected clients
TEST(ScenarioRunner, SaveLoadInAHostClientVariantFails)
{
    auto e                 = entry("save_load_networked",
                                   [](ScenarioWorld& w)
                                   {
                       w.step(2);
                       w.saveLoad();
                   });
    const ScenarioResult r = runScenario(scenarioSetup(), e, ScenarioVariant::HostClient);
    EXPECT_FALSE(r.passed);
    EXPECT_TRUE(contains(r, "saveLoad is not supported in the host + client variants")) << r.report;
}
