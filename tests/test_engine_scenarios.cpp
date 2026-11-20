/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Engine scenarios with test-only game objects, run in the plain, repeat and save/load variants, and the tests of the
// scenario runner itself (docs/engine_design.md §17.2, §17.4; stage S12)

#include "saiga/engine/components/CharacterIntent.h"
#include "saiga/engine/components/Health.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/PlayerInput.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/gameplay/GameObject.h"
#include "saiga/engine/tools/ScenarioRunner.h"
#include "saiga/engine/world/World.h"

#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <thread>

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
    Count
};
enum class TestClass : uint16_t
{
    Wanderer = kFirstGameClass,
    Counter,
    Recorder,
    Lazy,
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
class Lazy final : public GameObject
{
   public:
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
// Writes whether a worker thread ran the entity: the result depends on the thread count (caught by varyThreads)
std::atomic<bool> g_workerRan{false};
int g_threads = 1;
void workerFlagSystem(World& w, double)
{
    g_workerRan = false;
    g_threads   = w.workerThreads();
    w.view<WorkerFlag>().parallelEach(
        [](Entity, WorkerFlag& f)
        {
            const bool worker = std::this_thread::get_id() != g_mainThread;
            if (worker) g_workerRan = true;
            // The caller waits (up to 1 s) until a worker took a chunk, so a run with workers always has some
            if (!worker && g_threads > 1)
            {
                const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(1);
                while (!g_workerRan && std::chrono::steady_clock::now() < until) std::this_thread::yield();
            }
            f.onWorker = worker;
        });
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
              w.check(std::abs(x - 2 * speed) < 0.2, "walked 2 s at moveSpeed: x = " + std::to_string(x));
              w.move(player, Vec3::Zero());
              w.step(10);
              // At most the step that was already under way
              w.check(std::abs(w.world().read<Transform>(player).position().x() - x) <= speed / 60 + 1e-9, "stops");
          });
    s.add("action_reaches_listener",
          [](ScenarioWorld& w)
          {
              auto player = w.spawnPlayer({0, 0, 0});
              w.input(player, 7);
              w.input(player, 8);
              w.step();
              const auto& log = w.world().readGlobal<ActionLog>().actions;
              w.check(log == std::vector<uint32_t>{7, 8}, "delivered in the next step in queue order");
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

struct Case
{
    std::string scenario;
    ScenarioVariant variant;
};
std::vector<Case> allCases()
{
    ScenarioList list;
    registerEngineScenarios(list);
    std::vector<Case> cases;
    for (const auto& e : list.entries())
        for (ScenarioVariant v : {ScenarioVariant::Plain, ScenarioVariant::Repeat, ScenarioVariant::SaveLoad})
            cases.push_back(Case{e.name, v});
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
    registerEngineScenarios(list);
    const ScenarioList::Entry* entry = list.find(GetParam().scenario);
    ASSERT_TRUE(entry);
    ScenarioOptions options;
    // The engine's own systems must not depend on the thread count, and continue identically after a load
    options.varyThreads        = true;
    options.strictContinuation = true;
    const ScenarioResult r     = runScenario(scenarioSetup(), *entry, GetParam().variant, options);
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
    const auto results = runScenarios(scenarioSetup(), list, "persistent");
    ASSERT_EQ(results.size(), 3u);
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
