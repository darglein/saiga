/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Non-gating ECS benchmark (docs/engine_design.md R-TEST-2, stage S07): 10^5 entities with three components. Prints
// the times of the basic operations; compare before and after changes under engine/ecs/. The only checks are
// correctness checks (the timings depend on the machine and are not compared).

#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/RenderInstance.h"
#include "saiga/engine/ecs/Registry.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/gameplay/GameObject.h"
#include "saiga/engine/save/SaveDiff.h"
#include "saiga/engine/sync/ExtractSystem.h"
#include "saiga/engine/sync/RecordingRenderSink.h"
#include "saiga/engine/world/World.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace Saiga;
using namespace Saiga::Engine;

namespace
{
enum class BenchComponent : uint8_t
{
    Vitality = kFirstGameComponent,
    Velocity,
    Lifetime,
    Count
};

struct Vitality
{
    static constexpr const char* name = "Vitality";
    static constexpr ComponentId id   = ComponentId(BenchComponent::Vitality);
    static constexpr uint32_t version = 1;
    double hp = 100, maxHp = 100;
    template <class V>
    void visit(V& v)
    {
        v("hp", hp);
        v("max_hp", maxHp);
    }
};

struct Velocity
{
    static constexpr const char* name = "Velocity";
    static constexpr ComponentId id   = ComponentId(BenchComponent::Velocity);
    static constexpr uint32_t version = 1;
    Vec3 v                            = Vec3::Zero();
    template <class V>
    void visit(V& vis)
    {
        vis("v", v);
    }
};

struct Lifetime
{
    static constexpr const char* name = "Lifetime";
    static constexpr ComponentId id   = ComponentId(BenchComponent::Lifetime);
    static constexpr uint32_t version = 1;
    double remaining                  = 10;
    template <class V>
    void visit(V& v)
    {
        v("remaining", remaining);
    }
};

struct Timer
{
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    double ms() const
    {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
};
}  // namespace

TEST(EcsBenchmark, HundredThousandEntities)
{
    const int n = 100000;
    Registry r;
    r.registerComponent<Vitality>();
    r.registerComponent<Velocity>();
    r.registerComponent<Lifetime>();

    std::vector<Entity> entities;
    entities.reserve(n);
    Timer t;
    for (int i = 0; i < n; ++i)
    {
        Entity e = r.create();
        r.add<Vitality>(e);
        r.add<Velocity>(e, Vec3(i, 0, 1));
        r.add<Lifetime>(e);
        entities.push_back(e);
    }
    double create = t.ms();

    t          = Timer();
    double sum = 0;
    r.view<const Vitality>().each([&](Entity, const Vitality& h) { sum += h.hp; });
    double view1 = t.ms();

    t = Timer();
    r.view<Transform, const Velocity>().each([](Entity, Transform& tr, const Velocity& v)
                                             { tr.setPosition(tr.position() + v.v * (1.0 / 60.0)); });
    double view2 = t.ms();

    t = Timer();
    r.view<Vitality, const Velocity, Lifetime>().each(
        [](Entity, Vitality& h, const Velocity& v, Lifetime& l)
        {
            l.remaining -= 1.0 / 60.0;
            h.hp -= v.v(2) * 0.001;
        });
    double view3 = t.ms();

    t           = Timer();
    uint64_t x  = 12345;
    double rsum = 0;
    for (int i = 0; i < n; ++i)
    {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        rsum += r.read<Velocity>(entities[size_t((x >> 33) % uint64_t(n))]).v(0);
    }
    double randomGet = t.ms();

    t = Timer();
    for (int i = 0; i < n; i += 2) r.remove<Lifetime>(entities[size_t(i)]);
    for (int i = 0; i < n; i += 2) r.add<Lifetime>(entities[size_t(i)]);
    double readd = t.ms();

    t = Timer();
    for (Entity e : entities) r.destroy(e);
    double destroy = t.ms();

    std::printf(
        "ECS benchmark, %d entities with 3 components (ms): create %.2f, view<1> %.2f, view<Transform, 1> %.2f, "
        "view<3> %.2f, random read %.2f, remove + re-add on half %.2f, destroy %.2f\n",
        n, create, view1, view2, view3, randomGet, readd, destroy);

    // Correctness only
    EXPECT_EQ(sum, 100.0 * n);
    EXPECT_GT(rsum, 0.0);
    EXPECT_EQ(r.aliveCount(), 0u);
}


namespace
{
void moveSystem(World& w, double dt)
{
    w.view<Transform, const Velocity>().each([dt](Entity, Transform& t, const Velocity& v)
                                             { t.setPosition(t.position() + v.v * dt); });
}
void benchSystems(SystemList& s)
{
    s.add(Phase::Gameplay, "move", moveSystem);
}
void benchComponents(Registry& r)
{
    r.registerComponent<Vitality>();
    r.registerComponent<Velocity>();
    r.registerComponent<Lifetime>();
}
}  // namespace

// World::step with 10^5 entities (R-TEST-2 for engine/ecs and the world): idle (only the engine's phase-6 scan) and
// with a system that moves all of them
TEST(EcsBenchmark, WorldStepHundredThousandEntities)
{
    const int n = 100000;
    WorldSetup idleSetup;
    idleSetup.components = benchComponents;
    World idle(idleSetup);
    for (int i = 0; i < n; ++i) idle.create();
    idle.step();
    Timer t;
    idle.step(10);
    double idleMs = t.ms() / 10;

    WorldSetup moveSetup = idleSetup;
    moveSetup.systems    = benchSystems;
    World moving(moveSetup);
    for (int i = 0; i < n; ++i) moving.add<Velocity>(moving.create(), Vec3(1, 0, 0));
    moving.step();
    t = Timer();
    moving.step(10);
    double moveMs = t.ms() / 10;

    std::printf("ECS benchmark, World::step with %d entities (ms per step): idle %.3f, moving %.3f\n", n, idleMs,
                moveMs);
    EXPECT_EQ(idle.registry().aliveCount(), uint32_t(n));
    EXPECT_EQ(moving.changedTransforms().size(), size_t(n));
}

namespace
{
void benchRenderAssets(World& w)
{
    auto lods = std::make_shared<Vulkan::MeshLods>();
    Vulkan::MeshLod lod;
    lod.vertices.resize(3);
    lod.indices = {0, 1, 2};
    lods->lods.push_back(lod);
    w.registerMesh("bench/box", lods, w.registerMaterial("bench/flat", Vulkan::MaterialData()));
}
}  // namespace

// S19b (R-TEST-2, §17.10): the extract of 10^4 moving entities with RenderInstance into a recording sink (the step's
// extract scope), the same entities idle (changes only: no per-step work), and a paused frame's catch-up with nothing
// changed (O(tracked) bitwise comparisons). Main-thread budget: steps + extract + recording <= 10 ms per frame.
TEST(EcsBenchmark, ExtractTenThousandEntities)
{
    const int n = 10000;
    WorldSetup s;
    s.components   = benchComponents;
    s.systems      = benchSystems;
    s.renderAssets = benchRenderAssets;
    World w(s);
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    std::vector<Entity> entities;
    for (int i = 0; i < n; ++i)
    {
        const Entity e = w.create();
        w.get<Transform>(e).setPosition(Vec3(i % 100, 0, i / 100));
        w.add<Velocity>(e, Vec3(1, 0, 0));
        w.add<RenderInstance>(e, RenderInstance("bench/box"));
        entities.push_back(e);
    }
    w.step(2);
    Profiler& prof   = w.profiler();
    const int scope  = prof.scope("step/phase:extract/extract");
    const int steps  = prof.scope("step");
    const int frames = 120;
    w.step(frames);
    const Profiler::Stats moving = prof.scopeStats(scope, frames), movingStep = prof.scopeStats(steps, frames);
    const uint64_t setsMoving = sink.counters().setDynamic;
    // Idle: the stop pairs of the next step, then nothing per step
    for (Entity e : entities) w.get<Velocity>(e).v = Vec3::Zero();
    w.step(2);
    const uint64_t setsBefore = sink.counters().setDynamic;
    w.step(frames);
    const Profiler::Stats idle = prof.scopeStats(scope, frames);
    EXPECT_EQ(sink.counters().setDynamic, setsBefore);
    // Paused frames without changes
    w.setPaused(true);
    Timer t;
    for (int i = 0; i < frames; ++i) w.extract().catchUp();
    const double catchUpMs = t.ms() / frames;
    EXPECT_EQ(sink.counters().setDynamic, setsBefore);
    std::printf(
        "ECS benchmark, extract of %d rendered entities (ms): moving: extract %.3f (median, p95 %.3f) of a step %.3f; "
        "idle: extract %.4f; paused catch-up without changes %.3f per frame\n",
        n, moving.p50, moving.p95, movingStep.p50, idle.p50, catchUpMs);
    EXPECT_EQ(sink.liveDynamics(), size_t(n));
    EXPECT_GE(setsMoving, uint64_t(n) * frames);
    w.setRenderSink(nullptr);
}

namespace
{
void emptySystem(World&, double) {}
void ownedMoveSystem(World& w, double dt)
{
    w.ownedBy<Transform, const Velocity>(w.localPeer())
        .each([dt](Entity, Transform& t, const Velocity& v) { t.setPosition(t.position() + v.v * dt); });
}
}  // namespace

// S16 (R-TEST-2): ownedBy (one owner lookup per candidate) against view over 10^5 entities, half of them owned by a
// client; and the RunOn filter of the step loop (100 systems, half of them client-only)
TEST(EcsBenchmark, OwnedByAndRunOn)
{
    const int n = 100000;
    WorldSetup setup;
    setup.components = benchComponents;
    World w(setup);
    for (int i = 0; i < n; ++i)
    {
        const Entity e = w.create();
        w.add<Velocity>(e, Vec3(1, 0, 0));
        if (i % 2) w.setOwner(e, 1);
    }
    w.ownerOf(Entity{});  // the owner table is built once (after the structural changes)
    Timer t;
    size_t viewed = 0, owned = 0;
    for (int k = 0; k < 10; ++k) w.view<const Velocity>().each([&](Entity, const Velocity&) { viewed++; });
    const double viewMs = t.ms() / 10;
    t                   = Timer();
    for (int k = 0; k < 10; ++k) w.ownedBy<const Velocity>(0).each([&](Entity, const Velocity&) { owned++; });
    const double ownedMs = t.ms() / 10;
    t                    = Timer();
    for (int k = 0; k < 10; ++k) w.ownedBy<const Transform>(1).each([&](Entity, const Transform&) { owned++; });
    const double ownedTransformMs = t.ms() / 10;
    EXPECT_EQ(viewed, size_t(10 * n));
    EXPECT_EQ(owned, size_t(10 * n));

    WorldSetup systems = setup;
    systems.systems    = [](SystemList& l)
    {
        static const char* names[100];
        static std::vector<std::string> storage;
        for (int i = 0; i < 100; ++i) storage.push_back("empty" + std::to_string(i));
        for (int i = 0; i < 100; ++i) names[i] = storage[size_t(i)].c_str();
        for (int i = 0; i < 100; ++i)
            l.add(Phase::Gameplay, names[i], emptySystem, i % 2 ? RunOn::Client : RunOn::Host);
        l.add(Phase::Gameplay, "owned_move", ownedMoveSystem, RunOn::Both);
    };
    World filtered(systems);
    filtered.step();
    t = Timer();
    filtered.step(1000);
    const double stepUs = t.ms();  // per 1000 steps in ms = per step in us
    std::printf(
        "ECS benchmark, ownedBy over %d entities (ms): view<1> %.3f, ownedBy<1>(0) %.3f, ownedBy<Transform>(1) %.3f; "
        "an empty world with 100 systems (50 skipped by RunOn): %.2f us per step\n",
        n, viewMs, ownedMs, ownedTransformMs, stepUs);
}

// The spatial grid (S09, R-TEST-2): World::step with 10^5 entities spread over 2 km x 2 km moving at 20 m/s (the grid
// updates every changed transform, some change cells), and 10^4 queryRadius calls of 50 m (about 200 hits each)
TEST(EcsBenchmark, SpatialGridHundredThousandEntities)
{
    const int n = 100000;
    WorldSetup setup;
    setup.components = benchComponents;
    setup.systems    = benchSystems;
    World w(setup);
    for (int i = 0; i < n; ++i)
    {
        Entity e = w.create();
        w.get<Transform>(e).setPosition(
            Vec3(w.random().uniform(-1000.0, 1000.0), 0, w.random().uniform(-1000.0, 1000.0)));
        const double a = w.random().uniform(0.0, 6.283185307179586);
        w.add<Velocity>(e, Vec3(20 * std::cos(a), 0, 20 * std::sin(a)));
    }
    w.step();
    Timer t;
    w.step(10);
    double stepMs = t.ms() / 10;

    std::vector<Entity> out;
    size_t hits = 0;
    t           = Timer();
    for (int q = 0; q < 10000; ++q)
    {
        const Vec3 c(w.random().uniform(-1000.0, 1000.0), 0, w.random().uniform(-1000.0, 1000.0));
        w.queryRadius(0, c, 50, ComponentMask(), out);
        hits += out.size();
    }
    double queryMs = t.ms();

    std::printf(
        "ECS benchmark, spatial grid with %d moving entities: World::step %.3f ms, 10^4 queryRadius(50 m) %.3f ms "
        "(%.1f hits per query)\n",
        n, stepMs, queryMs, double(hits) / 10000);
    EXPECT_EQ(w.spatialGrid().size(), size_t(n));
    EXPECT_GT(hits, 0u);
}

namespace
{
// A game object that does a little work per update (the Behaviour state counts the updates)
class BenchObject final : public GameObject
{
   public:
    static constexpr ClassId id       = kFirstGameClass;
    static constexpr const char* name = "BenchObject";
    void update(double dt) override
    {
        Transform& t = transformWrite();
        t.setPosition(t.position() + Vec3(0, 0, 0.1 * dt));
    }
};
}  // namespace

// Game objects (S10, R-TEST-2): World::step with 10^4 objects spread over 2 km x 2 km and one player (ticking by
// grid-cell rings: the ones near the player every step, the others every 0.1 or 0.5 s), with the player standing and
// with the player crossing a cell border every step (every object re-tiered)
TEST(EcsBenchmark, GameObjectsAndTicking)
{
    const int n = 10000;
    WorldSetup setup;
    setup.classes = [](ClassList& c) { c.add<BenchObject>(); };
    World w(setup);
    Entity player = w.create();
    w.add<Player>(player);
    for (int i = 0; i < n; ++i)
        w.spawn<BenchObject>(Vec3(w.random().uniform(-1000.0, 1000.0), 0, w.random().uniform(-1000.0, 1000.0)));
    w.step(60);
    w.profiler().reset();
    Timer t;
    w.step(60);
    const double standMs = t.ms() / 60;
    const double updated = w.profiler().counterStats(w.profiler().counter("objects_updated")).avg;
    t                    = Timer();
    for (int s = 0; s < 60; ++s)
    {
        w.get<Transform>(player).setPosition(Vec3(double(s) * 33, 0, 0));
        w.step();
    }
    const double crossMs = t.ms() / 60;
    std::printf(
        "ECS benchmark, %d game objects (1 player): World::step %.3f ms standing (%.0f updates per step), %.3f ms "
        "crossing a cell border every step\n",
        n, standMs, updated, crossMs);
    EXPECT_GT(updated, 0);
}

namespace
{
// Vitality with a changed meaning (a migration), for the slow load path
struct VitalityV2
{
    static constexpr const char* name = "Vitality";
    static constexpr ComponentId id   = ComponentId(BenchComponent::Vitality);
    static constexpr uint32_t version = 2;
    double hp = 100, maxHp = 100;
    template <class V>
    void visit(V& v)
    {
        v("hp", hp);
        v("max_hp", maxHp);
    }
};
void saveComponents(Registry& r)
{
    r.registerComponent<Vitality>();
    r.registerComponent<Velocity>();
    r.registerComponent<Lifetime>();
}
void saveComponentsV2(Registry& r)
{
    r.registerComponent<VitalityV2>();
    r.registerComponent<Velocity>();
    r.registerComponent<Lifetime>();
}
}  // namespace

// Saves (S11, R-TEST-2): save and load of 10^5 entities (Transform + 3 components; 10^4 of them game objects), the
// load with one migrated component (the slow path by name), and the comparison of two saves with one difference
TEST(EcsBenchmark, SaveAndLoadHundredThousandEntities)
{
    const int n = 100000, objects = 10000;
    WorldSetup setup;
    setup.components = saveComponents;
    setup.classes    = [](ClassList& c) { c.add<BenchObject>(); };
    World w(setup);
    for (int i = 0; i < n; ++i)
    {
        const Vec3 p(w.random().uniform(-1000.0, 1000.0), 0, w.random().uniform(-1000.0, 1000.0));
        const Entity e = i < objects ? w.spawn<BenchObject>(p) : w.create();
        if (i >= objects) w.get<Transform>(e).setPosition(p);
        w.add<Vitality>(e);
        w.add<Velocity>(e);
        w.add<Lifetime>(e);
    }
    Timer t;
    const std::vector<uint8_t> blob = w.save();
    const double saveMs             = t.ms();

    World loaded(setup);
    t = Timer();
    ASSERT_TRUE(loaded.tryLoad(blob).ok());
    const double loadMs = t.ms();

    WorldSetup v2 = setup;
    v2.components = saveComponentsV2;
    v2.migrations = [](MigrationList& m) { m.add<VitalityV2>(1, [](FieldMap&) {}); };
    World migrated(v2);
    t = Timer();
    ASSERT_TRUE(migrated.tryLoad(blob).ok());
    const double migrateMs = t.ms();

    loaded.get<Vitality>(loaded.registry().aliveAt(n / 2)).hp = 1;
    const std::vector<uint8_t> changed                        = loaded.save();
    t                                                         = Timer();
    const SaveDiff d                                          = diffSaves(blob, changed);
    const double diffMs                                       = t.ms();
    EXPECT_EQ(d.total, 1u);
    std::printf(
        "ECS benchmark, saves of %d entities (%d game objects, %.1f MB): save %.1f ms, load %.1f ms, load with one "
        "migrated component %.1f ms, diff %.1f ms\n",
        n, objects, double(blob.size()) / 1e6, saveMs, loadMs, migrateMs, diffMs);
}
