/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Non-gating ECS benchmark (docs/engine_design.md R-TEST-2, stage S07): 10^5 entities with three components. Prints
// the times of the basic operations; compare before and after changes under engine/ecs/. The only checks are
// correctness checks (the timings depend on the machine and are not compared).

#include "saiga/engine/ecs/Registry.h"
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
