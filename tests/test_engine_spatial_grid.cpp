/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The spatial grid and World::queryRadius (docs/engine_design.md §9.5, stage S09)

#include "saiga/engine/components/LocalTransform.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/world/SpatialGridSystem.h"
#include "saiga/engine/world/World.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace Saiga;
using namespace Saiga::Engine;

namespace
{
enum class TestComponent : uint8_t
{
    Tag = kFirstGameComponent,
    Count
};

struct Tag
{
    static constexpr const char* name = "Tag";
    static constexpr ComponentId id   = ComponentId(TestComponent::Tag);
    static constexpr uint32_t version = 1;
    int value                         = 0;
    template <class V>
    void visit(V& v)
    {
        v("value", value);
    }
};

void testComponents(Registry& r)
{
    r.registerComponent<Tag>(Replicate::Never);
}

WorldSetup testSetup(void (*systems)(SystemList&) = nullptr)
{
    WorldSetup s;
    s.components = testComponents;
    s.systems    = systems;
    return s;
}

Entity createAt(World& w, const Vec3& p)
{
    Entity e = w.create();
    w.get<Transform>(e).setPosition(p);
    return e;
}

std::vector<Entity> sorted(std::vector<Entity> v)
{
    std::sort(v.begin(), v.end());
    return v;
}
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// SpatialGrid (the store)

TEST(SpatialGrid, FindsWithinTheRadiusAcrossCellsAndSpaces)
{
    SpatialGrid g;
    g.update(Entity{1, 0}, 0, Vec3(0, 0, 0));
    g.update(Entity{2, 0}, 0, Vec3(31.9, 0, 0));     // same cell as 1
    g.update(Entity{3, 0}, 0, Vec3(32.1, 0, 0));     // next cell
    g.update(Entity{4, 0}, 0, Vec3(-0.5, 0, -0.5));  // negative cell
    g.update(Entity{5, 0}, 1, Vec3(0, 0, 0));        // another space
    g.update(Entity{6, 0}, 0, Vec3(0, 40, 0));       // above: same column, too far in 3D
    EXPECT_EQ(g.size(), 6u);

    std::vector<Entity> found;
    g.forEachWithin(0, Vec3(0, 0, 0), 33, [&](const SpatialGrid::Item& it) { found.push_back(it.entity); });
    EXPECT_EQ(sorted(found), (std::vector<Entity>{{1, 0}, {2, 0}, {3, 0}, {4, 0}}));

    found.clear();
    g.forEachWithin(1, Vec3(0, 0, 0), 1, [&](const SpatialGrid::Item& it) { found.push_back(it.entity); });
    EXPECT_EQ(found, (std::vector<Entity>{{5, 0}}));
    // A space without entities
    g.forEachWithin(7, Vec3(0, 0, 0), 1e9, [&](const SpatialGrid::Item&) { ADD_FAILURE(); });
}

TEST(SpatialGrid, MoveRemoveAndReplace)
{
    SpatialGrid g;
    g.update(Entity{1, 0}, 0, Vec3(0, 0, 0));
    g.update(Entity{2, 0}, 0, Vec3(1, 0, 0));
    g.update(Entity{3, 0}, 0, Vec3(2, 0, 0));
    // Moving 1 to another cell swaps 3 into its slot of the first cell
    g.update(Entity{1, 0}, 0, Vec3(100, 0, 100));
    g.remove(2);
    g.remove(2);  // no-op
    EXPECT_EQ(g.size(), 2u);
    EXPECT_FALSE(g.contains(2));
    EXPECT_EQ(g.entityAt(3), (Entity{3, 0}));
    std::vector<Entity> found;
    g.forEachWithin(0, Vec3(0, 0, 0), 10, [&](const SpatialGrid::Item& it) { found.push_back(it.entity); });
    EXPECT_EQ(found, (std::vector<Entity>{{3, 0}}));
    // A newer handle of the same index replaces the old one (same cell: position and generation)
    g.update(Entity{3, 1}, 0, Vec3(3, 0, 0));
    EXPECT_EQ(g.size(), 2u);
    EXPECT_EQ(g.entityAt(3), (Entity{3, 1}));
    found.clear();
    g.forEachWithin(0, Vec3(3, 0, 0), 0, [&](const SpatialGrid::Item& it) { found.push_back(it.entity); });
    EXPECT_EQ(found, (std::vector<Entity>{{3, 1}}));
    g.clear();
    EXPECT_EQ(g.size(), 0u);
}

TEST(SpatialGrid, LargeInvalidAndFarQueries)
{
    SpatialGrid g;
    for (uint32_t i = 1; i <= 50; ++i) g.update(Entity{i, 0}, 0, Vec3(double(i) * 1000, 0, -double(i) * 700));
    g.update(Entity{51, 0}, 0, Vec3(std::numeric_limits<double>::quiet_NaN(), 0, 0));
    g.update(Entity{52, 0}, 0, Vec3(1e300, 0, -1e300));
    size_t n = 0;
    // A radius covering far more cells than are occupied visits the occupied cells
    g.forEachWithin(0, Vec3(0, 0, 0), 1e6, [&](const SpatialGrid::Item&) { n++; });
    EXPECT_EQ(n, 50u);
    n = 0;
    g.forEachWithin(0, Vec3(0, 0, 0), std::numeric_limits<double>::infinity(), [&](const SpatialGrid::Item&) { n++; });
    EXPECT_EQ(n, 51u);  // not the NaN position
    n = 0;
    g.forEachWithin(0, Vec3(1e300, 0, -1e300), 1, [&](const SpatialGrid::Item&) { n++; });
    EXPECT_EQ(n, 1u);  // a clamped far cell
    g.forEachWithin(0, Vec3(0, 0, 0), -1, [&](const SpatialGrid::Item&) { ADD_FAILURE(); });
    g.forEachWithin(0, Vec3(0, 0, 0), std::numeric_limits<double>::quiet_NaN(),
                    [&](const SpatialGrid::Item&) { ADD_FAILURE(); });
    g.forEachWithin(0, Vec3(std::numeric_limits<double>::quiet_NaN(), 0, 0), 10,
                    [&](const SpatialGrid::Item&) { ADD_FAILURE(); });
}

// ---------------------------------------------------------------------------------------------------------------------
// World::queryRadius

TEST(SpatialQuery, SortedByIndexWithAMask)
{
    World w(testSetup());
    // Created in an order where cells and indices disagree
    std::vector<Entity> e;
    for (int i = 0; i < 20; ++i) e.push_back(createAt(w, Vec3(double((i * 7) % 20) * 10 - 100, 0, 0)));
    for (int i = 0; i < 20; i += 3) w.add<Tag>(e[size_t(i)], Tag{i});
    w.step();

    auto all = w.queryRadius(0, Vec3(0, 0, 0), 1000);
    EXPECT_EQ(all, e);  // e is in index order
    auto tagged = w.queryRadius(0, Vec3(0, 0, 0), 1000, World::mask<Tag>());
    std::vector<Entity> expected;
    for (int i = 0; i < 20; i += 3) expected.push_back(e[size_t(i)]);
    EXPECT_EQ(tagged, expected);
    // The boundary is included
    auto exact = w.queryRadius(0, Vec3(-100, 0, 0), 0);
    ASSERT_EQ(exact.size(), 1u);
    EXPECT_EQ(w.read<Transform>(exact[0]).position(), Vec3(-100, 0, 0));
    // The out variant clears first
    std::vector<Entity> out = {Entity{99, 0}};
    w.queryRadius(0, Vec3(0, 0, 0), 1000, World::mask<Tag>(), out);
    EXPECT_EQ(out, expected);
}

TEST(SpatialQuery, PositionsOfTheLastPhase6)
{
    World w(testSetup());
    Entity a = createAt(w, Vec3(0, 0, 0));
    // Created between steps: not found until phase 6
    EXPECT_TRUE(w.queryRadius(0, Vec3(0, 0, 0), 1).empty());
    w.step();
    EXPECT_EQ(w.queryRadius(0, Vec3(0, 0, 0), 1), std::vector<Entity>{a});
    // Moved: found at the old position until phase 6
    w.get<Transform>(a).setPosition(Vec3(500, 0, 500));
    EXPECT_EQ(w.queryRadius(0, Vec3(0, 0, 0), 1), std::vector<Entity>{a});
    w.step();
    EXPECT_TRUE(w.queryRadius(0, Vec3(0, 0, 0), 1).empty());
    EXPECT_EQ(w.queryRadius(0, Vec3(500, 0, 500), 1), std::vector<Entity>{a});
    // Destroyed: never returned, also before phase 6, and not after the slot is reused
    w.destroy(a);
    EXPECT_TRUE(w.queryRadius(0, Vec3(500, 0, 500), 1).empty());
    Entity b = createAt(w, Vec3(0, 0, 0));
    EXPECT_EQ(b.index, a.index);
    EXPECT_TRUE(w.queryRadius(0, Vec3(500, 0, 500), 1).empty());
    w.step();
    EXPECT_TRUE(w.queryRadius(0, Vec3(500, 0, 500), 1).empty());
    EXPECT_EQ(w.queryRadius(0, Vec3(0, 0, 0), 1), std::vector<Entity>{b});
    EXPECT_EQ(w.spatialGrid().size(), 1u);
}

namespace
{
Entity g_created;
std::vector<Entity> g_seenInGameplay, g_seenInExtract;
bool g_createOnce = false;
void sysCreate(World& w, double)
{
    g_seenInGameplay = w.queryRadius(0, Vec3(10, 0, 10), 1);
    if (g_createOnce)
    {
        g_createOnce = false;
        g_created    = w.create();  // reserved now, activated at the end of phase 4
        w.add<Tag>(g_created, Tag{1});
    }
}
void sysPlace(World& w, double)
{
    // Phase 5: the entity is alive now; placed before phase 6
    if (w.alive(g_created) && w.read<Transform>(g_created).position() == Vec3::Zero())
        w.get<Transform>(g_created).setPosition(Vec3(10, 0, 10));
}
void sysExtract(World& w, double)
{
    g_seenInExtract = w.queryRadius(0, Vec3(10, 0, 10), 1, World::mask<Tag>());
}
void createDuringAStepSystems(SystemList& s)
{
    s.add(Phase::Gameplay, "create", sysCreate);
    s.add(Phase::Physics, "place", sysPlace);
    s.add(Phase::Extract, "extract", sysExtract);
}
}  // namespace

TEST(SpatialQuery, CreatedDuringAStepIsFoundAfterPhase6)
{
    World w(testSetup(createDuringAStepSystems));
    g_createOnce = true;
    w.step();
    EXPECT_TRUE(g_seenInGameplay.empty());
    EXPECT_EQ(g_seenInExtract, std::vector<Entity>{g_created});
    w.step();
    EXPECT_EQ(g_seenInGameplay, std::vector<Entity>{g_created});
}

TEST(SpatialQuery, ChildrenAreFoundWhereTheirParentMovedThem)
{
    World w(testSetup());
    Entity parent = createAt(w, Vec3(0, 0, 0));
    Entity child  = w.create();
    w.add<Parent>(child, Parent{parent});
    w.add<LocalTransform>(child, LocalTransform{Vec3(5, 0, 0), Quat::Identity(), Vec3::Ones()});
    w.step();
    EXPECT_EQ(w.queryRadius(0, Vec3(5, 0, 0), 0.5), std::vector<Entity>{child});
    w.get<Transform>(parent).setPosition(Vec3(1000, 0, 0));
    w.step();
    EXPECT_EQ(w.queryRadius(0, Vec3(1005, 0, 0), 0.5), std::vector<Entity>{child});
    EXPECT_TRUE(w.queryRadius(0, Vec3(5, 0, 0), 0.5).empty());
}

TEST(SpatialQuery, DestroyingAParentRemovesTheChildren)
{
    World w(testSetup());
    Entity parent = createAt(w, Vec3(0, 0, 0));
    Entity child  = w.create();
    w.add<Parent>(child, Parent{parent});
    w.step();
    EXPECT_EQ(w.queryRadius(0, Vec3(0, 0, 0), 1).size(), 2u);
    w.destroy(parent);
    EXPECT_TRUE(w.queryRadius(0, Vec3(0, 0, 0), 1).empty());
    w.step();
    EXPECT_EQ(w.spatialGrid().size(), 0u);
}

TEST(SpatialQuery, SameHistorySameResults)
{
    // Two worlds with the same seed and history return the same entities in the same order after every step
    auto run = [](std::vector<std::vector<Entity>>& results)
    {
        World w(testSetup());
        std::vector<Entity> live;
        for (int stepIndex = 0; stepIndex < 30; ++stepIndex)
        {
            for (int k = 0; k < 20; ++k)
            {
                const double x = w.random().uniform(-200.0, 200.0), z = w.random().uniform(-200.0, 200.0);
                live.push_back(createAt(w, Vec3(x, 0, z)));
            }
            for (int k = 0; k < 5 && !live.empty(); ++k)
            {
                size_t i = size_t(w.random().uniform(0.0, double(live.size()) - 0.5));
                w.destroy(live[i]);
                live.erase(live.begin() + std::ptrdiff_t(i));
            }
            for (Entity e : live)
            {
                Vec3 p = w.read<Transform>(e).position();
                w.get<Transform>(e).setPosition(p + Vec3(w.random().uniform(-10.0, 10.0), 0, 0));
            }
            w.step();
            results.push_back(w.queryRadius(0, Vec3(0, 0, 0), 120));
        }
    };
    std::vector<std::vector<Entity>> a, b;
    run(a);
    run(b);
    EXPECT_EQ(a, b);
    // Each result is sorted, unique and matches a brute-force check
    for (auto& r : a)
    {
        EXPECT_TRUE(std::is_sorted(r.begin(), r.end()));
        EXPECT_TRUE(std::adjacent_find(r.begin(), r.end()) == r.end());
    }
}

TEST(SpatialQuery, MatchesBruteForce)
{
    World w(testSetup());
    std::vector<Entity> all;
    for (int i = 0; i < 2000; ++i)
    {
        const double x = w.random().uniform(-300.0, 300.0), y = w.random().uniform(-50.0, 50.0),
                     z = w.random().uniform(-300.0, 300.0);
        all.push_back(createAt(w, Vec3(x, y, z)));
    }
    w.step();
    for (int q = 0; q < 50; ++q)
    {
        const Vec3 c(w.random().uniform(-300.0, 300.0), 0, w.random().uniform(-300.0, 300.0));
        const double r = w.random().uniform(0.0, 150.0);
        std::vector<Entity> expected;
        for (Entity e : all)
            if ((w.read<Transform>(e).position() - c).squaredNorm() <= r * r) expected.push_back(e);
        EXPECT_EQ(w.queryRadius(0, c, r), expected);
    }
}

TEST(SpatialQuery, RebuildBetweenSteps)
{
    World w(testSetup());
    Entity a = createAt(w, Vec3(3, 0, 3));
    rebuildSpatialGrid(w);
    EXPECT_EQ(w.queryRadius(0, Vec3(3, 0, 3), 0), std::vector<Entity>{a});
    // The log records consumed by the rebuild are not applied again
    w.step();
    EXPECT_EQ(w.spatialGrid().size(), 1u);
    EXPECT_EQ(w.queryRadius(0, Vec3(3, 0, 3), 0), std::vector<Entity>{a});
}

TEST(SpatialQuery, IdleEntitiesCauseNoGridWork)
{
    World w(testSetup());
    for (int i = 0; i < 100; ++i) createAt(w, Vec3(double(i), 0, 0));
    w.step();
    w.step();
    // Unchanged for a full step: not in the changed list, so the grid is not touched
    w.step();
    EXPECT_TRUE(w.changedTransforms().empty());
    EXPECT_EQ(w.queryRadius(0, Vec3(50, 0, 0), 1000).size(), 100u);
}

namespace
{
std::vector<size_t> g_parallelCounts;
void sysParallelQuery(World& w, double)
{
    // Queries from parallelEach bodies (reads only); each entity writes only its own result slot
    w.view<const Transform, Tag>().parallelEach(
        [&](Entity, const Transform& t, Tag& tag) { tag.value = int(w.queryRadius(0, t.position(), 15).size()); }, 16);
}
void parallelQuerySystems(SystemList& s)
{
    s.add(Phase::Gameplay, "parallel_query", sysParallelQuery);
}
}  // namespace

TEST(SpatialQuery, QueriesInParallelEach)
{
    for (int threads : {1, 4})
    {
        WorldSetup s    = testSetup(parallelQuerySystems);
        s.workerThreads = threads;
        World w(s);
        for (int i = 0; i < 400; ++i) w.add<Tag>(createAt(w, Vec3(double(i % 40) * 10, 0, double(i / 40) * 10)), Tag{});
        w.step();  // the grid is built in phase 6
        w.step();
        std::vector<size_t> counts;
        w.viewAll<const Tag>().each([&](Entity, const Tag& t) { counts.push_back(size_t(t.value)); });
        if (threads == 1)
            g_parallelCounts = counts;
        else
            EXPECT_EQ(counts, g_parallelCounts);
        // An interior entity has its 8 neighbours (diagonal 14.1 m) and itself within 15 m
        EXPECT_EQ(*std::max_element(counts.begin(), counts.end()), 9u);
    }
}
