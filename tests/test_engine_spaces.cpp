/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Spaces (docs/engine_design.md §7.4, §7.8, §7.9, §12.4; S21): several space templates, spawning in every space,
// moveToSpace, space-state entities per space, saves with spaces and the transitions through entrances.

#include "saiga/engine/components/CharacterIntent.h"
#include "saiga/engine/components/LocalTransform.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/core/Hash.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/gameplay/GameObject.h"
#include "saiga/engine/net/LoopbackTransport.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/world/World.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "terrain_test_hooks.h"

using namespace Saiga;
using namespace Saiga::Engine;
using namespace TerrainTestHooks;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);

// A second cheap integer hook (space 2): a slope along z and a seed offset
int32_t otherTestHook(uint64_t seed, int64_t x, int64_t z)
{
    return int32_t(int64_t(seed % 500) * 32 - x * 2 + z * 12 + 512);
}

enum class SpaceTestComponent : uint8_t
{
    Zone = kFirstGameComponent,
    Count
};

// A space-state component; its default constructor counts (staged defaults per space are distinct)
int g_zoneCounter = 0;
struct Zone
{
    static constexpr const char* name = "Zone";
    static constexpr ComponentId id   = ComponentId(SpaceTestComponent::Zone);
    static constexpr uint32_t version = 1;
    int32_t serial                    = ++g_zoneCounter;
    double danger                     = 0;
    template <class V>
    void visit(V& v)
    {
        v("serial", serial);
        v("danger", danger);
    }
};

// The three spaces of the S21 tests: 0 open world (integerTestHook, 2 m), 1 the enclosed "cellar" (4 chunks), 2 open
// world (otherTestHook, 1 m)
std::vector<SpaceTemplate> threeSpaces()
{
    SpaceTemplate a = testSpaceTemplate(16, 2, integerTestHook);
    a.name          = "overworld";
    SpaceTemplate c = testSpaceTemplate(16, 1, otherTestHook);
    c.name          = "caves";
    return {a, SpaceTemplate::enclosed("cellar", 4), c};
}

enum class SpaceTestClass : uint16_t
{
    Marker = kFirstGameClass,
    Breeder,
    Diver,
    Count
};

// A plain object
class Marker final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(SpaceTestClass::Marker);
    static constexpr const char* name = "Marker";
};

// The Markers spawned by Breeders (test-only)
std::vector<Entity> g_bred;

// Spawns one Marker next to itself in every update (GameObject::spawn: in its own space)
class Breeder final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(SpaceTestClass::Breeder);
    static constexpr const char* name = "Breeder";
    void update(double) override { g_bred.push_back(spawn<Marker>(transform().position() + Vec3(1, 0, 0))); }
};

// Test systems: phase 4 and phase 7 (after the transform phase) run these when set
std::function<void(World&)> g_phase4, g_phase7;

// Moves itself to space 1 in its first update (during a step)
class Diver final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(SpaceTestClass::Diver);
    static constexpr const char* name = "Diver";
    void update(double) override
    {
        if (transform().space() == 0) world().moveToSpace(entity(), 1, Vec3(641, 0, 640), Quat::Identity());
    }
};

WorldSetup spacesSetup(uint64_t seed = 9)
{
    WorldSetup s;
    s.seed                  = seed;
    s.profiling             = false;
    s.config.spaceTemplates = threeSpaces();
    s.classes               = [](ClassList& c)
    {
        c.add<Marker>();
        c.add<Breeder>();
        c.add<Diver>();
    };
    s.systems = [](SystemList& l)
    {
        l.add(Phase::Gameplay, "test_phase4",
              [](World& w, double)
              {
                  if (g_phase4) g_phase4(w);
              });
        l.add(Phase::Streaming, "test_phase7",
              [](World& w, double)
              {
                  if (g_phase7) g_phase7(w);
              });
    };
    return s;
}

// Clears the test hooks at the end of a test
struct HookGuard
{
    ~HookGuard()
    {
        g_phase4 = nullptr;
        g_phase7 = nullptr;
    }
};

Vec3 composed(const Transform& parent, const LocalTransform& local)
{
    return parent.position() + parent.rotation() * parent.scale().cwiseProduct(local.position);
}

// A root with a child and a grandchild (LocalTransforms)
struct Family
{
    Entity root, child, grandchild;
};
Family makeFamily(World& w, const Vec3& p)
{
    Family f;
    f.root = w.create();
    w.get<Transform>(f.root).teleport(p, Quat::Identity());
    f.child = w.create();
    LocalTransform l;
    l.position = Vec3(1, 0, 0);
    w.add<LocalTransform>(f.child, l);
    w.add<Parent>(f.child, Parent{f.root});
    f.grandchild = w.create();
    l.position   = Vec3(0, 0, 2);
    w.add<LocalTransform>(f.grandchild, l);
    w.add<Parent>(f.grandchild, Parent{f.child});
    return f;
}
}  // namespace

// ---- configuration (Scope 1)

TEST(SpaceConfig, CountTemplateAndSeed)
{
    World w(spacesSetup(41));
    ASSERT_EQ(w.spaceCount(), 3u);
    EXPECT_EQ(w.spaceTemplate(0).name, "overworld");
    EXPECT_EQ(w.spaceTemplate(1).name, "cellar");
    EXPECT_EQ(w.spaceTemplate(2).name, "caves");
    EXPECT_EQ(w.spaceTemplate(0).kind, SpaceKind::OpenWorld);
    EXPECT_EQ(w.spaceTemplate(1).kind, SpaceKind::Enclosed);
    EXPECT_FALSE(w.spaceTemplate(1).hasTerrain());
    EXPECT_FALSE(w.spaceTemplate(1).seaLevel.has_value());
    EXPECT_EQ(w.spaceTemplate(1).sizeChunks, 4);
    EXPECT_EQ(w.spaceSeed(0), 41u);
    EXPECT_EQ(w.spaceSeed(1), hashCombine(41, 1));
    EXPECT_EQ(w.spaceSeed(2), hashCombine(41, 2));
    EXPECT_EQ(SpaceTemplate::enclosedOrigin(4), 640);
    EXPECT_EQ(SpaceTemplate::enclosedOrigin(1), 128);
    EXPECT_EQ(SpaceTemplate::enclosedOrigin(5), 640);
}

// D1: an open-world template without a hook is valid and flat
TEST(SpaceConfig, HooklessOpenWorldIsFlat)
{
    WorldSetup s;
    s.profiling = false;
    SpaceTemplate flat;
    flat.name               = "flat";
    s.config.spaceTemplates = {testSpaceTemplate(16, 2, integerTestHook), flat};
    World w(s);
    EXPECT_FALSE(w.spaceTemplate(1).hasTerrain());
    EXPECT_EQ(w.terrainHeight(1, 100.5, 2000.25), 0.0);
    EXPECT_NE(w.terrainHeight(0, 100.5, 2000.25), 0.0);
}

// An enclosed template written by hand: S20's default seaLevel = 0 and baseSpacing are ignored, not checked
TEST(SpaceConfig, HandWrittenEnclosedIsValid)
{
    WorldSetup s;
    s.profiling = false;
    SpaceTemplate t;
    t.name                  = "vault";
    t.kind                  = SpaceKind::Enclosed;
    t.baseSpacing           = 7;
    t.sizeChunks            = 3;
    s.config.spaceTemplates = {SpaceTemplate(), t};
    World w(s);
    EXPECT_EQ(w.spaceCount(), 2u);
    EXPECT_EQ(w.terrainHeight(1, 10, 10), 0.0);
}

TEST(SpaceConfigDeathTest, InvalidTemplates)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    auto with = [](std::vector<SpaceTemplate> list)
    {
        WorldSetup s;
        s.profiling             = false;
        s.config.spaceTemplates = std::move(list);
        World w(s);
    };
    SpaceTemplate a;
    a.name          = "a";
    SpaceTemplate b = a;
    EXPECT_DEATH(with({a, b}), "space template 'a': the name is used by another space template");
    SpaceTemplate empty;
    empty.name = "";
    EXPECT_DEATH(with({a, empty}), "empty name");
    SpaceTemplate hooked   = SpaceTemplate::enclosed("cellar");
    hooked.prototypeHeight = integerTestHook;
    EXPECT_DEATH(with({a, hooked}),
                 "space template 'cellar': an enclosed space has no terrain .*SpaceTemplate::enclosed");
    EXPECT_DEATH(with({a, SpaceTemplate::enclosed("cellar", 0)}),
                 "sizeChunks of an enclosed space must be in 1..512 \\(is 0\\)");
    EXPECT_DEATH(with({a, SpaceTemplate::enclosed("cellar", 513)}),
                 "sizeChunks of an enclosed space must be in 1..512 \\(is 513\\)");
}

TEST(SpaceConfig, TerrainHeightPerSpace)
{
    World w(spacesSetup());
    const uint64_t seed0 = w.spaceSeed(0), seed2 = w.spaceSeed(2);
    // Sample points (the hooks take metres): space 0 at 2 m spacing, space 2 at 1 m
    for (int64_t i : {3, 17, 250, 1000})
    {
        const int64_t j = i * 3 % 1500 + 5;
        EXPECT_EQ(w.terrainHeight(0, double(i * 2), double(j * 2)), integerTestHook(seed0, i * 2, j * 2) * kHeightUnit);
        EXPECT_EQ(w.terrainHeight(2, double(i), double(j)), otherTestHook(seed2, i, j) * kHeightUnit);
        // The enclosed space's floor plane
        EXPECT_EQ(w.terrainHeight(1, double(i), double(j)), 0.0);
    }
    // The kinematic mover in space 2 walks on space 2's surface
    Entity e = w.createIn(2);
    w.get<Transform>(e).teleport(Vec3(100, 0, 100), Quat::Identity());
    CharacterIntent intent;
    intent.desiredVelocity = Vec3(2, 0, 1);
    w.add<CharacterIntent>(e, intent);
    for (int k = 0; k < 60; ++k)
    {
        w.step();
        const Vec3 p = w.read<Transform>(e).position();
        ASSERT_EQ(p.y(), w.terrainHeight(2, p.x(), p.z())) << "step " << k;
        ASSERT_NE(p.y(), w.terrainHeight(0, p.x(), p.z())) << "step " << k;
    }
}

// ---- spawning in every space (Scope 1)

TEST(SpaceSpawn, SpawnInEverySpace)
{
    World w(spacesSetup());
    const Entity a = w.spawnIn<Marker>(1, Vec3(640, 0, 640));
    EXPECT_EQ(w.read<Transform>(a).space(), 1u);
    EXPECT_EQ(w.read<Transform>(a).position(), Vec3(640, 0, 640));
    const Entity b = w.spawnIn<Breeder>(2, Vec3(50, 0, 50));
    EXPECT_EQ(w.read<Transform>(b).space(), 2u);
    g_bred.clear();
    for (int k = 0; k < 240 && g_bred.empty(); ++k) w.step();
    // The Breeder's spawn (during the step, GameObject::spawn) is in its own space
    ASSERT_FALSE(g_bred.empty());
    const Entity c = g_bred.front();
    ASSERT_TRUE(w.alive(c));
    EXPECT_EQ(w.read<Transform>(c).space(), 2u);
    w.step();
    EXPECT_EQ(w.queryRadius(1, Vec3(640, 0, 640), 1), std::vector<Entity>{a});
    EXPECT_TRUE(w.queryRadius(0, Vec3(640, 0, 640), 1).empty());
    EXPECT_TRUE(w.queryRadius(2, Vec3(640, 0, 640), 1).empty());
    const std::vector<Entity> near2 = w.queryRadius(2, Vec3(51, 0, 50), 0.5);
    EXPECT_NE(std::find(near2.begin(), near2.end(), c), near2.end());
    EXPECT_TRUE(w.queryRadius(0, Vec3(51, 0, 50), 0.5).empty());
    // createIn (the world API's spawn without a class)
    const Entity d = w.createIn(1);
    EXPECT_EQ(w.read<Transform>(d).space(), 1u);
}

#ifdef SAIGA_ASSERTS
TEST(SpaceSpawnDeathTest, UnknownSpace)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    World w(spacesSetup());
    EXPECT_DEATH(w.spawnIn<Marker>(3, Vec3::Zero()), "unknown space 3");
    EXPECT_DEATH(w.createIn(3), "unknown space 3");
}
#endif

// ---- moveToSpace in one world (Scope 2)

TEST(MoveToSpace, RootAndDescendants)
{
    HookGuard guard;
    World w(spacesSetup());
    const Family f = makeFamily(w, Vec3(100, 0, 100));
    // A player in space 0 and an object near the cellar's origin in space 1: no player there, the far interval
    const Entity player = w.create();
    w.get<Transform>(player).teleport(Vec3(640, 0, 640), Quat::Identity());
    w.add<Player>(player);
    const Entity object = w.spawnIn<Marker>(1, Vec3(641, 0, 640));
    w.step(2);
    EXPECT_EQ(w.read<Behaviour>(object).tickInterval, w.worldConfig().ticking.farInterval);
    for (Entity e : {f.root, f.child, f.grandchild}) EXPECT_EQ(w.read<Transform>(e).space(), 0u);

    const uint8_t teleports = w.read<Transform>(f.root).teleportCount();
    const Vec3 target(640, 0, 650);
    w.moveToSpace(f.root, 1, target, Quat::Identity());
    // At once: the spaces of the subtree, one teleport of the root, no flags of the descendants yet
    for (Entity e : {f.root, f.child, f.grandchild}) EXPECT_EQ(w.read<Transform>(e).space(), 1u);
    EXPECT_EQ(w.read<Transform>(f.root).teleportCount(), uint8_t(teleports + 1));
    EXPECT_TRUE(w.read<Transform>(f.root).wasTeleported());
    EXPECT_FALSE(w.read<Transform>(f.child).wasTeleported());
    EXPECT_FALSE(w.read<Transform>(f.grandchild).wasTeleported());
    EXPECT_TRUE(w.read<Transform>(f.child).changedInThisStep());

    // After the next phase 6: the descendants' poses and teleport flags
    bool checked = false;
    g_phase7     = [&](World& world)
    {
        const Transform& r = world.read<Transform>(f.root);
        const Transform& c = world.read<Transform>(f.child);
        const Transform& g = world.read<Transform>(f.grandchild);
        EXPECT_EQ(c.position(), composed(r, world.read<LocalTransform>(f.child)));
        EXPECT_EQ(g.position(), composed(c, world.read<LocalTransform>(f.grandchild)));
        EXPECT_TRUE(c.wasTeleported());
        EXPECT_TRUE(g.wasTeleported());
        checked = true;
    };
    w.step();
    g_phase7 = nullptr;
    EXPECT_TRUE(checked);
    // The grid finds all three only in the new space
    for (Entity e : {f.root, f.child, f.grandchild})
    {
        const Vec3 p                  = w.read<Transform>(e).position();
        const std::vector<Entity> in1 = w.queryRadius(1, p, 0.01);
        EXPECT_NE(std::find(in1.begin(), in1.end(), e), in1.end());
        const std::vector<Entity> in0 = w.queryRadius(0, p, 0.01);
        EXPECT_EQ(std::find(in0.begin(), in0.end(), e), in0.end());
        const std::vector<Entity> old = w.queryRadius(0, Vec3(100, 0, 100), 5);
        EXPECT_EQ(std::find(old.begin(), old.end(), e), old.end());
    }
    // The tick player cells move with a moved player
    w.moveToSpace(player, 1, Vec3(640, 0, 640), Quat::Identity());
    w.step();
    EXPECT_EQ(w.read<Behaviour>(object).tickInterval, 0.0);
}

TEST(MoveToSpace, DuringAStep)
{
    HookGuard guard;
    World w(spacesSetup());
    w.setTickEveryStep(true);
    const Family f     = makeFamily(w, Vec3(10, 0, 10));
    const Entity diver = w.spawn<Diver>(Vec3(20, 0, 20));
    bool sawRoot = false, sawChildren = false;
    // From a system (phase 4): later systems see the spaces at once, the descendants' flags after phase 6
    g_phase4 = [&](World& world)
    {
        if (world.read<Transform>(f.root).space() != 0) return;
        world.moveToSpace(f.root, 2, Vec3(30, 0, 30), Quat::Identity());
        EXPECT_EQ(world.read<Transform>(f.child).space(), 2u);
        EXPECT_EQ(world.read<Transform>(f.grandchild).space(), 2u);
        sawRoot = true;
    };
    g_phase7 = [&](World& world)
    {
        if (!sawRoot || sawChildren) return;
        EXPECT_TRUE(world.read<Transform>(f.child).wasTeleported());
        EXPECT_TRUE(world.read<Transform>(f.grandchild).wasTeleported());
        EXPECT_EQ(world.read<Transform>(f.child).position(),
                  composed(world.read<Transform>(f.root), world.read<LocalTransform>(f.child)));
        sawChildren = true;
    };
    w.step();
    EXPECT_TRUE(sawRoot);
    EXPECT_TRUE(sawChildren);
    // From GameObject::update (phase 3)
    EXPECT_EQ(w.read<Transform>(diver).space(), 1u);
    EXPECT_EQ(w.read<Transform>(diver).position(), Vec3(641, 0, 640));
    w.step();
    EXPECT_EQ(w.queryRadius(1, Vec3(641, 0, 640), 0.01), std::vector<Entity>{diver});
}

TEST(MoveToSpace, SameSpaceIsATeleport)
{
    World w(spacesSetup());
    const Family f          = makeFamily(w, Vec3(10, 0, 10));
    const uint8_t teleports = w.read<Transform>(f.root).teleportCount();
    w.moveToSpace(f.root, 0, Vec3(12, 0, 10), Quat::Identity());
    EXPECT_EQ(w.read<Transform>(f.root).space(), 0u);
    EXPECT_EQ(w.read<Transform>(f.root).position(), Vec3(12, 0, 10));
    EXPECT_EQ(w.read<Transform>(f.root).teleportCount(), uint8_t(teleports + 1));
    w.step();
    EXPECT_EQ(w.read<Transform>(f.child).position(), Vec3(13, 0, 10));
    EXPECT_TRUE(w.read<Transform>(f.child).wasTeleportedLastStep());
}

TEST(MoveToSpace, AttachTakesTheParentsSpace)
{
    HookGuard guard;
    World w(spacesSetup());
    // Between steps: add<Parent> moves the child and its own child into the parent's space
    const Entity parent1 = w.createIn(1);
    const Family f       = makeFamily(w, Vec3(5, 0, 5));
    w.step();
    w.add<Parent>(f.root, Parent{parent1});
    for (Entity e : {f.root, f.child, f.grandchild}) EXPECT_EQ(w.read<Transform>(e).space(), 1u);
    EXPECT_TRUE(w.read<Transform>(f.root).wasTeleported());
    w.step();
    // reparentNow to a parent in space 2
    const Entity parent2 = w.createIn(2);
    EXPECT_TRUE(w.reparentNow(f.root, parent2));
    for (Entity e : {f.root, f.child, f.grandchild}) EXPECT_EQ(w.read<Transform>(e).space(), 2u);
    EXPECT_TRUE(w.read<Transform>(f.root).wasTeleported());
    w.step();
    // As a command during a step: moved at the flush
    const Entity loose = w.create();
    g_phase4           = [&](World& world)
    {
        if (world.has<Parent>(loose)) return;
        world.add<Parent>(loose, Parent{parent1});
        EXPECT_EQ(world.read<Transform>(loose).space(), 0u);
    };
    bool checked = false;
    g_phase7     = [&](World& world)
    {
        if (checked) return;
        EXPECT_EQ(world.read<Transform>(loose).space(), 1u);
        EXPECT_TRUE(world.read<Transform>(loose).wasTeleported());
        checked = true;
    };
    w.step();
    EXPECT_TRUE(checked);
}

#ifdef SAIGA_ASSERTS
TEST(MoveToSpaceDeathTest, Refusals)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    World w(spacesSetup());
    const Family f = makeFamily(w, Vec3(5, 0, 5));
    EXPECT_DEATH(w.moveToSpace(f.child, 1, Vec3::Zero(), Quat::Identity()), "is a child");
    EXPECT_DEATH(w.moveToSpace(f.root, 3, Vec3::Zero(), Quat::Identity()), "unknown space 3");
    EXPECT_DEATH(
        {
            const Entity dead = w.create();
            w.destroy(dead);
            w.moveToSpace(dead, 1, Vec3::Zero(), Quat::Identity());
        },
        "not alive");
    // A reserved entity (created during a step)
    EXPECT_DEATH(
        {
            g_phase4 = [](World& world) { world.moveToSpace(world.create(), 1, Vec3::Zero(), Quat::Identity()); };
            w.step();
        },
        "not alive");
    EXPECT_DEATH(
        {
            g_phase4 = [&](World& world)
            {
                world.view<Transform>().parallelEach([&](Entity e, Transform&)
                                                     { world.moveToSpace(e, 1, Vec3::Zero(), Quat::Identity()); });
            };
            w.step();
        },
        "parallelEach");
    // A state entity
    WorldSetup s = spacesSetup();
    s.components = [](Registry& r) { r.registerSpaceState<Zone>(Replicate::Host); };
    World withState(s);
    EXPECT_DEATH(withState.moveToSpace(withState.spaceStateEntity(0), 1, Vec3::Zero(), Quat::Identity()),
                 "state entity");
    // On a client, a replicated entity (R-NET-2)
    EXPECT_DEATH(
        {
            auto t        = LoopbackTransport::createNetwork(2, LoopbackOptions{});
            WorldSetup hs = spacesSetup();
            hs.transport  = t[0].get();
            World host(hs);
            WorldSetup cs = spacesSetup();
            cs.transport  = t[1].get();
            cs.role       = NetRole::Client;
            cs.localPeer  = 1;
            World client(cs);
            const Entity e = host.create();
            for (int k = 0; k < 3; ++k)
            {
                host.step();
                client.step();
            }
            const Entity c = client.replication().entityOfNetId(host.replication().netIdOf(e));
            if (c.valid()) client.moveToSpace(c, 1, Vec3::Zero(), Quat::Identity());
        },
        "a client cannot move a replicated entity");
}
#endif
