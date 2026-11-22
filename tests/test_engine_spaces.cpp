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
#include "saiga/engine/components/PendingTransition.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/RenderPose.h"
#include "saiga/engine/components/SpaceLink.h"
#include "saiga/engine/core/Hash.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/gameplay/GameObject.h"
#include "saiga/engine/gameplay/Systems.h"
#include "saiga/engine/net/LoopbackTransport.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/save/SaveFormat.h"
#include "saiga/engine/save/SaveReader.h"
#include "saiga/engine/world/World.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
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

// A second space-state component (registered since a save)
struct Lamp
{
    static constexpr const char* name = "Lamp";
    static constexpr ComponentId id   = ComponentId(kFirstGameComponent + 1);
    static constexpr uint32_t version = 1;
    bool on                           = true;
    template <class V>
    void visit(V& v)
    {
        v("on", on);
    }
};

void zoneComponents(Registry& r)
{
    r.registerSpaceState<Zone>(Replicate::Host);
}
void zoneAndLampComponents(Registry& r)
{
    r.registerSpaceState<Zone>(Replicate::Host);
    r.registerSpaceState<Lamp>(Replicate::Host);
}

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

// Test systems: phase 1 (after the engine's space_transition), phase 4 and phase 7 (after the transform phase) run
// these when set
std::function<void(World&)> g_phase1, g_phase4, g_phase7;

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

WorldSetup spacesSetup(uint64_t seed = 9, void (*components)(Registry&) = nullptr)
{
    WorldSetup s;
    s.seed                  = seed;
    s.profiling             = false;
    s.components            = components;
    s.config.spaceTemplates = threeSpaces();
    s.classes               = [](ClassList& c)
    {
        c.add<Marker>();
        c.add<Breeder>();
        c.add<Diver>();
    };
    s.systems = [](SystemList& l)
    {
        l.add(Phase::Input, "test_phase1",
              [](World& w, double)
              {
                  if (g_phase1) g_phase1(w);
              });
        l.add(Phase::Gameplay, "test_phase4",
              [](World& w, double)
              {
                  if (g_phase4) g_phase4(w);
              });
        // S21: the optional entrance system (games register it)
        l.add(Phase::Gameplay, "space_link", spaceLinkSystem, RunOn::Host);
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
        g_phase1 = nullptr;
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

// Regression (PR review, S21): an attached child's RenderPose kept the old space when no ancestor had a RenderPose
// (the extract drew it in the old space), and its interpolation buffer's old-space samples would rewrite it
TEST(MoveToSpace, AttachRefreshesARenderPose)
{
    World w(spacesSetup());
    const Entity parent = w.createIn(2);
    const Entity child  = w.create();
    w.get<Transform>(child).teleport(Vec3(5, 0, 5), Quat::Identity());
    w.add<RenderPose>(child, RenderPose{Vec3(5, 0, 5), Quat::Identity(), 0, 0});
    w.step();
    w.add<Parent>(child, Parent{parent});
    EXPECT_EQ(w.read<RenderPose>(child).space, 2u);
    EXPECT_EQ(w.read<RenderPose>(child).position, Vec3(5, 0, 5));
    w.step();
    EXPECT_EQ(w.read<RenderPose>(child).space, 2u);
    // The same space: untouched
    const Entity other                = w.createIn(2);
    w.get<RenderPose>(child).position = Vec3(1, 2, 3);
    EXPECT_TRUE(w.reparentNow(child, other));
    EXPECT_EQ(w.read<RenderPose>(child).position, Vec3(1, 2, 3));
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
    // On a client, a local root with a replicated child (R-NET-2). Regression (PR review, S21): only the root was
    // checked, the replicated descendant was moved with it
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
            if (!c.valid()) return;
            const Entity local = client.create();
            client.add<Parent>(c, Parent{local});
            if (client.has<Parent>(c)) client.moveToSpace(local, 1, Vec3::Zero(), Quat::Identity());
        },
        "a client cannot move a replicated entity");
    // On a client, a local attach of a replica under a local parent in another space (R-NET-2). Regression
    // (CodeRabbit, S21): the replica took the parent's space, and the host's snapshots never corrected it
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
            if (!c.valid()) return;
            const Entity local = client.createIn(1);
            client.add<Parent>(c, Parent{local});
        },
        "into the local parent's space");
}
#endif

// ---- space-state entities per space (Scope 3)

TEST(SpaceState, OneEntityPerSpace)
{
    World w(spacesSetup(9, zoneComponents));
    std::vector<Entity> states;
    for (SpaceId space = 0; space < 3; ++space)
    {
        const Entity e = w.spaceStateEntity(space);
        ASSERT_TRUE(w.alive(e));
        EXPECT_EQ(w.read<Transform>(e).space(), space);
        EXPECT_EQ(std::find(states.begin(), states.end(), e), states.end());
        states.push_back(e);
        w.spaceState<Zone>(space).danger = double(space) + 0.5;
    }
    EXPECT_NE(states[0], states[1]);
    EXPECT_NE(states[1], states[2]);
    for (SpaceId space = 0; space < 3; ++space) EXPECT_EQ(w.readSpaceState<Zone>(space).danger, double(space) + 0.5);
    // In SpaceId order after the world-state entity (none here): slots 1, 2, 3
    EXPECT_EQ(states[0].index, 1u);
    EXPECT_EQ(states[2].index, 3u);
    // A world without space-state components has none
    World plain(spacesSetup());
    EXPECT_FALSE(plain.spaceStateEntity(0).valid());
    EXPECT_EQ(plain.registry().aliveCount(), 0u);
}

// ---- saves with spaces (Scope 5)

namespace
{
// Sets the saved Transform::space of an entity index in a save blob (corrupt saves)
std::vector<uint8_t> withSavedSpace(std::vector<uint8_t> blob, uint32_t index, SpaceId space)
{
    ParsedSave p;
    std::string error;
    EXPECT_TRUE(parseSave(blob, p, error)) << error;
    for (const ParsedSave::Block& b : p.components)
    {
        if (p.schema.types[b.schemaIndex].name != "Transform") continue;
        // Each value: the entity index (u32), then the Transform (space first)
        const size_t valueSize = size_t(b.end - b.begin) / size_t(b.count);
        for (size_t at = b.begin; at < b.end; at += valueSize)
        {
            uint32_t i;
            std::memcpy(&i, blob.data() + at, 4);
            if (i == index)
            {
                std::memcpy(blob.data() + at + 4, &space, sizeof(SpaceId));
                return blob;
            }
        }
    }
    ADD_FAILURE() << "no Transform of entity index " << index;
    return blob;
}
}  // namespace

TEST(SpaceSave, RoundTrip)
{
    World a(spacesSetup(9, zoneComponents));
    const Entity m0              = a.spawnIn<Marker>(0, Vec3(10, 0, 10));
    const Entity m1              = a.spawnIn<Marker>(1, Vec3(640, 0, 641));
    const Entity m2              = a.spawnIn<Marker>(2, Vec3(70, 0, 71));
    const Family f               = makeFamily(a, Vec3(20, 0, 20));
    a.spaceState<Zone>(2).danger = 7;
    a.step();
    a.moveToSpace(f.root, 2, Vec3(80, 0, 80), Quat::Identity());
    a.step(2);
    const auto blob = a.save();
    World b(spacesSetup(9, zoneComponents));
    const LoadResult r = b.tryLoad(blob);
    ASSERT_TRUE(r.ok()) << r.error;
    EXPECT_EQ(b.save(), blob);
    EXPECT_EQ(b.read<Transform>(m0).space(), 0u);
    EXPECT_EQ(b.read<Transform>(m1).space(), 1u);
    EXPECT_EQ(b.read<Transform>(m2).space(), 2u);
    for (Entity e : {f.root, f.child, f.grandchild}) EXPECT_EQ(b.read<Transform>(e).space(), 2u);
    EXPECT_EQ(b.readSpaceState<Zone>(2).danger, 7);
    for (SpaceId space = 0; space < 3; ++space) EXPECT_EQ(b.read<Transform>(b.spaceStateEntity(space)).space(), space);
    EXPECT_EQ(b.queryRadius(1, Vec3(640, 0, 641), 0.01), std::vector<Entity>{m1});
    EXPECT_TRUE(b.queryRadius(0, Vec3(640, 0, 641), 0.01).empty());
    EXPECT_EQ(b.queryRadius(2, Vec3(80, 0, 80), 0.01), std::vector<Entity>{f.root});
    b.step();
    a.step();
    EXPECT_EQ(a.hash(), b.hash());
}

TEST(SpaceSave, SingleSpaceHasNoSpaceTable)
{
    WorldSetup s;
    s.profiling = false;
    World w(s);
    w.create();
    ParsedSave p;
    std::string error;
    ASSERT_TRUE(parseSave(w.save(), p, error)) << error;
    EXPECT_FALSE(p.spaces.has_value());
    World three(spacesSetup());
    ASSERT_TRUE(parseSave(three.save(), p, error)) << error;
    ASSERT_TRUE(p.spaces.has_value());
    EXPECT_EQ(p.spaces->rows.size(), 3u);
}

TEST(SpaceSave, SpaceTable)
{
    World a(spacesSetup(9, zoneComponents));
    a.spawnIn<Marker>(2, Vec3(5, 0, 5));
    const auto blob = a.save();
    // Four spaces (one appended): loads, the new space gets its space-state entity
    {
        WorldSetup s = spacesSetup(9, zoneComponents);
        s.config.spaceTemplates.push_back(SpaceTemplate::enclosed("attic", 2));
        World b(s);
        const LoadResult r = b.tryLoad(blob);
        ASSERT_TRUE(r.ok()) << r.error;
        ASSERT_TRUE(b.alive(b.spaceStateEntity(3)));
        EXPECT_EQ(b.read<Transform>(b.spaceStateEntity(3)).space(), 3u);
    }
    auto rejected = [&](std::vector<SpaceTemplate> templates, const std::string& message)
    {
        WorldSetup s            = spacesSetup(9, zoneComponents);
        s.config.spaceTemplates = std::move(templates);
        World b(s);
        const auto before  = b.save();
        const LoadResult r = b.tryLoad(blob);
        EXPECT_FALSE(r.committed);
        EXPECT_EQ(r.error, message);
        EXPECT_EQ(b.save(), before);
    };
    std::vector<SpaceTemplate> t = threeSpaces();
    rejected({t[0], t[1]}, "the save has 3 spaces and the setup 2 (space templates are append-only)");
    std::vector<SpaceTemplate> renamed = t;
    renamed[1].name                    = "cave";
    rejected(renamed, "space 1 is 'cellar' in the save and 'cave' in the setup");
    std::vector<SpaceTemplate> kind = t;
    kind[2]                         = SpaceTemplate::enclosed("caves");
    rejected(kind, "space 2 ('caves') is open world in the save and enclosed in the setup");
}

TEST(SpaceSave, StateEntitiesPerSpace)
{
    // Two spaces added since the save, at free slots: the destroy order decides the slots, the mapping is by space
    bool outOfOrder = false;
    for (int order = 0; order < 2; ++order)
    {
        WorldSetup one = spacesSetup(9, zoneComponents);
        one.config.spaceTemplates.resize(1);
        World a(one);
        const Entity x = a.create(), y = a.create();
        a.create();
        a.destroy(order == 0 ? x : y);
        a.destroy(order == 0 ? y : x);
        a.spaceState<Zone>(0).danger = 3;
        const auto blob              = a.save();

        World b(spacesSetup(9, zoneComponents));
        const int counter  = g_zoneCounter;
        const LoadResult r = b.tryLoad(blob);
        ASSERT_TRUE(r.ok()) << r.error;
        // Two staged defaults (one per new entity): distinct serials
        EXPECT_GE(g_zoneCounter - counter, 2);
        EXPECT_NE(b.readSpaceState<Zone>(1).serial, b.readSpaceState<Zone>(2).serial);
        EXPECT_EQ(b.readSpaceState<Zone>(0).danger, 3);
        for (SpaceId space = 0; space < 3; ++space)
            EXPECT_EQ(b.read<Transform>(b.spaceStateEntity(space)).space(), space);
        outOfOrder = outOfOrder || b.spaceStateEntity(1).index > b.spaceStateEntity(2).index;
        // The mapping survives another round trip (slots not in space order)
        b.spaceState<Zone>(2).danger = 9;
        const auto blob2             = b.save();
        World c(spacesSetup(9, zoneComponents));
        ASSERT_TRUE(c.tryLoad(blob2).ok());
        EXPECT_EQ(c.readSpaceState<Zone>(2).danger, 9);
        EXPECT_EQ(c.spaceStateEntity(1), b.spaceStateEntity(1));
        EXPECT_EQ(c.spaceStateEntity(2), b.spaceStateEntity(2));
    }
    EXPECT_TRUE(outOfOrder);

    World a(spacesSetup(9, zoneComponents));
    const auto blob = a.save();
    // A space-state component registered since the save: added to every space-state entity
    {
        World b(spacesSetup(9, zoneAndLampComponents));
        ASSERT_TRUE(b.tryLoad(blob).ok());
        for (SpaceId space = 0; space < 3; ++space)
        {
            EXPECT_EQ(b.spaceStateEntity(space), a.spaceStateEntity(space));
            EXPECT_TRUE(b.has<Lamp>(b.spaceStateEntity(space)));
        }
    }
    // No space-state component any more: all of them are freed
    {
        WorldSetup none = spacesSetup();
        none.migrations = [](MigrationList& m) { m.addRemoved("Zone", RemovedPolicy::DropComponent); };
        World b(none);
        const LoadResult r = b.tryLoad(blob);
        ASSERT_TRUE(r.ok()) << r.error;
        EXPECT_FALSE(b.spaceStateEntity(0).valid());
        EXPECT_EQ(b.registry().aliveCount(), 0u);
    }
}

TEST(SpaceSave, CorruptStates)
{
    World a(spacesSetup(9, zoneComponents));
    const Entity m  = a.spawnIn<Marker>(1, Vec3(640, 0, 640));
    const Family f  = makeFamily(a, Vec3(1, 0, 1));
    const auto blob = a.save();
    auto rejected   = [&](const std::vector<uint8_t>& bad, const std::string& message)
    {
        World b(spacesSetup(9, zoneComponents));
        b.spawnIn<Marker>(2, Vec3(1, 0, 1));
        const auto before  = b.save();
        const LoadResult r = b.tryLoad(bad);
        EXPECT_FALSE(r.committed);
        EXPECT_NE(r.error.find(message), std::string::npos) << r.error;
        EXPECT_EQ(b.save(), before);
    };
    ASSERT_TRUE(World(spacesSetup(9, zoneComponents)).tryLoad(blob).ok());
    // Two space-state entities of space 0 (the one of space 1 claims space 0)
    rejected(withSavedSpace(blob, a.spaceStateEntity(1).index, 0), "two space-state entities of space 0");
    // An entity in an unknown space
    rejected(withSavedSpace(blob, m.index, 5), "is in space 5, but the save has 3 spaces");
    // A child in another space than its parent
    rejected(withSavedSpace(blob, f.grandchild.index, 2), "its parent " + std::to_string(f.child.index));
}

// ---- transitions (Scope 6)

namespace
{
Quat yaw(double degrees)
{
    const double a = degrees * 3.14159265358979323846 / 360.0;  // half the angle
    return Quat(std::cos(a), 0, std::sin(a), 0);
}
Entity makeLink(World& w, SpaceId space, const Vec3& p, SpaceId target, const Vec3& arrival,
                const Quat& rotation = Quat::Identity())
{
    const Entity e = w.createIn(space);
    w.get<Transform>(e).setPosition(p);
    SpaceLink l;
    l.target          = target;
    l.arrival         = arrival;
    l.arrivalRotation = rotation;
    w.add<SpaceLink>(e, l);
    return e;
}
// The log lines of a category since a sequence number, joined
std::string logSince(uint64_t before)
{
    std::string out;
    for (const auto& l : engineLog().since(before)) out += l.message + "\n";
    return out;
}
}  // namespace

TEST(Transition, CompletesInTheNextStep)
{
    HookGuard guard;
    World w(spacesSetup());
    const Entity link   = makeLink(w, 0, Vec3(100, 0, 100), 1, Vec3(640, 0, 640), yaw(90));
    const Entity back   = makeLink(w, 1, Vec3(645, 0, 640), 0, Vec3(103, 0, 102), yaw(180));
    const Entity player = w.create();
    w.get<Transform>(player).teleport(Vec3(101, 0, 100), Quat::Identity());
    w.step();
    // Between steps: applied at once; the pose is copied (a later edit of the link changes nothing)
    w.requestTransition(player, link);
    ASSERT_TRUE(w.has<PendingTransition>(player));
    EXPECT_EQ(w.read<PendingTransition>(player).target, 1u);
    w.get<SpaceLink>(link).arrival = Vec3(1, 2, 3);
    bool checked                   = false;
    g_phase4                       = [&](World& world)
    {
        if (checked) return;
        // Moved in phase 1 of the next step, the component removed at the phase-1 flush
        EXPECT_EQ(world.read<Transform>(player).space(), 1u);
        EXPECT_EQ(world.read<Transform>(player).position(), Vec3(640, 0, 640));
        EXPECT_FALSE(world.has<PendingTransition>(player));
        checked = true;
    };
    w.step();
    EXPECT_TRUE(checked);
    EXPECT_LT((w.read<Transform>(player).rotation().coeffs() - yaw(90).coeffs()).norm(), 1e-12);
    // The event is readable in the following step
    std::vector<SpaceTransitionEvent> events;
    g_phase4 = [&](World& world)
    {
        for (const SpaceTransitionEvent& ev : world.events.read<SpaceTransitionEvent>()) events.push_back(ev);
    };
    w.step();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].entity, player);
    EXPECT_EQ(events[0].link, link);
    EXPECT_EQ(events[0].from, 0u);
    EXPECT_EQ(events[0].to, 1u);
    events.clear();

    // From a phase-4 system; the link destroyed in the same phase: the copied pose still applies
    bool requested = false;
    g_phase4       = [&](World& world)
    {
        for (const SpaceTransitionEvent& ev : world.events.read<SpaceTransitionEvent>()) events.push_back(ev);
        if (requested) return;
        requested = true;
        world.requestTransition(player, back);
        world.destroy(back);
        EXPECT_FALSE(world.has<PendingTransition>(player));  // a command
    };
    w.step();
    EXPECT_TRUE(w.has<PendingTransition>(player));
    EXPECT_EQ(w.read<Transform>(player).space(), 1u);
    w.step();
    EXPECT_EQ(w.read<Transform>(player).space(), 0u);
    EXPECT_EQ(w.read<Transform>(player).position(), Vec3(103, 0, 102));
    w.step();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].link, back);
    EXPECT_FALSE(w.alive(back));
    EXPECT_EQ(events[0].from, 1u);
    EXPECT_EQ(events[0].to, 0u);
}

namespace
{
// A game message whose host handler requests a transition (handlers run in phase 1 before spaceTransitionSystem)
Entity g_msgPlayer, g_msgLink;
struct EnterLink
{
    static constexpr MessageId id     = 64;
    static constexpr const char* name = "TestEnterLink";
    static constexpr uint32_t version = 1;
    uint32_t value                    = 0;
    template <class V>
    void visit(V& v)
    {
        v("value", value);
    }
};
}  // namespace

// Regression (PR review, S21): a request of a message handler completed in the same step (the handler's command is
// flushed before the phase-1 systems), not in the next one
TEST(Transition, MessageHandlerRequestCompletesInTheNextStep)
{
    auto t        = LoopbackTransport::createNetwork(2, LoopbackOptions{});
    WorldSetup hs = spacesSetup();
    hs.transport  = t[0].get();
    hs.messages   = [](MessageList& l)
    {
        l.add<EnterLink>(Reliability::Reliable,
                         [](World& w, uint8_t, const EnterLink&) { w.requestTransition(g_msgPlayer, g_msgLink); });
    };
    World host(hs);
    WorldSetup cs = spacesSetup();
    cs.transport  = t[1].get();
    cs.role       = NetRole::Client;
    cs.localPeer  = 1;
    cs.messages   = hs.messages;
    World client(cs);
    g_msgLink   = makeLink(host, 0, Vec3(10, 0, 10), 1, Vec3(640, 0, 640), yaw(0));
    g_msgPlayer = host.create();
    for (int k = 0; k < 3; ++k)
    {
        host.step();
        client.step();
    }
    client.send(0, EnterLink{});
    client.step();
    host.step();
    ASSERT_TRUE(host.has<PendingTransition>(g_msgPlayer));
    EXPECT_EQ(host.read<Transform>(g_msgPlayer).space(), 0u);
    host.step();
    EXPECT_FALSE(host.has<PendingTransition>(g_msgPlayer));
    EXPECT_EQ(host.read<Transform>(g_msgPlayer).space(), 1u);
}

// Regression (CodeRabbit, S21): a phase-1 abort after the move kept it but dropped the removal of PendingTransition;
// the rerun moved the root again and reported from = to = 1. The move of the aborted phase is undone now
TEST(Transition, AbortedPhaseOneMovesOnce)
{
    HookGuard guard;
    World w(spacesSetup());
    const Entity link   = makeLink(w, 0, Vec3(10, 0, 10), 1, Vec3(640, 0, 640), yaw(90));
    const Entity player = w.create();
    w.get<Transform>(player).teleport(Vec3(11, 0, 10), Quat::Identity());
    w.step();
    w.requestTransition(player, link);
    int throws = 1;
    g_phase1   = [&](World&)
    {
        if (throws-- > 0) throw std::runtime_error("phase 1 aborts");
    };
    EXPECT_THROW(w.step(), std::runtime_error);
    EXPECT_EQ(w.read<Transform>(player).space(), 0u);
    EXPECT_EQ(w.read<Transform>(player).position(), Vec3(11, 0, 10));
    EXPECT_TRUE(w.has<PendingTransition>(player));
    w.step();
    EXPECT_EQ(w.read<Transform>(player).space(), 1u);
    EXPECT_FALSE(w.has<PendingTransition>(player));
    std::vector<SpaceTransitionEvent> events;
    g_phase4 = [&](World& world)
    {
        for (const SpaceTransitionEvent& ev : world.events.read<SpaceTransitionEvent>()) events.push_back(ev);
    };
    w.step();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].from, 0u);
    EXPECT_EQ(events[0].to, 1u);
}

TEST(Transition, FirstRequestWins)
{
    HookGuard guard;
    World w(spacesSetup());
    const Entity a      = makeLink(w, 0, Vec3(10, 0, 10), 1, Vec3(640, 0, 640));
    const Entity b      = makeLink(w, 0, Vec3(12, 0, 10), 2, Vec3(50, 0, 50));
    const Entity player = w.create();
    bool once           = false;
    g_phase4            = [&](World& world)
    {
        if (once) return;
        once = true;
        world.requestTransition(player, a);
        world.requestTransition(player, b);
    };
    const uint64_t before = engineLog().lastSequence();
    w.step();
    ASSERT_TRUE(w.has<PendingTransition>(player));
    EXPECT_EQ(w.read<PendingTransition>(player).link, a);
    EXPECT_NE(logSince(before).find("dropped at the flush"), std::string::npos) << logSince(before);
    // Between steps: refused at once
    const uint64_t later = engineLog().lastSequence();
    w.requestTransition(player, b);
    EXPECT_EQ(w.read<PendingTransition>(player).link, a);
    EXPECT_NE(logSince(later).find("the first request wins"), std::string::npos) << logSince(later);
    w.step();
    EXPECT_EQ(w.read<Transform>(player).space(), 1u);
}

TEST(Transition, TargetOutOfRangeIsRefused)
{
    World w(spacesSetup());
    const Entity link     = makeLink(w, 0, Vec3(10, 0, 10), 7, Vec3(640, 0, 640));
    const Entity player   = w.create();
    const uint64_t before = engineLog().lastSequence();
    w.requestTransition(player, link);
    EXPECT_FALSE(w.has<PendingTransition>(player));
    EXPECT_NE(logSince(before).find("unknown space 7"), std::string::npos) << logSince(before);
}

TEST(Transition, RevalidatedAtCompletion)
{
    World w(spacesSetup());
    const Entity link   = makeLink(w, 0, Vec3(10, 0, 10), 1, Vec3(640, 0, 640));
    const Entity player = w.create();
    const Entity parent = w.create();
    w.requestTransition(player, link);
    // A parent since the request: removed with a log line, no move
    w.add<Parent>(player, Parent{parent});
    uint64_t before = engineLog().lastSequence();
    w.step();
    EXPECT_FALSE(w.has<PendingTransition>(player));
    EXPECT_EQ(w.read<Transform>(player).space(), 0u);
    EXPECT_NE(logSince(before).find("is a child now"), std::string::npos) << logSince(before);
    // A PendingTransition with an unknown target (as a tool's set_component writes it)
    const Entity other = w.create();
    PendingTransition p;
    p.link   = link;
    p.target = 9;
    w.add<PendingTransition>(other, p);
    before = engineLog().lastSequence();
    w.step();
    EXPECT_FALSE(w.has<PendingTransition>(other));
    EXPECT_EQ(w.read<Transform>(other).space(), 0u);
    EXPECT_NE(logSince(before).find("the target is not a space"), std::string::npos) << logSince(before);
}

TEST(Transition, SpaceLinkSystem)
{
    World w(spacesSetup());
    const Entity link   = makeLink(w, 0, Vec3(10, 0, 10), 2, Vec3(50, 0, 50));
    const Entity plain  = w.create();  // no SpaceLink
    const Entity player = w.create();
    const Entity far    = w.createIn(1);  // in another space than the link
    w.step();
    w.events.emit(InteractEvent{link, player, 0});
    w.events.emit(InteractEvent{plain, far, 0});
    w.events.emit(InteractEvent{link, far, 0});
    const uint64_t before = engineLog().lastSequence();
    w.step();
    ASSERT_TRUE(w.has<PendingTransition>(player));
    EXPECT_FALSE(w.has<PendingTransition>(far));
    EXPECT_NE(logSince(before).find("in another space"), std::string::npos) << logSince(before);
    w.step();
    EXPECT_EQ(w.read<Transform>(player).space(), 2u);
    EXPECT_EQ(w.read<Transform>(player).position(), Vec3(50, 0, 50));
    EXPECT_EQ(w.read<Transform>(far).space(), 1u);
}

#ifdef SAIGA_ASSERTS
TEST(TransitionDeathTest, Refusals)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    World w(spacesSetup());
    const Entity link   = makeLink(w, 0, Vec3(10, 0, 10), 1, Vec3(640, 0, 640));
    const Entity player = w.create();
    const Entity child  = w.create();
    w.add<Parent>(child, Parent{player});
    EXPECT_DEATH(w.requestTransition(child, link), "is a child");
    EXPECT_DEATH(w.requestTransition(player, w.create()), "has no SpaceLink");
    EXPECT_DEATH(
        {
            const Entity dead = makeLink(w, 0, Vec3::Zero(), 1, Vec3::Zero());
            w.destroy(dead);
            w.requestTransition(player, dead);
        },
        "has no SpaceLink");
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
            client.requestTransition(client.create(), client.create());
        },
        "on a client");
}
#endif

TEST(SpaceSave, PendingTransitionSurvives)
{
    HookGuard guard;
    World a(spacesSetup());
    const Entity link   = makeLink(a, 0, Vec3(10, 0, 10), 1, Vec3(640, 0, 640), yaw(90));
    const Entity player = a.create();
    a.step();
    a.requestTransition(player, link);
    const auto blob = a.save();
    World b(spacesSetup());
    ASSERT_TRUE(b.tryLoad(blob).ok());
    ASSERT_TRUE(b.has<PendingTransition>(player));
    b.step();
    EXPECT_EQ(b.read<Transform>(player).space(), 1u);
    EXPECT_EQ(b.read<Transform>(player).position(), Vec3(640, 0, 640));
    std::vector<SpaceTransitionEvent> events;
    g_phase4 = [&](World& world)
    {
        for (const SpaceTransitionEvent& ev : world.events.read<SpaceTransitionEvent>()) events.push_back(ev);
    };
    b.step();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].entity, player);
    EXPECT_EQ(events[0].to, 1u);
    // The same as the world that never saved
    a.step(2);
    EXPECT_EQ(a.hash(), b.hash());
}

TEST(SpaceSave, CorruptTransitionTargets)
{
    World a(spacesSetup());
    const Entity link   = makeLink(a, 0, Vec3(10, 0, 10), 1, Vec3(640, 0, 640));
    const Entity player = a.create();
    PendingTransition p;
    p.link   = link;
    p.target = 9;
    a.add<PendingTransition>(player, p);
    auto rejected = [](const std::vector<uint8_t>& bad, const std::string& message)
    {
        World b(spacesSetup());
        const auto before  = b.save();
        const LoadResult r = b.tryLoad(bad);
        EXPECT_FALSE(r.committed);
        EXPECT_NE(r.error.find(message), std::string::npos) << r.error;
        EXPECT_EQ(b.save(), before);
    };
    rejected(a.save(), "a PendingTransition to the unknown space 9");
    a.remove<PendingTransition>(player);
    a.get<SpaceLink>(link).target = 3;
    rejected(a.save(), "a SpaceLink to the unknown space 3");
}
