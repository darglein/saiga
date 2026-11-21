/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Spaces (docs/engine_design.md §7.4, §7.8, §7.9, §12.4; S21): several space templates, spawning in every space,
// moveToSpace, space-state entities per space, saves with spaces and the transitions through entrances.

#include "saiga/engine/components/CharacterIntent.h"
#include "saiga/engine/core/Hash.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/gameplay/GameObject.h"
#include "saiga/engine/world/World.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
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
    };
    return s;
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
