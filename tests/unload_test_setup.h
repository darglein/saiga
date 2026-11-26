/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The unload test setup (docs/engine_design.md §9.6, §12.4, §15.1; S25): shared by test_engine_unload,
// test_engine_save_directory, test_engine_save_crash and test_engine_save_benchmark. unloadTestSetup(): a streaming
// world of flat open-world spaces (generator 0: no terrain, no slots, so chunks need no parts and become resident in
// the update that requests them): "flat" (32 chunks, 8 km) and "cellar" (16 chunks), S24's scenario radii (fullRadius
// 128: required and requested 128 m, kept 192 m; simulationRadius 64), and the test game object Grower with its saved
// component GrowerState (an in-group reference, a reference out of the group, a PersistentId of a friend; onCatchUp
// adds the elapsed time to grown) plus the static hook counters of the tests.

#pragma once

#include "saiga/engine/components/Player.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/world/World.h"

#include <stdexcept>
#include <vector>

namespace UnloadTest
{
using namespace Saiga;
using namespace Saiga::Engine;

enum class UnloadTestComponent : uint8_t
{
    GrowerState = kFirstGameComponent,
    Crate,
    Count
};

struct GrowerState
{
    static constexpr const char* name = "GrowerState";
    static constexpr ComponentId id   = ComponentId(UnloadTestComponent::GrowerState);
    static constexpr uint32_t version = 1;
    double grown                      = 0;
    Entity buddy;    // inside the group (a child)
    Entity outside;  // another root: null after an unload
    PersistentId friendId;
    uint32_t catchUps = 0;
    template <class V>
    void visit(V& v)
    {
        v("grown", grown);
        v("buddy", buddy);
        v("outside", outside);
        v("friendId", friendId);
        v("catchUps", catchUps);
    }
};

// A test component for the schema tables and migrations (version 1 here; the migration tests register version 2)
struct Crate
{
    static constexpr const char* name = "Crate";
    static constexpr ComponentId id   = ComponentId(UnloadTestComponent::Crate);
    static constexpr uint32_t version = 1;
    int32_t items                     = 0;
    template <class V>
    void visit(V& v)
    {
        v("items", items);
    }
};

// The hook counters and switches of the tests (reset by resetHooks)
struct Hooks
{
    int spawns = 0, destroys = 0, configures = 0, loads = 0, catchUps = 0, replicatedDestroys = 0;
    // onLoad resolved the friend's PersistentId (another group of the same chunk)
    int friendsResolved = 0;
    // The order of the hooks of one reload: 'c' configure, 'l' onLoad, 'u' onCatchUp
    std::string order;
    bool throwOnCatchUp = false;
    bool spawnOnCatchUp = false;
    bool addSavedOnLoad = false;
    std::vector<double> elapsed;
};
inline Hooks& hooks()
{
    static Hooks h;
    return h;
}
inline void resetHooks()
{
    hooks() = Hooks();
}

enum class UnloadTestClass : uint16_t
{
    Grower = 0,
    Count
};

class Grower final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(UnloadTestClass::Grower);
    static constexpr const char* name = "Grower";
    void configure() override
    {
        hooks().configures++;
        hooks().order += 'c';
    }
    void onSpawn() override
    {
        hooks().spawns++;
        if (!has<GrowerState>()) add<GrowerState>();
    }
    void onLoad() override
    {
        hooks().loads++;
        hooks().order += 'l';
        if (has<GrowerState>() && read<GrowerState>().friendId.valid() &&
            world().resolve(read<GrowerState>().friendId).valid())
            hooks().friendsResolved++;
        if (hooks().addSavedOnLoad) add<Crate>();
    }
    void onCatchUp(double elapsedSeconds) override
    {
        hooks().catchUps++;
        hooks().order += 'u';
        hooks().elapsed.push_back(elapsedSeconds);
        if (hooks().throwOnCatchUp) throw std::runtime_error("onCatchUp failed");
        if (has<GrowerState>())
        {
            write<GrowerState>().grown += elapsedSeconds;
            write<GrowerState>().catchUps++;
        }
        if (hooks().spawnOnCatchUp) world().spawn<Grower>(world().read<Transform>(entity()).position());
    }
    void onDestroy() override { hooks().destroys++; }
    void onReplicatedDestroy() override { hooks().replicatedDestroys++; }
};

inline void registerUnloadTestComponents(Registry& r)
{
    r.registerComponent<GrowerState>(Replicate::Host);
    r.registerComponent<Crate>(Replicate::Never);
}
inline void registerUnloadTestClasses(ClassList& c)
{
    c.add<Grower>();
}

// The setup (streaming on unless streaming = false)
inline WorldSetup unloadTestSetup(bool streaming = true)
{
    WorldSetup s;
    s.config.streaming = streaming;
    SpaceTemplate flat;
    flat.name               = "flat";
    flat.generator          = 0;
    flat.sizeChunks         = 32;
    SpaceTemplate cellar    = flat;
    cellar.name             = "cellar";
    cellar.sizeChunks       = 16;
    s.config.spaceTemplates = {flat, cellar};
    for (SpaceTemplate& t : s.config.spaceTemplates)
    {
        t.fullRadius       = 128;
        t.simulationRadius = 64;
        t.prefetchMargin   = 0;
        t.unloadMargin     = 64;
    }
    s.components = registerUnloadTestComponents;
    s.classes    = registerUnloadTestClasses;
    return s;
}

// A Player root (Player, a Transform at p)
inline Entity addPlayer(World& w, const Vec3& p, SpaceId space = 0)
{
    const Entity e = w.createIn(space);
    w.add<Player>(e, Player{});
    w.get<Transform>(e).teleport(p, Quat::Identity());
    return e;
}

inline ChunkId chunkOf(SpaceId space, double x, double z)
{
    ChunkId c;
    c.space = space;
    c.x     = int32_t(std::floor(x / 256));
    c.z     = int32_t(std::floor(z / 256));
    return c;
}

// The entities of the world with this PersistentId, resident or stored (0 or 1 in a consistent world)
inline int countPersistent(World& w, PersistentId id)
{
    return w.resolve(id).valid() ? 1 : 0;
}

}  // namespace UnloadTest
