/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Entity unloading and reloading (docs/engine_design.md §9.6, §12.3, §12.4; S25): the group form and its remapping,
// the unload command (no onDestroy, unloaded log records, Despawn), the exclusions, the one unload pass in phase 7 over
// the kept set, the reloads with configure / onLoad / onCatchUp, abort safety, determinism across saves, blobs with
// stored groups, and space unloading. Worlds of tests/unload_test_setup.h (flat spaces: no generation cost).

#include "saiga/engine/api/WorldApi.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/PendingTransition.h"
#include "saiga/engine/components/Weather.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/lighting/EnvironmentWrites.h"
#include "saiga/engine/net/LoopbackTransport.h"
#include "saiga/engine/net/NetOwner.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/save/ChunkStore.h"
#include "saiga/engine/save/EntityGroup.h"
#include "saiga/engine/save/EntityUnloading.h"
#include "saiga/engine/save/SaveDiff.h"
#include "saiga/engine/save/SaveReader.h"
#include "saiga/engine/streaming/ChunkResidency.h"

#include "gtest/gtest.h"

#include "unload_test_setup.h"

using namespace Saiga;
using namespace Saiga::Engine;
using namespace UnloadTest;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);

std::unique_ptr<World> makeWorld(bool streaming = true, void (*systems)(SystemList&) = nullptr,
                                 bool environment = false)
{
    WorldSetup s = unloadTestSetup(streaming);
    s.systems    = systems;
    if (environment) s.environment.enabled = true;
    auto w = std::make_unique<World>(s);
    w->setStreamingSynchronous(true);
    return w;
}

void teleport(World& w, Entity e, const Vec3& p)
{
    w.get<Transform>(e).teleport(p, Quat::Identity());
}

size_t groupsAt(const World& w, const Vec3& p, SpaceId space = 0)
{
    return w.chunkStoreView().groupCount(chunkOf(space, p.x(), p.z()));
}

bool hasStoredGroups(const std::vector<uint8_t>& blob)
{
    ParsedSave p;
    std::string error;
    return parseSave(blob, p, error) && p.storedGroups.has_value();
}

// The player at A, everything else 2 km away
const Vec3 kA(1000, 0, 1000), kB(3000, 0, 1000);
}  // namespace

// ==== EntityGroup ====================================================================================================

// Regression (CodeRabbit on #91): the cached subtree links still named a child whose Parent was removed; the traversal
// read the missing Parent instead of rebuilding
TEST(EntityGroup, SubtreeAfterParentRemoval)
{
    auto wp           = makeWorld();
    World& w          = *wp;
    const Entity root = w.create();
    const Entity c1   = w.create();
    const Entity c2   = w.create();
    w.add<Parent>(c1, Parent{root});
    w.add<Parent>(c2, Parent{root});
    EXPECT_EQ(w.subtree(root).size(), 3u);  // builds the link cache
    w.remove<Parent>(c1);
    const std::vector<Entity> s = w.subtree(root);
    ASSERT_EQ(s.size(), 2u);
    EXPECT_EQ(s[1], c2);
}

TEST(EntityGroup, RoundTripRemapsEntities)
{
    resetHooks();
    auto wp        = makeWorld();
    World& w       = *wp;
    const Entity p = addPlayer(w, kA);
    // A root with two children and a grandchild; a reference inside the group and one to another root
    const Entity root  = w.spawn<Grower>(kA + Vec3(100, 0, 0));
    const Entity c1    = w.create();
    const Entity c2    = w.create();
    const Entity gc    = w.create();
    const Entity other = w.spawn<Grower>(kA + Vec3(0, 0, 50));
    w.add<Parent>(c1, Parent{root});
    w.add<Parent>(c2, Parent{root});
    w.add<Parent>(gc, Parent{c1});
    w.get<GrowerState>(root).buddy   = gc;
    w.get<GrowerState>(root).outside = other;
    const PersistentId rootId = w.makePersistent(root), gcId = w.makePersistent(gc);
    w.step();
    // The player leaves: both roots' chunk is no longer kept (the root first, by entity index: other is alive then)
    teleport(w, p, kB);
    w.step();
    EXPECT_FALSE(w.alive(root));
    EXPECT_FALSE(w.alive(gc));
    EXPECT_FALSE(w.resolve(rootId).valid());
    EXPECT_EQ(w.chunkStoreView().totalGroups(), 2u);
    EXPECT_EQ(w.chunkStoreView().stats().entitiesUnloaded, 5u);
    EXPECT_EQ(w.chunkStoreView().stats().droppedReferences, 1u);
    EXPECT_EQ(hooks().destroys, 0);
    // Back: the groups reload at the step start
    teleport(w, p, kA);
    w.step();
    const Entity r2 = w.resolve(rootId), g2 = w.resolve(gcId);
    ASSERT_TRUE(r2.valid());
    ASSERT_TRUE(g2.valid());
    EXPECT_EQ(w.read<GrowerState>(r2).buddy, g2);
    EXPECT_FALSE(w.read<GrowerState>(r2).outside.valid());
    const Entity c1b = w.read<Parent>(g2).entity;
    ASSERT_TRUE(w.alive(c1b));
    EXPECT_EQ(w.read<Parent>(c1b).entity, r2);
    EXPECT_EQ(groupsAt(w, kA), 0u);
    EXPECT_EQ(w.chunkStoreView().stats().entitiesReloaded, 5u);
    EXPECT_EQ(hooks().spawns, 2);
}

TEST(EntityGroup, FormBounds)
{
    std::string e;
    GroupForm f;
    auto form = [](uint32_t entities, uint32_t blocks)
    {
        std::vector<uint8_t> b;
        BinaryWriter w(b);
        w.write(uint64_t(0));
        w.write(entities);
        w.write(uint32_t(0));
        w.write(blocks);
        return b;
    };
    std::vector<uint8_t> b = form(0, 0);
    EXPECT_FALSE(parseGroupForm(b.data(), b.size(), 10, f, e));
    b = form(65537, 0);
    EXPECT_FALSE(parseGroupForm(b.data(), b.size(), 10, f, e));
    b = form(1, 0);
    EXPECT_TRUE(parseGroupForm(b.data(), b.size(), 10, f, e)) << e;
    // A block with a schema index out of range
    b = form(1, 1);
    BinaryWriter w(b);
    w.write(uint32_t(99));
    w.write(uint32_t(1));
    w.write(uint64_t(4));
    w.write(uint32_t(1));
    EXPECT_FALSE(parseGroupForm(b.data(), b.size(), 10, f, e));

    // A stored group whose values break the rules is kept at reload (never a crash): a local index out of range
    resetHooks();
    auto wp        = makeWorld();
    World& world   = *wp;
    const Entity p = addPlayer(world, kA);
    const Entity g = world.spawn<Grower>(kA + Vec3(50, 0, 0));
    world.step();
    teleport(world, p, kB);
    world.step();
    ASSERT_FALSE(world.alive(g));
    // Corrupt the slot of the first value of the first block (Transform) to 7 (the group has 1 entity)
    ChunkStore& store = world.chunkStore();
    ChunkEntry& entry = store.entriesForUpdate()[chunkOf(0, kA.x() + 50, kA.z())];
    ASSERT_EQ(entry.groups.size(), 1u);
    std::vector<uint8_t> bytes = *entry.groups[0].bytes;
    GroupForm gf;
    ASSERT_TRUE(parseGroupForm(bytes.data(), bytes.size(), 1000, gf, e));
    bytes[gf.blocks[0].begin] = 7;
    entry.groups[0].bytes     = std::make_shared<const std::vector<uint8_t>>(bytes);
    teleport(world, p, kA);
    world.step();
    EXPECT_EQ(store.stats().failedGroups, 1u);
    EXPECT_EQ(groupsAt(world, kA + Vec3(50, 0, 0)), 1u);
    // A blob with that group is refused at load? No: groups are decoded at reload (lazy); the blob loads
    World copy(unloadTestSetup());
    EXPECT_TRUE(copy.tryLoad(world.save()).ok());
}

// ==== Unload ========================================================================================================

TEST(Unload, NoOnDestroyAndLogFlag)
{
    resetHooks();
    auto wp        = makeWorld();
    World& w       = *wp;
    const Entity p = addPlayer(w, kA);
    const Entity g = w.spawn<Grower>(kA + Vec3(30, 0, 0));
    w.step();
    const int consumer = w.structuralLog().addConsumer();
    WorldApi api(w);
    const ApiResult c0 = api.call("changes_since", json11::Json(json11::Json::object{{"cursor", 0}}));
    ASSERT_TRUE(c0.ok) << c0.error;
    teleport(w, p, kB);
    w.step();
    // changes_since reports the unload as "unloaded" (not "destroyed")
    const ApiResult c1 = api.call("changes_since", json11::Json(json11::Json::object{{"cursor", c0.value["cursor"]}}));
    ASSERT_TRUE(c1.ok) << c1.error;
    bool reported = false;
    for (const auto& c : c1.value["changes"].array_items())
        reported = reported ||
                   (c["kind"].string_value() == "unloaded" && c["entity"]["index"].number_value() == double(g.index));
    EXPECT_TRUE(reported) << c1.value.dump();
    EXPECT_FALSE(w.alive(g));
    EXPECT_EQ(hooks().destroys, 0);
    int unloaded = 0, removed = 0;
    w.structuralLog().consume(consumer,
                              [&](const StructuralLog::Record& r)
                              {
                                  if (r.entity != g || r.added) return;
                                  removed++;
                                  unloaded += r.unloaded ? 1 : 0;
                              });
    w.structuralLog().removeConsumer(consumer);
    EXPECT_GT(removed, 0);
    EXPECT_EQ(unloaded, removed);
    EXPECT_TRUE(w.queryRadius(0, kA, 100).empty());
}

TEST(Unload, Exclusions)
{
    resetHooks();
    auto wp        = makeWorld(true, nullptr, true);
    World& w       = *wp;
    const Entity p = addPlayer(w, kA);
    // A group with a Player child (a centre itself: kept anyway), a client-owned root, an arrival, an alwaysSimulate
    const Entity withPlayerChild = w.spawn<Grower>(kB + Vec3(2000, 0, 0));
    const Entity child           = w.create();
    w.add<Parent>(child, Parent{withPlayerChild});
    w.add<Player>(child, Player{});
    const Entity owned = w.spawn<Grower>(kB);
    w.add<NetOwner>(owned, NetOwner{2});
    const Entity arriving = w.spawn<Grower>(kB + Vec3(0, 0, 300));
    w.add<PendingTransition>(arriving, PendingTransition{});
    const Entity always                     = w.spawn<Grower>(kB + Vec3(0, 0, 600));
    w.get<Behaviour>(always).alwaysSimulate = true;
    EXPECT_FALSE(w.entityUnloading().unloadable(p));
    EXPECT_FALSE(w.entityUnloading().unloadable(withPlayerChild));
    EXPECT_FALSE(w.entityUnloading().unloadable(owned));
    EXPECT_FALSE(w.entityUnloading().unloadable(arriving));
    EXPECT_FALSE(w.entityUnloading().unloadable(w.worldStateEntity()));
    EXPECT_TRUE(w.entityUnloading().unloadable(always));
    w.remove<PendingTransition>(arriving);
    w.step();
    EXPECT_TRUE(w.alive(p));
    EXPECT_TRUE(w.alive(withPlayerChild));
    EXPECT_TRUE(w.alive(owned));
    EXPECT_FALSE(w.alive(arriving));
    EXPECT_FALSE(w.alive(always));
    EXPECT_TRUE(w.alive(w.worldStateEntity()));
}

TEST(Unload, StreamingOffUnloadsNothing)
{
    auto wp  = makeWorld(false, nullptr, true);
    World& w = *wp;
    addPlayer(w, kA);
    const Entity far    = w.spawn<Grower>(kB + Vec3(2000, 0, 0));
    const Entity cellar = w.spawnIn<Grower>(1, Vec3(100, 0, 100));
    w.step(3);
    EXPECT_TRUE(w.alive(far));
    EXPECT_TRUE(w.alive(cellar));
    EXPECT_TRUE(w.spaceLoaded(1));
    EXPECT_TRUE(w.spaceStateEntity(1).valid());
    EXPECT_FALSE(w.chunkStoreView().hasContent());
    EXPECT_FALSE(hasStoredGroups(w.save()));
    EXPECT_FALSE(w.entityUnloading().attached());
}

TEST(Unload, NeverRequestedBandRoot)
{
    // The player at x 1100: chunk 5 (from 1280) is 180 m away: kept (192), never requested (128)
    auto wp        = makeWorld();
    World& w       = *wp;
    const Entity p = addPlayer(w, Vec3(1100, 0, 900));
    const Entity g = w.spawn<Grower>(Vec3(1300, 0, 900));
    w.step(2);
    EXPECT_TRUE(w.alive(g));
    EXPECT_FALSE(w.residency().requested(chunkOf(0, 1300, 900)));
    EXPECT_TRUE(w.residency().kept(chunkOf(0, 1300, 900)));
    // A copy loaded now; the player steps back 50 m in both: 230 m, not kept: unloaded in the same step
    World copy(unloadTestSetup());
    copy.setStreamingSynchronous(true);
    ASSERT_TRUE(copy.tryLoad(w.save()).ok());
    teleport(w, p, Vec3(1050, 0, 900));
    teleport(copy, p, Vec3(1050, 0, 900));
    w.step();
    copy.step();
    EXPECT_FALSE(w.alive(g));
    EXPECT_FALSE(copy.alive(g));
    EXPECT_EQ(w.chunkStoreView().stats().farUnloads, 1u);
    EXPECT_EQ(w.save(), copy.save());
}

TEST(Unload, FollowsKeptSet)
{
    resetHooks();
    auto wp               = makeWorld();
    World& w              = *wp;
    const Entity p        = addPlayer(w, Vec3(1100, 0, 900));
    const Entity g        = w.spawn<Grower>(Vec3(1300, 0, 900));
    const PersistentId id = w.makePersistent(g);
    w.step();
    // Between the requested and the kept distance: resident and frozen
    EXPECT_TRUE(w.alive(g));
    EXPECT_FALSE(w.simulated(g));
    // A player oscillating 20 m across the requested distance (1152 = 1280 - 128) causes no unload or reload
    for (int i = 0; i < 50; ++i)
    {
        teleport(w, p, Vec3(i % 2 ? 1162 : 1142, 0, 900));
        w.step();
    }
    EXPECT_EQ(w.chunkStoreView().stats().groupsUnloaded, 0u);
    EXPECT_TRUE(w.alive(g));
    // Beyond the kept distance: unloaded; back: reloaded from memory once ready
    teleport(w, p, Vec3(900, 0, 900));
    w.step();
    EXPECT_FALSE(w.resolve(id).valid());
    teleport(w, p, Vec3(1200, 0, 900));
    w.step();
    EXPECT_TRUE(w.resolve(id).valid());
    EXPECT_EQ(w.chunkStoreView().stats().recordsRead, 0u);
}

TEST(Unload, SimulatedAfterReload)
{
    auto wp                   = makeWorld();
    World& w                  = *wp;
    const Entity p            = addPlayer(w, kA);
    const PersistentId nearId = w.makePersistent(w.spawn<Grower>(kA + Vec3(20, 0, 0)));
    const PersistentId farId  = w.makePersistent(w.spawn<Grower>(kA + Vec3(110, 0, 0)));
    w.step();
    teleport(w, p, kB);
    w.step();
    ASSERT_EQ(w.chunkStoreView().totalGroups(), 2u);
    teleport(w, p, kA);
    w.step();
    ASSERT_TRUE(w.resolve(nearId).valid());
    ASSERT_TRUE(w.resolve(farId).valid());
    EXPECT_TRUE(w.simulated(w.resolve(nearId)));
    EXPECT_FALSE(w.simulated(w.resolve(farId)));
}

TEST(Unload, CatchUpElapsed)
{
    resetHooks();
    auto wp        = makeWorld();
    World& w       = *wp;
    const Entity p = addPlayer(w, kA);
    const Entity a = w.spawn<Grower>(kA + Vec3(30, 0, 0));
    const Entity b = w.spawn<Grower>(kA + Vec3(40, 0, 0));
    // Each references the other by PersistentId (resolved in onLoad: both groups exist before the hooks)
    const PersistentId ida = w.makePersistent(a), idb = w.makePersistent(b);
    w.get<GrowerState>(a).friendId = idb;
    w.get<GrowerState>(b).friendId = ida;
    w.step();
    teleport(w, p, kB);
    const uint64_t unloadStep = w.stepIndex();
    w.step();
    ASSERT_FALSE(w.alive(a));
    w.step(7);
    teleport(w, p, kA);
    hooks().order.clear();
    const uint64_t reloadStep = w.stepIndex();
    w.step();
    EXPECT_EQ(hooks().order, "cclluu");
    ASSERT_EQ(hooks().elapsed.size(), 2u);
    EXPECT_EQ(hooks().elapsed[0], double(reloadStep - unloadStep) * w.dt());
    EXPECT_EQ(hooks().friendsResolved, 2);
    EXPECT_EQ(hooks().spawns, 2);
    EXPECT_EQ(w.read<GrowerState>(w.resolve(ida)).grown, double(reloadStep - unloadStep) * w.dt());
    // An onCatchUp that spawns: the spawn applies at once; an onLoad that adds a saved component is refused (its object
    // is removed, as in loads)
    teleport(w, p, kB);
    w.step();
    hooks().spawnOnCatchUp = true;
    teleport(w, p, kA);
    w.step();
    EXPECT_EQ(hooks().spawns, 4);
    hooks().spawnOnCatchUp = false;
    teleport(w, p, kB);
    w.step();
    hooks().addSavedOnLoad = true;
    teleport(w, p, kA);
    w.step();
    hooks().addSavedOnLoad = false;
    EXPECT_EQ(w.chunkStoreView().stats().reloadHookErrors, 4u);
    EXPECT_FALSE(w.resolve(ida).valid());
}

namespace
{
bool g_throwInPhase7 = false;
Vec3 g_moveTo        = Vec3::Zero();
Entity g_mover;
void abortSystems(SystemList& s)
{
    s.add(Phase::Gameplay, "test_move",
          [](World& w, double)
          {
              if (g_mover.valid() && w.alive(g_mover)) w.get<Transform>(g_mover).teleport(g_moveTo, Quat::Identity());
              g_mover = Entity{};
          });
    s.add(Phase::Streaming, "test_throw",
          [](World&, double)
          {
              if (!g_throwInPhase7) return;
              g_throwInPhase7 = false;
              throw std::runtime_error("phase 7 aborted");
          });
}
}  // namespace

TEST(Unload, AbortedPhase7LosesNothing)
{
    resetHooks();
    auto wp                = makeWorld(true, abortSystems);
    World& w               = *wp;
    const Entity p         = addPlayer(w, kB);
    const PersistentId atB = w.makePersistent(w.spawn<Grower>(kB + Vec3(20, 0, 0)));
    w.step();
    teleport(w, p, kA);
    w.step();
    ASSERT_EQ(groupsAt(w, kB + Vec3(20, 0, 0)), 1u);
    const PersistentId atA = w.makePersistent(w.spawn<Grower>(kA + Vec3(20, 0, 0)));
    w.step();
    // In one step: phase 2 moves the player to B, phase 7 would reload B's group and unload A's root, then throws
    g_mover                  = p;
    g_moveTo                 = kB;
    g_throwInPhase7          = true;
    const size_t aliveBefore = w.registry().aliveCount();
    EXPECT_ANY_THROW(w.step());
    EXPECT_EQ(w.registry().aliveCount(), aliveBefore);
    EXPECT_TRUE(w.resolve(atA).valid());
    EXPECT_EQ(groupsAt(w, kB + Vec3(20, 0, 0)), 1u);
    // The next step does both
    w.step();
    EXPECT_TRUE(w.resolve(atB).valid());
    EXPECT_FALSE(w.resolve(atA).valid());
    EXPECT_EQ(groupsAt(w, kA + Vec3(20, 0, 0)), 1u);
    EXPECT_EQ(groupsAt(w, kB + Vec3(20, 0, 0)), 0u);
}

TEST(Unload, ReloadBeforeArrivalPhase1)
{
    resetHooks();
    auto wp               = makeWorld();
    World& w              = *wp;
    const Entity p        = addPlayer(w, kB);
    const PersistentId id = w.makePersistent(w.spawn<Grower>(kB + Vec3(20, 0, 0)));
    w.step();
    teleport(w, p, kA);
    w.step();
    ASSERT_FALSE(w.resolve(id).valid());
    // A between-step arrival back: its target is a centre at once; the step start reloads before phase 1
    ASSERT_TRUE(w.requestArrival(p, 0, kB));
    World copy(unloadTestSetup());
    copy.setStreamingSynchronous(true);
    ASSERT_TRUE(copy.tryLoad(w.save()).ok());
    hooks().order.clear();
    w.step();
    copy.step();
    EXPECT_TRUE(w.resolve(id).valid());
    EXPECT_FALSE(w.has<PendingTransition>(p));
    EXPECT_EQ(w.save(), copy.save());
}

TEST(Unload, ReloadHookThrows)
{
    resetHooks();
    auto wp        = makeWorld();
    World& w       = *wp;
    const Entity p = addPlayer(w, kA);
    const Entity a = w.spawn<Grower>(kA + Vec3(30, 0, 0));
    const Entity c = w.create();
    w.add<Parent>(c, Parent{a});
    const PersistentId ida = w.makePersistent(a), idc = w.makePersistent(c);
    const PersistentId idb = w.makePersistent(w.spawn<Grower>(kA + Vec3(40, 0, 0)));
    w.step();
    teleport(w, p, kB);
    w.step();
    hooks().throwOnCatchUp = true;
    hooks().catchUps       = 0;
    teleport(w, p, kA);
    w.step();
    // Both throw (the flag is global): each object is removed with its descendants, without hooks
    EXPECT_FALSE(w.resolve(ida).valid());
    EXPECT_FALSE(w.resolve(idc).valid());
    EXPECT_FALSE(w.resolve(idb).valid());
    EXPECT_EQ(w.chunkStoreView().stats().reloadHookErrors, 2u);
    EXPECT_EQ(hooks().destroys, 0);
    hooks().throwOnCatchUp = false;
}

TEST(Unload, FarSpawnUnloadsInPhase7)
{
    resetHooks();
    auto wp        = makeWorld();
    World& w       = *wp;
    const Entity p = addPlayer(w, kA);
    const Entity g = w.spawn<Grower>(kA + Vec3(2000, 0, 0));
    const Entity c = w.create();
    w.add<Parent>(c, Parent{g});
    const PersistentId id = w.makePersistent(g);
    w.step();
    EXPECT_FALSE(w.alive(g));
    EXPECT_FALSE(w.alive(c));
    EXPECT_EQ(hooks().spawns, 1);
    EXPECT_EQ(w.chunkStoreView().stats().entitiesUnloaded, 2u);
    ASSERT_TRUE(w.requestArrival(p, 0, kA + Vec3(2010, 0, 0)));
    w.step(2);
    ASSERT_TRUE(w.resolve(id).valid());
    EXPECT_EQ(hooks().catchUps, 1);
    EXPECT_EQ(hooks().spawns, 1);
}

TEST(Unload, CandidatesNotFullScan)
{
    auto wp  = makeWorld();
    World& w = *wp;
    addPlayer(w, kA);
    // 10^4 roots inside the kept set (no game objects: cheap)
    for (int i = 0; i < 10000; ++i)
    {
        const Entity e = w.create();
        teleport(w, e, kA + Vec3(double(i % 100) - 50, 0, double(i / 100) - 50));
    }
    // A second player far away: its Player removed later
    const Entity far = addPlayer(w, kB);
    w.step(2);
    const uint64_t before = w.chunkStoreView().stats().unloadCandidates;
    w.step(3);
    EXPECT_LT(w.chunkStoreView().stats().unloadCandidates - before, 30u);
    // The far root's Player is removed: unloaded in that step (no full scan)
    w.remove<Player>(far);
    w.step();
    EXPECT_FALSE(w.alive(far));
    // After a load: one full pass
    ASSERT_TRUE(w.tryLoad(w.save()).ok());
    const uint64_t loaded = w.chunkStoreView().stats().unloadCandidates;
    w.step();
    EXPECT_GE(w.chunkStoreView().stats().unloadCandidates - loaded, 10000u);
    // (the load marks every Transform changed: phase 6's list of this and the previous step names them once more)
    w.step();
    const uint64_t once = w.chunkStoreView().stats().unloadCandidates;
    w.step();
    EXPECT_LT(w.chunkStoreView().stats().unloadCandidates - once, 30u);
}

TEST(Unload, LoadEqualsPlain)
{
    auto wp        = makeWorld();
    World& w       = *wp;
    const Entity p = addPlayer(w, kA);
    for (int i = 0; i < 6; ++i) w.spawn<Grower>(kA + Vec3(30.0 * i, 0, 10));
    for (int i = 0; i < 4; ++i) w.spawn<Grower>(kB + Vec3(30.0 * i, 0, 10));
    w.step(2);
    // The player teleported 1 km right before the save
    teleport(w, p, kB);
    World copy(unloadTestSetup());
    copy.setStreamingSynchronous(true);
    ASSERT_TRUE(copy.tryLoad(w.save()).ok());
    for (int i = 0; i < 3; ++i)
    {
        w.step();
        copy.step();
        ASSERT_EQ(w.save(), copy.save()) << World::diff(w, copy);
    }
    EXPECT_GT(w.chunkStoreView().totalGroups(), 0u);
}

TEST(Unload, DeterministicHash)
{
    auto run = [](bool changed)
    {
        resetHooks();
        auto wp        = makeWorld();
        World& w       = *wp;
        const Entity p = addPlayer(w, kA);
        const Entity g = w.spawn<Grower>(kA + Vec3(30, 0, 0));
        if (changed) w.get<GrowerState>(g).grown = 1;
        std::vector<uint64_t> hashes;
        for (int i = 0; i < 12; ++i)
        {
            if (i == 3) teleport(w, p, kB);
            if (i == 8) teleport(w, p, kA);
            w.step();
            hashes.push_back(w.hash());
        }
        return hashes;
    };
    const std::vector<uint64_t> a = run(false), b = run(false), c = run(true);
    EXPECT_EQ(a, b);
    // The hash differs while the changed group is stored
    EXPECT_NE(a[5], c[5]);
}

TEST(Unload, BlobRoundTrip)
{
    auto wp        = makeWorld(true, nullptr, true);
    World& w       = *wp;
    const Entity p = addPlayer(w, kA);
    w.spawn<Grower>(kA + Vec3(30, 0, 0));
    w.spawn<Grower>(kA + Vec3(35, 0, 0));
    w.step();
    teleport(w, p, kB);
    w.step(2);
    ASSERT_EQ(w.chunkStoreView().totalGroups(), 2u);
    ASSERT_FALSE(w.spaceLoaded(1));
    const std::vector<uint8_t> blob = w.save();
    EXPECT_TRUE(hasStoredGroups(blob));
    WorldSetup s          = unloadTestSetup();
    s.environment.enabled = true;
    World copy(s);
    copy.setStreamingSynchronous(true);
    ASSERT_TRUE(copy.tryLoad(blob).ok());
    EXPECT_EQ(copy.save(), blob);
    EXPECT_EQ(copy.hash(), w.hash());
    EXPECT_FALSE(copy.spaceLoaded(1));
    // A changed group is reported as stored_group
    std::vector<StoredGroup>& groups = copy.chunkStore().entriesForUpdate()[chunkOf(0, kA.x() + 30, kA.z())].groups;
    ASSERT_EQ(groups.size(), 2u);
    std::swap(groups[0], groups[1]);
    const SaveDiff d = diffSaves(blob, copy.save(), DiffFilter());
    ASSERT_TRUE(d.error.empty()) << d.error;
    bool found = false;
    for (const SaveDifference& x : d.differences) found = found || x.kind == "stored_group";
    EXPECT_TRUE(found);
    // Back in both: the groups and the space reload
    teleport(w, p, kA);
    w.step();
    EXPECT_EQ(w.chunkStoreView().totalGroups(), 0u);
    EXPECT_FALSE(w.chunkStoreView().hasContent() && !w.spaceLoaded(0));
}

TEST(Unload, ReplicationDespawn)
{
    resetHooks();
    LoopbackOptions o;
    auto net               = LoopbackTransport::createNetwork(2, o);
    WorldSetup hs          = unloadTestSetup();
    hs.environment.enabled = true;
    hs.transport           = net[0].get();
    World host(hs);
    WorldSetup cs = hs;
    cs.transport  = net[1].get();
    cs.role       = NetRole::Client;
    cs.localPeer  = 1;
    World client(cs);
    host.setStreamingSynchronous(true);
    client.setStreamingSynchronous(true);
    auto step = [&](int n)
    {
        for (int i = 0; i < n; ++i)
        {
            host.step();
            client.step();
        }
    };
    const Entity p        = addPlayer(host, kA);
    const Entity g        = host.spawn<Grower>(kA + Vec3(30, 0, 0));
    const PersistentId id = host.makePersistent(g);
    step(3);
    const NetId netId = host.replication().netIdOf(g);
    ASSERT_NE(netId, 0u);
    ASSERT_TRUE(client.replication().entityOfNetId(netId).valid());
    // The space-state entity of space 1 (unloaded on the host: no Despawn, the client keeps it)
    ASSERT_FALSE(host.spaceLoaded(1));
    EXPECT_TRUE(client.spaceStateEntity(1).valid());
    teleport(host, p, kB);
    step(3);
    EXPECT_FALSE(client.replication().entityOfNetId(netId).valid());
    EXPECT_EQ(hooks().replicatedDestroys, 0);
    EXPECT_GT(client.replication().stats().despawnsApplied, 0u);
    EXPECT_EQ(host.replication().stats().destroysSent, 0u);
    // The reload is a new spawn with a new NetId
    teleport(host, p, kA);
    step(3);
    const Entity g2 = host.resolve(id);
    ASSERT_TRUE(g2.valid());
    const NetId again = host.replication().netIdOf(g2);
    EXPECT_NE(again, netId);
    EXPECT_TRUE(client.replication().entityOfNetId(again).valid());
}

// ==== SpaceUnload ====================================================================================================

TEST(SpaceUnload, StateEntityStoredAndRestored)
{
    auto wp        = makeWorld(true, nullptr, true);
    World& w       = *wp;
    const Entity p = addPlayer(w, kA);
    w.step();
    ASSERT_TRUE(w.spaceLoaded(0));
    EXPECT_FALSE(w.spaceLoaded(1));
    w.spaceState<Weather>(0).cloudiness = 0.75;
    const Entity g                      = w.spawn<Grower>(kA + Vec3(20, 0, 0));
    const PersistentId id               = w.makePersistent(g);
    // The player enters the cellar: space 0's roots unload, then (one step later) the space
    w.moveToSpace(p, 1, Vec3(100, 0, 100), Quat::Identity());
    w.step(3);
    EXPECT_TRUE(w.spaceLoaded(1));
    EXPECT_TRUE(w.spaceStateEntity(1).valid());
    EXPECT_FALSE(w.spaceLoaded(0));
    EXPECT_FALSE(w.spaceStateEntity(0).valid());
    EXPECT_FALSE(w.resolve(id).valid());
    EXPECT_TRUE(w.alive(w.worldStateEntity()));
    WeatherChange change;
    change.rain = 0.5;
    std::string error;
    EXPECT_FALSE(setWeather(w, 0, change, error));
    EXPECT_NE(error.find("not loaded"), std::string::npos);
    // Back: the Weather values are restored
    w.moveToSpace(p, 0, kA, Quat::Identity());
    w.step();
    ASSERT_TRUE(w.spaceLoaded(0));
    EXPECT_EQ(w.readSpaceState<Weather>(0).cloudiness, 0.75);
    EXPECT_TRUE(w.resolve(id).valid());
}
