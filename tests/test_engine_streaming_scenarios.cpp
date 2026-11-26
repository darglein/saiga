/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The streaming scenarios (docs/engine_design.md §12.3, §17.2; S24): registerStreamingScenarios on
// streamingScenarioSetup() (tests/streaming_test_setup.h; the runner's overrides make the required and requested
// distance 128 m and the kept distance 192 m), in every variant (plain, repeat with varyThreads, save/load with
// strictContinuation, host + client with delay 0 and 2 and asynchronous stepping). Their own binary: S25 and S26 add
// theirs. The content cache is shared by every run (once per process, as runScenarios shares one per setup).

#include "saiga/engine/components/LocalTransform.h"
#include "saiga/engine/components/PendingTransition.h"
#include "saiga/engine/components/RenderInstance.h"
#include "saiga/engine/components/SpaceLink.h"
#include "saiga/engine/components/Weather.h"
#include "saiga/engine/content/ChunkRecords.h"
#include "saiga/engine/content/ContentConvert.h"
#include "saiga/engine/content/ContentRenderSystem.h"
#include "saiga/engine/content/GenerationPipeline.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/save/ChunkStore.h"
#include "saiga/engine/sync/RecordingRenderSink.h"
#include "saiga/engine/terrain/TerrainEditStore.h"
#include "saiga/engine/tools/ScenarioList.h"
#include "saiga/engine/tools/ScenarioRunner.h"
#include "saiga/engine/tools/ScenarioWorld.h"
#include "saiga/engine/world/Events.h"

#include "gtest/gtest.h"

#include <set>

#include "streaming_test_setup.h"
#include "unload_test_setup.h"

using namespace Saiga;
using namespace Saiga::Engine;
using namespace StreamingTest;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);

// The setup of the streaming scenarios: one content cache for every run
WorldSetup streamingScenarioSetup(const std::string& = "")
{
    return streamingTestSetup(sharedStreamingCache());
}

// The world of a peer (0: the host)
World& peerWorld(ScenarioWorld& w, int peer)
{
    return peer == 0 ? w.world() : w.client(peer);
}

// The chunks whose square is within d of p in x/z
std::vector<ChunkId> chunksNear(const World& world, SpaceId space, const Vec3& p, double d)
{
    std::vector<ChunkId> out;
    const int32_t n = world.spaceTemplate(space).sizeChunks;
    for (int32_t z = std::max(0, int32_t(std::floor((p.z() - d) / 256)) - 1);
         z <= std::min(n - 1, int32_t(std::floor((p.z() + d) / 256))); ++z)
        for (int32_t x = std::max(0, int32_t(std::floor((p.x() - d) / 256)) - 1);
             x <= std::min(n - 1, int32_t(std::floor((p.x() + d) / 256))); ++x)
        {
            const double dx = std::max({double(x) * 256 - p.x(), 0.0, p.x() - double(x + 1) * 256});
            const double dz = std::max({double(z) * 256 - p.z(), 0.0, p.z() - double(z + 1) * 256});
            if (dx * dx + dz * dz > d * d) continue;
            ChunkId c;
            c.space = space;
            c.x     = x;
            c.z     = z;
            out.push_back(c);
        }
    return out;
}

using Json = json11::Json;

// The bound of a host change reaching the client (the scenarios' usual rule)
int syncBound(ScenarioWorld& w)
{
    return 2 * w.networkConfig().delaySteps + 10 + (w.networkConfig().async ? 6 : 0);
}

// Teleports a player on its owner's world; true once the host sees it within 2 m of p
bool teleportPlayerOnOwner(ScenarioWorld& w, Entity player, const Vec3& p)
{
    w.edit(player, [p](World& owner, Entity e) { owner.get<Transform>(e).teleport(p, Quat::Identity()); });
    return w.stepUntil(
        [&]
        {
            const Vec3 q = w.world().read<Transform>(player).position();
            return std::abs(q.x() - p.x()) < 2 && std::abs(q.z() - p.z()) < 2;
        },
        syncBound(w));
}

void registerStreamingScenarios(ScenarioList& s)
{
    // A player walks 256 m east, past site 0 toward site 1: on its owner every chunk within fullRadius is resident in
    // every step (synchronous: no stall); a site chunk is stamped from the step in which it became ready, never before
    // (host); the chunks behind it beyond the kept distance are None
    s.add("streaming_follows_player",
          [](ScenarioWorld& w)
          {
              const Vec3 start(1150, 0, 1400);
              const Entity player = w.spawnPlayer(start);
              // Player is the host's (replicated to the owner)
              w.world().get<Player>(player).moveSpeed = 32;
              w.step(2);
              std::set<ChunkId> stamped;
              int violations = 0, unstamped = 0;
              // S26: on a client a chunk is resident only with the host's snapshot (one round trip after its request,
              // the subscription): every chunk within fullRadius is resident at most 2 * delaySteps + 3 steps after it
              // was requested (asynchronous: the client's irregular steps add up to 4)
              std::map<ChunkId, uint64_t> requestedAt;
              const uint64_t bound = uint64_t(2 * w.networkConfig().delaySteps + 3 + (w.networkConfig().async ? 4 : 0));
              w.addStepListener(
                  [&]()
                  {
                      // The owner's world (the client in the network variants)
                      const int owner = w.networked() ? int(w.defaultPeer()) : 0;
                      World& ow       = peerWorld(w, owner);
                      const Entity pe = owner == 0 ? player : w.onClient(owner, player);
                      if (pe.valid() && ow.alive(pe))
                      {
                          const Transform& t = ow.read<Transform>(pe);
                          for (const ChunkId& c :
                               chunksNear(ow, t.space(), t.position(), ow.spaceTemplate(t.space()).fullRadius))
                          {
                              if (ow.residency().ready(c))
                              {
                                  requestedAt.erase(c);
                                  continue;
                              }
                              if (owner == 0 || !ow.residency().requested(c))
                              {
                                  violations++;
                                  continue;
                              }
                              const uint64_t first = requestedAt.emplace(c, w.steps()).first->second;
                              if (w.steps() - first > bound) violations++;
                          }
                      }
                      // Host: stamped only when ready; a ready site chunk within fullRadius is stamped
                      World& host = w.world();
                      for (const ChunkId& c : host.contentRecords().stamped())
                          if (stamped.insert(c).second && !host.residency().ready(c)) violations++;
                      const Transform& ht = host.read<Transform>(player);
                      for (const ChunkId& c : chunksNear(host, 0, ht.position(), host.spaceTemplate(0).fullRadius))
                          if (host.residency().ready(c) && !host.generation().stampList(c).empty() &&
                              !host.contentRecords().isStamped(c))
                              unstamped++;
                  });
              w.move(player, Vec3(1, 0, 0));
              w.check(
                  w.stepUntil([&] { return w.world().read<Transform>(player).position().x() >= start.x() + 256; }, 600),
                  "walked 256 m");
              w.move(player, Vec3::Zero());
              w.step(2);
              w.check(violations == 0,
                      "every chunk within fullRadius was resident and stamping followed the ready "
                      "set (" +
                          std::to_string(violations) + " violations)");
              w.check(unstamped == 0, "every ready site chunk was stamped (" + std::to_string(unstamped) + ")");
              w.check(w.world().residency().stats().streamingStalls == 0, "no stalls (synchronous)");
              // Behind it beyond the kept distance (192 m): None
              const Vec3 at = w.world().read<Transform>(player).position();
              ChunkId behind;
              behind.x = 3;
              behind.z = 5;
              w.check(at.x() - 1024 > w.world().spaceTemplate(0).keptDistance(),
                      "chunk (3, 5) is beyond the kept "
                      "distance");
              w.check(w.world().residency().state(behind) == ChunkState::None, "chunk (3, 5) is unloaded");
          });

    // Tickers at 40 m and 100 m (beyond the simulation radius 64, inside the requested 128 m): the far one does not
    // tick until the player walks to it; then the near one, now far, stops
    s.add("far_entities_freeze",
          [](ScenarioWorld& w)
          {
              const Vec3 start(1150, 0, 1400);
              const Entity player = w.spawnPlayer(start);
              // S25: by PersistentId (the host extrapolates an owner-side teleport for a few steps in the delayed
              // variants, S16: the transient pose can unload the tickers, which reload when the player arrives)
              const PersistentId nearT = w.world().makePersistent(w.spawn<Ticker>(start + Vec3(40, 0, 0)));
              const PersistentId farT  = w.world().makePersistent(w.spawn<Ticker>(start + Vec3(-100, 0, 0)));
              auto ticks               = [&](PersistentId id)
              {
                  const Entity e = w.world().resolve(id);
                  return e.valid() ? w.world().read<TickCount>(e).updates : 0u;
              };
              w.step(10);
              w.check(ticks(nearT) > 0, "the near one ticks");
              w.check(ticks(farT) == 0, "the far one is frozen");
              w.check(teleportPlayerOnOwner(w, player, start + Vec3(-100, 0, 5)), "at the far one");
              w.step(3);
              const uint32_t nearCount = ticks(nearT);
              const uint32_t farCount  = ticks(farT);
              w.step(10);
              w.check(ticks(farT) > farCount, "the far one ticks now");
              w.check(ticks(nearT) == nearCount, "the near one (now far) stopped");
          });

    // request_arrival 3 km away: the player is at the target exactly one step later (synchronous), the target's chunks
    // are resident and the old ones unloaded; SpaceTransitionEvent is readable in the step after
    s.add("arrival_far_teleport",
          [](ScenarioWorld& w)
          {
              const Vec3 start(1150, 0, 1400), target(4150, 0, 1400);
              const Entity player = w.spawnPlayer(start);
              w.step(5);
              const Json pj = Json::object{{"index", double(player.index)}, {"generation", double(player.generation)}};
              const ApiResult r =
                  w.call("request_arrival",
                         Json::object{{"entity", pj}, {"position", Json::array{target.x(), target.y(), target.z()}}});
              w.check(r.ok, "request_arrival: " + r.error);
              w.step();
              const Vec3 at = w.world().read<Transform>(player).position();
              w.check(std::abs(at.x() - target.x()) < 1e-6 && std::abs(at.z() - target.z()) < 1e-6,
                      "at the target one step later");
              w.check(!w.world().has<PendingTransition>(player), "the arrival completed");
              for (const ChunkId& c : chunksNear(w.world(), 0, target, w.world().spaceTemplate(0).requiredDistance()))
                  w.check(w.world().residency().ready(c), "the target's chunks are resident");
              w.check(w.world().residency().state(chunkOf(0, start.x(), start.z())) == ChunkState::None,
                      "the old chunk is unloaded");
              w.step();
              bool event = false;
              for (const SpaceTransitionEvent& e : w.world().events.read<SpaceTransitionEvent>())
                  event = event || (e.entity == player && !e.link.valid() && e.from == 0 && e.to == 0);
              w.check(event, "SpaceTransitionEvent (no link, from == to)");
              if (!w.networked()) return;
              World& client  = w.client(w.defaultPeer());
              const Entity c = w.onClient(w.defaultPeer(), player);
              w.check(w.stepUntil(
                          [&]
                          {
                              return c.valid() &&
                                     std::abs(client.read<Transform>(c).position().x() - target.x()) < 1e-3 &&
                                     !client.residency().clientAwaitsMove(c);
                          },
                          syncBound(w)),
                      "the owner's copy arrived");
              w.check(w.stepUntil([&] { return client.chunkReady(0, target.x(), target.z()); }, syncBound(w)),
                      "the client's chunk at the target is ready");
          });

    // An edit and a tombstone 2 km away (not resident), then an arrival there: every peer's terrain shows the edit and
    // every peer's recording sink hides the tombstoned instance
    s.add("edit_far_chunk_then_visit",
          [](ScenarioWorld& w)
          {
              const Vec3 start(1150, 0, 1400), far(3150, 0, 1400);
              const Entity player = w.spawnPlayer(start);
              w.step(3);
              const ChunkId c = chunkOf(0, far.x(), far.z());
              w.check(w.world().residency().state(c) == ChunkState::None, "the far chunk is not resident");
              // A tree of the chunk away from the brush
              const auto placement = w.world().generation().chunkPlacement(c, w.world().effectivePlacementVersion(c));
              InstanceKey tree;
              bool found = false;
              for (const TierOneInstance& t : placement->instances)
              {
                  const Vec3 p = contentPosition(c, t.x, t.y, t.z);
                  if (!found && t.key.generatorSlot == 0 && (p - far).norm() > 30)
                  {
                      tree  = t.key;
                      found = true;
                  }
              }
              if (!w.check(found, "a tree away from the brush")) return;
              TerrainBrush b;
              b.op       = TerrainBrush::Add;
              b.center   = far;
              b.radius   = 6;
              b.strength = 2;
              w.editTerrain(b);
              w.world().tombstone(c, tree);
              w.check(w.world().residency().state(c) == ChunkState::None, "the edit created no residency");
              const Json pj = Json::object{{"index", double(player.index)}, {"generation", double(player.generation)}};
              w.check(w.call("request_arrival",
                             Json::object{{"entity", pj}, {"position", Json::array{far.x() + 5, 0, far.z()}}})
                          .ok,
                      "request_arrival");
              const double edited = w.world().terrainHeight(0, far.x(), far.z());
              auto everyPeer      = [&]()
              {
                  for (int p = 0; p <= (w.networked() ? w.clientCount() : 0); ++p)
                  {
                      World& world = peerWorld(w, p);
                      if (world.terrainHeight(0, far.x(), far.z()) != edited || !world.isTombstoned(c, tree))
                          return false;
                      RegionId r;
                      r.space                                     = 0;
                      r.level                                     = 0;
                      r.x                                         = c.x;
                      r.z                                         = c.z;
                      const RecordingRenderSink::InstanceSet* set = w.renderSink(p).instanceSetAt(r);
                      if (!set) return false;
                      std::vector<InstanceKey> keys;
                      world.contentRenderForUpdate().buildInstances(c, world.effectivePlacementVersion(c), &keys);
                      const size_t i = size_t(std::lower_bound(keys.begin(), keys.end(), tree) - keys.begin());
                      if (i >= set->instances.size() || set->instances[i].scale != 0) return false;
                  }
                  return true;
              };
              w.check(w.stepUntil(everyPeer, 4 * syncBound(w) + 20),
                      "every peer shows the edit and hides the tombstoned tree");
              w.check(w.world().residency().ready(c), "the visited chunk is resident");
          });

    // S26: the host edits the player's chunk C every 3 steps for 30 steps while the player is moved 600 m away (beyond
    // the kept distance of 192 m: C is unloaded and, on a client, unsubscribed and dropped) and back, dwelling 16 steps
    // at each end (more than twice the 2 * delaySteps + 3 sync bound at delay 2): edits are in flight during the
    // unsubscribe, the resubscription and the pending snapshot; every peer ends with the edited surface (the final
    // comparison checks C's sequence and dense delta on the clients)
    s.add("streaming_reload_during_edit",
          [](ScenarioWorld& w)
          {
              const Vec3 start(1150, 0, 1400), far(1750, 0, 1400);
              const Entity player = w.spawnPlayer(start);
              w.step(3);
              int steps = 0, edits = 0;
              auto step = [&]()
              {
                  if (steps % 3 == 0 && edits < 10)
                  {
                      TerrainBrush b;
                      b.op       = TerrainBrush::Add;
                      b.center   = start + Vec3(10, 0, 0);
                      b.radius   = 8;
                      b.strength = 0.5;
                      w.editTerrain(b);
                      edits++;
                  }
                  w.step();
                  steps++;
              };
              auto arriveAt = [&](const Vec3& p)
              {
                  w.check(w.world().requestArrival(player, 0, p), "requestArrival");
                  for (int k = 0; k < 64; ++k)
                  {
                      const Vec3 at = w.world().read<Transform>(player).position();
                      if (std::abs(at.x() - p.x()) < 1e-6 && std::abs(at.z() - p.z()) < 1e-6) break;
                      step();
                  }
                  for (int k = 0; k < 16; ++k) step();
              };
              step();
              arriveAt(far);
              const ChunkId c = chunkOf(0, start.x(), start.z());
              w.check(w.world().residency().state(c) == ChunkState::None, "C is unloaded on the host");
              for (int p = 1; p <= (w.networked() ? w.clientCount() : 0); ++p)
                  w.check(w.client(p).replication().chunkSync().state(c) == ClientChunkSync::State::None &&
                              w.client(p).terrainEdits().entry(c) == nullptr,
                          "C is unsubscribed and dropped on the client");
              arriveAt(start);
              while (edits < 10) step();
              w.check(edits == 10, "ten edits");
              w.step(syncBound(w));
              const double edited = w.world().terrainHeight(0, start.x() + 10, start.z());
              for (int p = 1; p <= (w.networked() ? w.clientCount() : 0); ++p)
              {
                  w.check(w.terrainHeight(start.x() + 10, start.z(), p) == edited,
                          "the client's height at the edit equals the host's");
                  w.check(w.client(p).chunkSynced(c), "C is synced on the client");
              }
          });

    // S21's link chain into test_dungeon on the streaming setup: the site's stamped entrance; a transition requested
    // between steps moves the player in the next step (S21's latency), the dungeon's island is resident
    s.add("space_link_with_streaming",
          [](ScenarioWorld& w)
          {
              const Site ruin = w.world().generation().space(0).plan->sites[0];
              const Vec3 site(double(ruin.x), 0, double(ruin.z));
              const Entity player = w.spawnPlayer(site + Vec3(0, 0, -60));
              std::vector<Entity> links;
              w.check(w.stepUntil(
                          [&]
                          {
                              links.clear();
                              w.world().viewAll<const SpaceLink, const Transform>().each(
                                  [&](Entity e, const SpaceLink&, const Transform& t)
                                  {
                                      if (t.space() == 0) links.push_back(e);
                                  });
                              return links.size() == 1;
                          },
                          10),
                      "the entrance is stamped");
              if (links.size() != 1) return;
              const SpaceId target = w.world().read<SpaceLink>(links[0]).target;
              const uint64_t at    = w.steps();
              w.check(w.world().requestTransition(player, links[0]), "the transition is accepted");
              w.check(w.stepUntil([&] { return w.world().read<Transform>(player).space() == target; }, 10),
                      "in the dungeon");
              w.check(w.steps() == at + 1, "moved in the next step (S21's latency)");
              w.check(w.world().chunkReady(target, w.world().read<Transform>(player).position().x(),
                                           w.world().read<Transform>(player).position().z()),
                      "the dungeon's chunk is ready");
          });
}

// ---- S25: the unload scenarios (registerUnloadScenarios) on tests/unload_test_setup.h (flat spaces, S24's scenario
// distances: a group unloads once its chunk is 192 m from the last player), with the environment (space-state
// entities) for the space unload

WorldSetup unloadScenarioSetup()
{
    WorldSetup s          = UnloadTest::unloadTestSetup();
    s.environment.enabled = true;
    return s;
}

// The player's position on the host
bool playerAt(ScenarioWorld& w, Entity player, SpaceId space, const Vec3& p)
{
    const Transform& t = w.world().read<Transform>(player);
    return t.space() == space && std::abs(t.position().x() - p.x()) < 1 && std::abs(t.position().z() - p.z()) < 1;
}

// The replicated entity of a NetId is gone on every client (or never arrived)
bool goneOnClients(ScenarioWorld& w, NetId id)
{
    for (int c = 1; c <= w.clientCount(); ++c)
        if (w.client(c).replication().entityOfNetId(id).valid()) return false;
    return true;
}

void registerUnloadScenarios(ScenarioList& s)
{
    using UnloadTest::Grower;
    using UnloadTest::GrowerState;
    // The acceptance (§22): a group (root, child, an in-group reference, an outside reference, a PersistentId) 100 m
    // from the player; the player arrives 1 km away: the group is stored, resolve is empty, on clients the entity is
    // gone; the player returns: the group is resident again with remapped references, the outside reference null and
    // grown advanced by the elapsed time
    s.add("chunk_unload_reload",
          [](ScenarioWorld& w)
          {
              const Vec3 start(1000, 0, 1000), away(2000, 0, 1000);
              const Entity player = w.spawnPlayer(start);
              const Entity root   = w.spawn<Grower>(start + Vec3(100, 0, 0));
              const Entity child  = w.world().create();
              w.world().add<LocalTransform>(child);
              w.world().add<Parent>(child, Parent{root});
              w.world().get<GrowerState>(root).buddy   = child;
              w.world().get<GrowerState>(root).outside = player;
              const PersistentId id                    = w.world().makePersistent(root);
              w.step(3);
              const NetId netId = w.networked() ? w.world().replication().netIdOf(root) : 0;
              w.check(w.world().requestArrival(player, 0, away), "requestArrival away");
              w.check(w.stepUntil([&] { return playerAt(w, player, 0, away); }, 30), "arrived away");
              w.step(2);
              const ChunkId chunk = UnloadTest::chunkOf(0, start.x() + 100, start.z());
              w.check(w.world().chunkStoreView().groupCount(chunk) == 1, "the group is stored");
              w.check(!w.world().resolve(id).valid(), "resolve is empty while stored");
              if (w.networked())
                  w.check(w.stepUntil([&] { return goneOnClients(w, netId); }, syncBound(w)), "gone on the clients");
              w.step(10);
              w.check(w.world().requestArrival(player, 0, start), "requestArrival back");
              w.check(w.stepUntil([&] { return playerAt(w, player, 0, start); }, 30), "arrived back");
              const Entity r = w.world().resolve(id);
              if (!w.check(r.valid(), "the group is resident again")) return;
              const GrowerState& g = w.world().read<GrowerState>(r);
              w.check(g.grown > 0.1, "grown advanced by the elapsed time");
              w.check(w.world().alive(g.buddy) && w.world().read<Parent>(g.buddy).entity == r,
                      "the in-group reference is remapped");
              w.check(!g.outside.valid(), "the outside reference is null");
              w.check(w.world().chunkStoreView().groupCount(chunk) == 0, "the chunk keeps no groups");
          });
    // A Grower spawned 2 km from the player is unloaded in its spawning step and reloads when the player arrives
    s.add("far_spawn_reload",
          [](ScenarioWorld& w)
          {
              const Vec3 start(1000, 0, 1000), far(3000, 0, 1000);
              const Entity player   = w.spawnPlayer(start);
              const PersistentId id = w.world().makePersistent(w.spawn<Grower>(far));
              w.step();
              w.check(!w.world().resolve(id).valid(), "unloaded in its spawning step");
              w.check(w.world().chunkStoreView().groupCount(UnloadTest::chunkOf(0, far.x(), far.z())) == 1,
                      "stored at its chunk");
              w.step(5);
              w.check(w.world().requestArrival(player, 0, far + Vec3(10, 0, 0)), "requestArrival");
              w.check(w.stepUntil([&] { return w.world().resolve(id).valid(); }, 30), "reloaded");
              w.step(2);
              const Entity e = w.world().resolve(id);
              w.check(e.valid() && w.world().read<GrowerState>(e).catchUps == 1, "onCatchUp ran once");
          });
    // The player enters the cellar through a SpaceLink: space 0 unloads with its Grower and its state entity; back in
    // space 0 the player finds them with caught-up grown and the Weather restored
    s.add("space_unload_reload",
          [](ScenarioWorld& w)
          {
              const Vec3 start(1000, 0, 1000);
              const Entity player = w.spawnPlayer(start);
              const Entity link   = w.world().createIn(0);
              w.world().get<Transform>(link).teleport(start + Vec3(5, 0, 0), Quat::Identity());
              SpaceLink l;
              l.target  = 1;
              l.arrival = Vec3(100, 0, 100);
              w.world().add<SpaceLink>(link, l);
              const PersistentId id = w.world().makePersistent(w.spawn<Grower>(start + Vec3(20, 0, 0)));
              w.world().spaceState<Weather>(0).cloudiness = 0.75;
              w.step(3);
              w.check(w.world().requestTransition(player, link), "requestTransition into the cellar");
              w.check(w.stepUntil([&] { return w.world().read<Transform>(player).space() == 1; }, 30), "in the cellar");
              w.check(w.stepUntil([&] { return !w.world().spaceLoaded(0); }, 10), "space 0 unloads");
              w.check(!w.world().spaceStateEntity(0).valid(), "its state entity is stored");
              w.check(!w.world().resolve(id).valid(), "its Grower is stored");
              w.step(10);
              w.check(w.world().requestArrival(player, 0, start), "requestArrival back");
              w.check(w.stepUntil([&] { return playerAt(w, player, 0, start); }, 30), "back in space 0");
              w.check(w.world().spaceLoaded(0), "space 0 is loaded");
              w.check(w.world().readSpaceState<Weather>(0).cloudiness == 0.75, "the Weather is restored");
              const Entity e = w.world().resolve(id);
              w.check(e.valid() && w.world().read<GrowerState>(e).grown > 0.1, "the Grower caught up");
          });
}

struct Case
{
    std::string scenario;
    ScenarioVariant variant;
    bool unload = false;
};

std::vector<Case> allCases()
{
    std::vector<Case> cases;
    {
        ScenarioList list;
        registerUnloadScenarios(list);
        ScenarioOptions options;
        options.networkVariants = true;
        for (const auto& e : list.entries())
            for (ScenarioVariant v : variantsFor(e, options)) cases.push_back(Case{e.name, v, true});
    }
    ScenarioList list;
    registerStreamingScenarios(list);
    ScenarioOptions options;
    options.networkVariants = true;
    for (const auto& e : list.entries())
        for (ScenarioVariant v : variantsFor(e, options)) cases.push_back(Case{e.name, v});
    return cases;
}
}  // namespace

class StreamingScenario : public ::testing::TestWithParam<Case>
{
};

TEST_P(StreamingScenario, Passes)
{
    ScenarioList list;
    if (GetParam().unload)
        registerUnloadScenarios(list);
    else
        registerStreamingScenarios(list);
    const ScenarioList::Entry* entry = list.find(GetParam().scenario);
    ASSERT_TRUE(entry);
    ScenarioOptions options;
    options.varyThreads        = true;
    options.strictContinuation = true;
    options.networkVariants    = true;
    const WorldSetup setup     = GetParam().unload ? unloadScenarioSetup() : streamingScenarioSetup(entry->name);
    const ScenarioResult r     = runScenario(setup, *entry, GetParam().variant, options);
    EXPECT_TRUE(r.passed) << r.report;
}

INSTANTIATE_TEST_SUITE_P(Streaming, StreamingScenario, ::testing::ValuesIn(allCases()),
                         [](const ::testing::TestParamInfo<Case>& info)
                         { return info.param.scenario + "_" + variantName(info.param.variant); });
