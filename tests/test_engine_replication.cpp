/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Replication (docs/engine_design.md §13.3-§13.5; stage S16): the network visitors, roles and RunOn filtering, client
// restrictions, ownership and ownedBy, spawns, destroys, hierarchy and component changes, snapshots and the
// interpolation buffer, owner state and owner commands, PlayerAction messages, the host clock, resync after a load,
// aborted steps and late joiners, over LoopbackTransport.

#include "saiga/engine/api/WorldApi.h"
#include "saiga/engine/components/CharacterIntent.h"
#include "saiga/engine/components/Health.h"
#include "saiga/engine/components/LocalTransform.h"
#include "saiga/engine/components/NetLocal.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/PlayerInput.h"
#include "saiga/engine/components/RenderPose.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/ecs/FieldList.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/gameplay/GameObject.h"
#include "saiga/engine/net/EngineMessages.h"
#include "saiga/engine/net/HostClock.h"
#include "saiga/engine/net/LoopbackTransport.h"
#include "saiga/engine/net/NetVisitors.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/world/World.h"

#include "gtest/gtest.h"

#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>

using namespace Saiga;
using namespace Saiga::Engine;
using Json = json11::Json;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);

enum class TestComponent : uint8_t
{
    Inventory = kFirstGameComponent,
    Target,
    Score,
    Weather,
    Big,
    Count
};
enum class TestClass : uint16_t
{
    Probe = kFirstGameClass,
    Thrower,
    Count
};

// HostReliable: sent with the spawn and when its bytes change
struct Inventory
{
    static constexpr const char* name = "Inventory";
    static constexpr ComponentId id   = ComponentId(TestComponent::Inventory);
    static constexpr uint32_t version = 1;
    std::vector<uint32_t> items;
    Entity owner;
    template <class V>
    void visit(V& v)
    {
        v("items", items);
        v("owner", owner);
    }
};
// An Entity field in a Host component
struct Target
{
    static constexpr const char* name = "Target";
    static constexpr ComponentId id   = ComponentId(TestComponent::Target);
    static constexpr uint32_t version = 1;
    Entity entity;
    double strength = 0;
    template <class V>
    void visit(V& v)
    {
        v("entity", entity);
        v("strength", strength, Quantize{0.01});
    }
};
struct Score
{
    static constexpr const char* name = "Score";
    static constexpr ComponentId id   = ComponentId(TestComponent::Score);
    static constexpr uint32_t version = 1;
    int32_t points                    = 0;
    template <class V>
    void visit(V& v)
    {
        v("points", points);
    }
};
struct Weather
{
    static constexpr const char* name = "Weather";
    static constexpr ComponentId id   = ComponentId(TestComponent::Weather);
    static constexpr uint32_t version = 1;
    double rain                       = 0;
    template <class V>
    void visit(V& v)
    {
        v("rain", rain);
    }
};
// Too large for a snapshot (the 1100-byte bound)
struct Big
{
    static constexpr const char* name = "Big";
    static constexpr ComponentId id   = ComponentId(TestComponent::Big);
    static constexpr uint32_t version = 1;
    std::vector<uint8_t> data;
    template <class V>
    void visit(V& v)
    {
        v("data", data);
    }
};

// The hooks a peer called (by the peer's role)
struct Hooks
{
    int configure = 0, spawn = 0, replicatedSpawn = 0, replicatedDestroy = 0, destroy = 0, update = 0;
};
std::map<bool, Hooks> g_hooks;  // by isHost

class Probe final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(TestClass::Probe);
    static constexpr const char* name = "Probe";
    void configure() override { g_hooks[world().isHost()].configure++; }
    void onSpawn() override { g_hooks[world().isHost()].spawn++; }
    void onReplicatedSpawn() override { g_hooks[world().isHost()].replicatedSpawn++; }
    void onReplicatedDestroy() override
    {
        g_hooks[world().isHost()].replicatedDestroy++;
        if (throwOnReplicatedDestroy) throw std::runtime_error("replicated destroy boom");
    }
    static inline bool throwOnReplicatedDestroy = false;
    void onDestroy() override { g_hooks[world().isHost()].destroy++; }
    void update(double) override { g_hooks[world().isHost()].update++; }
};
class Thrower final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(TestClass::Thrower);
    static constexpr const char* name = "Thrower";
    void onReplicatedSpawn() override { throw std::runtime_error("replicated spawn boom"); }
};

std::map<std::pair<bool, int>, int> g_ran;  // (isHost, RunOn) -> runs
void hostSystem(World& w, double)
{
    g_ran[{w.isHost(), 0}]++;
}
void clientSystem(World& w, double)
{
    g_ran[{w.isHost(), 1}]++;
}
void bothSystem(World& w, double)
{
    g_ran[{w.isHost(), 2}]++;
}

// Test hooks: an ownerTeleport during phase 4, an exception in phase 4 or 6
Entity g_teleportInStep;
Phase g_throwIn = Phase::None;
void teleportSystem(World& w, double)
{
    if (w.alive(g_teleportInStep)) w.ownerTeleport(g_teleportInStep, Vec3(7, 0, 0), Quat::Identity());
}
void throwSystem(World& w, double)
{
    if (w.currentPhase() == g_throwIn) throw std::runtime_error("test abort");
}

WorldSetup baseSetup()
{
    WorldSetup s;
    s.components = [](Registry& r)
    {
        r.registerComponent<Inventory>(Replicate::HostReliable);
        r.registerComponent<Target>(Replicate::Host);
        r.registerGlobal<Score>(Replicate::Host);
        r.registerSpaceState<Weather>(Replicate::Host);
        r.registerComponent<Big>(Replicate::Host);
    };
    s.classes = [](ClassList& c)
    {
        c.add<Probe>();
        c.add<Thrower>();
    };
    s.systems = [](SystemList& l)
    {
        l.add(Phase::Gameplay, "test_host", hostSystem, RunOn::Host);
        l.add(Phase::Gameplay, "test_client", clientSystem, RunOn::Client);
        l.add(Phase::Gameplay, "test_both", bothSystem, RunOn::Both);
        l.add(Phase::Gameplay, "test_teleport", teleportSystem, RunOn::Host);
        l.add(Phase::Gameplay, "test_throw4", throwSystem, RunOn::Host);
        l.add(Phase::Transform, "test_throw6", throwSystem, RunOn::Host);
    };
    return s;
}

// A host and clients over loopback; step() steps the host, then every client
struct Net
{
    std::vector<std::unique_ptr<LoopbackTransport>> t;
    std::unique_ptr<World> host;
    std::vector<std::unique_ptr<World>> clients;
    explicit Net(int n = 1, int delay = 0, bool createClients = true)
    {
        LoopbackOptions o;
        o.delayRounds = delay;
        t             = LoopbackTransport::createNetwork(1 + n, o);
        WorldSetup s  = baseSetup();
        s.transport   = t[0].get();
        host          = std::make_unique<World>(s);
        if (createClients)
            for (int i = 1; i <= n; ++i) addClient(i);
    }
    void addClient(int i)
    {
        WorldSetup s = baseSetup();
        s.transport  = t[size_t(i)].get();
        s.role       = NetRole::Client;
        s.localPeer  = uint8_t(i);
        clients.push_back(std::make_unique<World>(s));
    }
    World& client(int i = 1) { return *clients[size_t(i - 1)]; }
    void step(int n = 1)
    {
        for (int k = 0; k < n; ++k)
        {
            host->step();
            for (auto& c : clients) c->step();
        }
    }
    Entity onClient(Entity h, int i = 1)
    {
        return client(i).replication().entityOfNetId(host->replication().netIdOf(h));
    }
};

Entity makePlayer(World& w, uint8_t peer, const Vec3& p = Vec3::Zero())
{
    const Entity e = w.create();
    w.get<Transform>(e).setPosition(p);
    Player pl;
    pl.peer = peer;
    w.add<Player>(e, pl);
    w.add<PlayerInput>(e);
    w.add<CharacterIntent>(e);
    if (peer != 0) w.setOwner(e, peer);
    return e;
}

// A fixed resolver for the visitor tests
struct MapResolver : NetIdResolver
{
    std::map<uint32_t, NetId> ids;
    std::map<NetId, Entity> entities;
    NetId netIdOf(Entity e) const override
    {
        auto it = ids.find(e.index);
        return it == ids.end() ? 0 : it->second;
    }
    Entity entityOfNetId(NetId id) const override
    {
        auto it = entities.find(id);
        return it == entities.end() ? Entity{} : it->second;
    }
};
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// Network visitors

TEST(NetVisitors, EntitiesAsNetIdsAndUnresolvedTargets)
{
    MapResolver host;
    host.ids[5] = 200000;
    Target t;
    t.entity   = Entity{5, 3};
    t.strength = 1.234;
    std::vector<uint8_t> bytes;
    NetWriter w(bytes, &host);
    w.write(t);
    ASSERT_TRUE(w.ok());
    // On a client that knows the NetId
    MapResolver client;
    client.entities[200000] = Entity{9, 1};
    Target got;
    NetReader r(bytes.data(), bytes.size(), &client);
    r.read(got);
    ASSERT_TRUE(r.ok() && r.atEnd());
    EXPECT_FALSE(r.unresolved());
    EXPECT_EQ(got.entity, (Entity{9, 1}));
    EXPECT_NEAR(got.strength, 1.23, 1e-12);  // quantized to 0.01
    // The target has not arrived: Entity{} and unresolved
    MapResolver empty;
    NetReader r2(bytes.data(), bytes.size(), &empty);
    Target got2;
    r2.read(got2);
    EXPECT_TRUE(r2.ok());
    EXPECT_TRUE(r2.unresolved());
    EXPECT_FALSE(got2.entity.valid());
    // An entity without a NetId (local) travels as 0: none, not unresolved
    Target local;
    local.entity = Entity{77, 1};
    bytes.clear();
    NetWriter w3(bytes, &host);
    w3.write(local);
    NetReader r3(bytes.data(), bytes.size(), &client);
    Target got3;
    r3.read(got3);
    EXPECT_FALSE(r3.unresolved());
    EXPECT_FALSE(got3.entity.valid());
}

TEST(NetVisitors, QuantizeRoundTripsAtTheStepBoundaries)
{
    // round(v / q): values on and next to the half steps, negative values, zigzag
    for (double v : {0.0, 0.005, 0.0049999, -0.005, -0.0050001, 1.0, -1.0, 123.456, -987.654, 1e6})
    {
        Target t;
        t.strength = v;
        std::vector<uint8_t> bytes;
        NetWriter w(bytes, nullptr);
        w.write(t);
        Target got;
        NetReader r(bytes.data(), bytes.size(), nullptr);
        r.read(got);
        ASSERT_TRUE(r.ok());
        EXPECT_NEAR(got.strength, std::llround(v / 0.01) * 0.01, 1e-9) << v;
        EXPECT_LE(std::abs(got.strength - v), 0.005 + 1e-12) << v;
    }
    // Transform: position 1e-4, rotation 1e-5 and normalized after reading; saves keep the exact values
    Transform t;
    t.teleport(Vec3(1.00004, -2.00006, 3), Quat(0.7, 0.1, 0.2, 0.3).normalized());
    std::vector<uint8_t> bytes;
    NetWriter w(bytes, nullptr);
    t.visit(w);
    Transform got;
    NetReader r(bytes.data(), bytes.size(), nullptr);
    got.visit(r);
    ASSERT_TRUE(r.ok() && r.atEnd());
    EXPECT_NEAR(got.position().x(), 1.0, 1e-12);
    EXPECT_NEAR(got.position().y(), -2.0001, 1e-12);
    const Quat q = got.rotation();
    EXPECT_NEAR(q.w() * q.w() + q.x() * q.x() + q.y() * q.y() + q.z() * q.z(), 1.0, 1e-12);
    EXPECT_NEAR(q.x(), t.rotation().x(), 2e-5);
    EXPECT_LT(bytes.size(), toBytes(t).size());  // smaller than the binary form
    // The field list keeps the step; the save schema does not change
    const auto fields = describeFields<Transform>();
    ASSERT_EQ(fields.size(), 4u);
    EXPECT_EQ(fields[1].quantize, 1e-4);
    EXPECT_EQ(fields[2].quantize, 1e-5);
    EXPECT_EQ(fields[3].quantize, 0.0);
}

TEST(NetVisitors, MalformedInputIsAnError)
{
    const std::vector<uint8_t> bad{0xFF, 0xFF};
    Target t;
    NetReader r(bad.data(), bad.size(), nullptr);
    r.read(t);
    EXPECT_FALSE(r.ok());
}

// The per-entity bound (§13.4): stops the program in every build, naming the entity and its largest component
TEST(ReplicationDeathTest, EntityBound)
{
    auto run = []()
    {
        Net n;
        const Entity e = n.host->create();
        n.host->add<Big>(e, Big{std::vector<uint8_t>(2000, 1)});
        n.step(2);
    };
    EXPECT_DEATH(run(), "largest component is Big");
}

TEST(Replication, CompressedOrRaw)
{
    Net n;
    // Many idle entities: their records compress
    for (int i = 0; i < 300; ++i) n.host->create();
    n.step(2);
    EXPECT_GT(n.host->replication().stats().compressedMessages, 0u);
    // Random bytes do not shrink: sent raw (still decodable). Entities of about a message each, random poses too
    Net m;
    Rng rng(3);
    Entity e;
    Big b;
    for (int k = 0; k < 3; ++k)
    {
        e = m.host->create();
        m.host->get<Transform>(e).setPosition(Vec3(rng.uniform(), rng.uniform(), rng.uniform()) * 1000);
        m.host->get<Transform>(e).setScale(Vec3(rng.uniform(), rng.uniform(), rng.uniform()));
        b.data.clear();
        for (int i = 0; i < 1000; ++i) b.data.push_back(uint8_t(rng.nextU32()));
        m.host->add<Big>(e, b);
    }
    m.step(2);
    const auto& stats = m.host->replication().stats();
    EXPECT_LT(stats.compressedMessages, stats.snapshotMessagesSent);
    const Entity c = m.onClient(e);
    ASSERT_TRUE(c.valid());
    EXPECT_EQ(m.client().read<Big>(c).data, b.data);
}

// ---------------------------------------------------------------------------------------------------------------------
// Roles and restrictions

TEST(Roles, RunOnFiltering)
{
    g_ran.clear();
    Net n;
    n.step(3);
    EXPECT_EQ((g_ran[{true, 0}]), 3);
    EXPECT_EQ((g_ran[{true, 1}]), 0);
    EXPECT_EQ((g_ran[{true, 2}]), 3);
    EXPECT_EQ((g_ran[{false, 0}]), 0);
    EXPECT_EQ((g_ran[{false, 1}]), 3);
    EXPECT_EQ((g_ran[{false, 2}]), 3);
    // The engine systems' RunOn (§8.2)
    std::map<std::string, RunOn> engine;
    for (int p = kFirstPhase; p <= kLastPhase; ++p)
        for (const auto& s : n.host->systemList().systems(Phase(p))) engine[s.name] = s.runOn;
    EXPECT_EQ(engine["behaviour_events"], RunOn::Host);
    EXPECT_EQ(engine["behaviour"], RunOn::Host);
    EXPECT_EQ(engine["owner_pose"], RunOn::Host);
    for (const char* both : {"player_control", "character_move", "transform", "spatial_grid", "tick"})
        EXPECT_EQ(engine[both], RunOn::Both) << both;
    // Game object hooks on a client: configure and onReplicatedSpawn only (no update)
    g_hooks.clear();
    n.host->setTickEveryStep(true);
    n.client().setTickEveryStep(true);
    n.host->spawn<Probe>(Vec3(1, 0, 0));
    n.step(5);
    EXPECT_EQ(g_hooks[true].update, 5);
    EXPECT_EQ(g_hooks[false].update, 0);
    EXPECT_EQ(g_hooks[false].configure, 1);
    EXPECT_EQ(g_hooks[false].replicatedSpawn, 1);
    EXPECT_EQ(g_hooks[false].spawn, 0);
}

TEST(Roles, OwnedByWithOwnersAndDescendants)
{
    World w(baseSetup());
    const Entity a = makePlayer(w, 1), b = makePlayer(w, 2), c = w.create(), child = w.create();
    w.add<Parent>(child, Parent{a});
    auto owned = [&](uint8_t peer)
    {
        std::vector<Entity> out;
        w.ownedBy<const Transform>(peer).each([&](Entity e, const Transform&) { out.push_back(e); });
        return out;
    };
    EXPECT_EQ(owned(1), (std::vector<Entity>{a, child}));
    EXPECT_EQ(owned(2), (std::vector<Entity>{b}));
    EXPECT_EQ(w.ownerOf(c), 0);
    // The cached owner follows a reparenting
    w.remove<Parent>(child);
    w.add<Parent>(child, Parent{b});
    EXPECT_EQ(owned(1), (std::vector<Entity>{a}));
    EXPECT_EQ(owned(2), (std::vector<Entity>{b, child}));
    // And a destroy (the slot is reused)
    w.destroy(b);
    const Entity reused = w.create();
    EXPECT_EQ(w.ownerOf(reused), 0);
    EXPECT_TRUE(owned(2).empty());
}

TEST(RolesDeathTest, OwnershipAsserts)
{
#ifdef SAIGA_ASSERTS
    // setOwner after the spawn was sent; Player.peer must match; reparenting across owners; Parent on an owned root
    auto afterSpawn = []()
    {
        Net n;
        const Entity e = n.host->create();
        n.step();
        n.host->setOwner(e, 1);
    };
    EXPECT_DEATH(afterSpawn(), "spawn was sent");
    auto playerPeer = []()
    {
        World w(baseSetup());
        const Entity p = makePlayer(w, 0);
        w.setOwner(p, 1);
    };
    EXPECT_DEATH(playerPeer(), "Player.peer");
    auto acrossOwners = []()
    {
        Net n(2);
        const Entity a = makePlayer(*n.host, 1);
        const Entity b = makePlayer(*n.host, 2);
        const Entity c = n.host->create();
        n.host->add<Parent>(c, Parent{a});
        n.step();
        n.host->remove<Parent>(c);
        n.host->add<Parent>(c, Parent{b});
        n.step();
    };
    EXPECT_DEATH(acrossOwners(), "change the owner of a spawned entity");
    auto removeParentOfOwnedChild = []()
    {
        Net n;
        const Entity a = makePlayer(*n.host, 1);
        const Entity c = n.host->create();
        n.host->add<Parent>(c, Parent{a});
        n.step();
        n.host->remove<Parent>(c);
    };
    EXPECT_DEATH(removeParentOfOwnedChild(), "change the owner of a spawned entity");
    auto addNetOwnerAfterSpawn = []()
    {
        Net n;
        const Entity e = n.host->create();
        n.step();
        n.host->add<NetOwner>(e, NetOwner{1});
    };
    EXPECT_DEATH(addNetOwnerAfterSpawn(), "change the owner of a spawned entity");
    auto removeNetOwnerAfterSpawn = []()
    {
        Net n;
        const Entity e = makePlayer(*n.host, 1);
        n.step();
        n.host->remove<NetOwner>(e);
    };
    EXPECT_DEATH(removeNetOwnerAfterSpawn(), "change the owner of a spawned entity");
    auto ownedChild = []()
    {
        World w(baseSetup());
        const Entity a = makePlayer(w, 1);
        const Entity b = w.create();
        w.add<Parent>(a, Parent{b});
    };
    EXPECT_DEATH(ownedChild(), "has a NetOwner");
#endif
}

TEST(RolesDeathTest, ClientRestrictions)
{
#ifdef SAIGA_ASSERTS
    auto withClient = [](const std::function<void(Net&, Entity)>& fn)
    {
        Net n;
        const Entity e = makePlayer(*n.host, 0);
        n.step(2);
        fn(n, n.onClient(e));
    };
    EXPECT_DEATH(withClient([](Net& n, Entity e) { n.client().destroy(e); }), "cannot destroy a replicated");
    // Regression (local review, S16): a replicated entity under a local one was destroyed by the cascade
    auto cascade = [](Net& n, Entity e)
    {
        const Entity local = n.client().create();
        n.client().add<Parent>(e, Parent{local});
        n.client().destroy(local);
    };
    EXPECT_DEATH(withClient(cascade), "cannot destroy a replicated entity \\(a descendant\\)");
    EXPECT_DEATH(withClient([](Net& n, Entity e) { n.client().setOwner(e, 1); }), "setOwner on a client");
    EXPECT_DEATH(withClient([](Net& n, Entity) { n.client().save(); }), "save on a client");
    EXPECT_DEATH(withClient([](Net& n, Entity) { n.client().tryLoad(n.host->save()); }), "load on a client");
    EXPECT_DEATH(withClient([](Net& n, Entity e) { n.client().queueAction(PlayerAction{e, 1, Vec3::Zero()}); }),
                 "only for the players it owns");
    EXPECT_DEATH(withClient([](Net& n, Entity e) { n.client().makePersistent(e); }), "makePersistent on a client");
    EXPECT_DEATH(withClient([](Net& n, Entity) { n.client().spawn<Probe>(Vec3::Zero()); }), "spawn<T> on a client");
#endif
}

TEST(Roles, ClientRestrictionsInTheWorldApiAndCanSave)
{
    Net n;
    const Entity hostOwned = n.host->create();
    n.host->add<Health>(hostOwned);
    const Entity player = makePlayer(*n.host, 1);
    n.step(2);
    EXPECT_FALSE(n.client().canSave());
    EXPECT_TRUE(n.host->canSave());
    WorldApi capi(n.client());
    const Json ce = WorldApi::entityJson(n.onClient(hostOwned));
    ApiResult r   = capi.call("set_component",
                              Json::object{{"entity", ce}, {"component", "Health"}, {"values", Json::object{{"hp", 1}}}});
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("replicated"), std::string::npos) << r.error;
    EXPECT_FALSE(capi.call("destroy", Json::object{{"entity", ce}}).ok);
    EXPECT_FALSE(capi.call("save_snapshot", Json::object{{"name", "x"}}).ok);
    // The owner may write its owned owner-authoritative components
    const Json cp = WorldApi::entityJson(n.onClient(player));
    EXPECT_TRUE(capi.call("set_component", Json::object{{"entity", cp},
                                                        {"component", "PlayerInput"},
                                                        {"values", Json::object{{"heldButtons", 3}}}})
                    .ok);
    // The host refuses edits of a client-owned entity's owner-authoritative components after the spawn
    WorldApi hapi(*n.host);
    const Json hp = WorldApi::entityJson(player);
    r             = hapi.call(
        "set_component",
        Json::object{{"entity", hp}, {"component", "PlayerInput"}, {"values", Json::object{{"heldButtons", 3}}}});
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("peer 1"), std::string::npos) << r.error;
    EXPECT_FALSE(hapi.call("set_transform", Json::object{{"entity", hp}, {"position", Json::array{1, 2, 3}}}).ok);
    // Host components of it stay editable on the host
    EXPECT_TRUE(
        hapi.call("set_component",
                  Json::object{{"entity", hp}, {"component", "Player"}, {"values", Json::object{{"moveSpeed", 3}}}})
            .ok);
}

// ---------------------------------------------------------------------------------------------------------------------
// Replication over a pair

TEST(Replication, SpawnWithClass)
{
    g_hooks.clear();
    Net n;
    const Entity e = n.host->spawn<Probe>(Vec3(1, 2, 3), Health{50, 100});
    n.step();
    const Entity c = n.onClient(e);
    ASSERT_TRUE(c.valid());
    EXPECT_TRUE(n.client().has<Behaviour>(c));
    EXPECT_EQ(g_hooks[false].configure, 1);
    EXPECT_EQ(g_hooks[false].replicatedSpawn, 1);
    EXPECT_EQ(g_hooks[false].spawn, 0);
    EXPECT_EQ(n.client().read<Health>(c).hp, 50);
    EXPECT_TRUE(n.client().read<Transform>(c).position().isApprox(Vec3(1, 2, 3), 1e-4));
    EXPECT_TRUE(n.client().has<RenderPose>(c));
    EXPECT_EQ(n.host->replication().netIdOf(e), kFirstDynamicNetId);
}

// Regression guard: a parent with a higher entity index than its child is spawned first
TEST(Replication, ParentWithAHigherIndexAndReparenting)
{
    Net n;
    const Entity child = n.host->create(), p1 = n.host->create(), p2 = n.host->create();
    ASSERT_LT(child.index, p1.index);
    n.host->get<Transform>(p1).setPosition(Vec3(10, 0, 0));
    n.host->get<Transform>(p2).setPosition(Vec3(-10, 0, 0));
    n.host->add<LocalTransform>(child, LocalTransform{Vec3(1, 0, 0), Quat::Identity(), Vec3::Ones()});
    n.host->add<Parent>(child, Parent{p1});
    n.step(2);
    const Entity c = n.onClient(child);
    ASSERT_TRUE(c.valid());
    ASSERT_TRUE(n.client().has<Parent>(c));
    EXPECT_EQ(n.client().read<Parent>(c).entity, n.onClient(p1));
    EXPECT_TRUE(n.client().read<Transform>(c).position().isApprox(Vec3(11, 0, 0), 1e-4));
    // Reparenting: the Hierarchy message with the new LocalTransform
    n.host->remove<Parent>(child);
    n.host->get<LocalTransform>(child).position = Vec3(0, 2, 0);
    n.host->add<Parent>(child, Parent{p2});
    n.step(2);
    EXPECT_EQ(n.client().read<Parent>(c).entity, n.onClient(p2));
    EXPECT_TRUE(n.client().read<Transform>(c).position().isApprox(Vec3(-10, 2, 0), 1e-4));
    EXPECT_GT(n.host->replication().stats().hierarchySent, 0u);
}

TEST(Replication, DestroyAndCascade)
{
    g_hooks.clear();
    Net n;
    const Entity parent = n.host->spawn<Probe>(Vec3::Zero());
    const Entity child  = n.host->spawn<Probe>(Vec3::Zero());
    n.host->add<Parent>(child, Parent{parent});
    n.step(2);
    const Entity cp = n.onClient(parent), cc = n.onClient(child);
    ASSERT_TRUE(cp.valid() && cc.valid());
    n.host->destroy(parent);
    n.step();
    EXPECT_FALSE(n.client().alive(cp));
    EXPECT_FALSE(n.client().alive(cc));
    EXPECT_EQ(g_hooks[false].replicatedDestroy, 2);
    EXPECT_EQ(g_hooks[false].destroy, 0);
    EXPECT_EQ(g_hooks[true].destroy, 2);
    // The child's Destroy arrives after the cascade removed it: ignored, not rejected
    EXPECT_EQ(n.client().netStats().rejected, 0u);
    EXPECT_TRUE(n.client().replication().netIds().size() <= 2);  // the state entities
}

TEST(Replication, SpawnedAndDestroyedInOneStepSendsNothing)
{
    Net n;
    n.step();
    const uint64_t spawns = n.host->replication().stats().spawnsSent;
    const Entity e        = n.host->create();
    n.host->destroy(e);
    n.step();
    EXPECT_EQ(n.host->replication().stats().spawnsSent, spawns);
    EXPECT_EQ(n.host->replication().stats().destroysSent, 0u);
}

TEST(Replication, WorldAndSpaceStateAreBound)
{
    Net n;
    n.host->global<Score>().points      = 7;
    n.host->spaceState<Weather>(0).rain = 0.5;
    const Entity clientWorldState       = n.client().worldStateEntity();
    const Entity clientSpaceState       = n.client().spaceStateEntity(0);
    n.step(2);
    EXPECT_EQ(n.client().readGlobal<Score>().points, 7);
    EXPECT_EQ(n.client().readSpaceState<Weather>(0).rain, 0.5);
    // The client's own state entities, bound by the fixed NetIds
    EXPECT_EQ(n.client().worldStateEntity(), clientWorldState);
    EXPECT_EQ(n.client().replication().netIdOf(clientWorldState), kWorldStateNetId);
    EXPECT_EQ(n.client().replication().netIdOf(clientSpaceState), kFirstSpaceStateNetId);
    n.host->global<Score>().points = 9;
    n.step();
    EXPECT_EQ(n.client().readGlobal<Score>().points, 9);
}

TEST(Replication, HostReliableOnlyWhenTheBytesChange)
{
    Net n;
    const Entity e = n.host->create();
    n.host->add<Inventory>(e, Inventory{{1, 2, 3}, Entity{}});
    n.step(2);
    const uint64_t changes = n.host->replication().stats().componentChangesSent;
    n.step(5);
    EXPECT_EQ(n.host->replication().stats().componentChangesSent, changes);
    n.host->get<Inventory>(e).items.push_back(4);
    n.step();
    EXPECT_EQ(n.host->replication().stats().componentChangesSent, changes + 1);
    EXPECT_EQ(n.client().read<Inventory>(n.onClient(e)).items, (std::vector<uint32_t>{1, 2, 3, 4}));
    // Not in snapshots (an idle step sends nothing for it)
    n.step(3);
    EXPECT_EQ(n.host->replication().stats().componentChangesSent, changes + 1);
}

// An Entity field whose target arrives later is decoded again (spawns and HostReliable components)
TEST(Replication, UnresolvedEntityFieldsAreResolvedOnArrival)
{
    Net n;
    const Entity holder = n.host->create();
    n.host->add<Inventory>(holder, Inventory{{1}, Entity{}});
    n.step(2);
    // The target is created after the holder's reference: in the same step the holder's change refers to it
    const Entity target                  = n.host->create();
    n.host->get<Inventory>(holder).owner = target;
    n.step(2);
    EXPECT_EQ(n.client().read<Inventory>(n.onClient(holder)).owner, n.onClient(target));
    // A component added and a Never component's presence (ComponentChange)
    n.host->add<Target>(holder, Target{target, 2.0});
    n.host->add<CharacterIntent>(holder);
    n.step();
    EXPECT_EQ(n.client().read<Target>(n.onClient(holder)).entity, n.onClient(target));
    EXPECT_TRUE(n.client().has<CharacterIntent>(n.onClient(holder)));
    n.host->remove<Target>(holder);
    n.host->remove<CharacterIntent>(holder);
    n.step();
    EXPECT_FALSE(n.client().has<Target>(n.onClient(holder)));
    EXPECT_FALSE(n.client().has<CharacterIntent>(n.onClient(holder)));
}

TEST(Replication, SeveralHostStepsInOneBatch)
{
    Net n;
    const Entity e = n.host->create();
    n.host->add<Health>(e);
    n.step();
    for (int i = 0; i < 3; ++i)
    {
        n.host->get<Transform>(e).setPosition(Vec3(i, 0, 0));
        n.host->get<Health>(e).hp = 10 + i;
        n.host->step();
    }
    n.client().step();
    const Entity c = n.onClient(e);
    // The buffer got every snapshot, the components the newest
    ASSERT_NE(n.client().replication().buffer(c), nullptr);
    EXPECT_GE(n.client().replication().buffer(c)->size(), 3u);
    EXPECT_EQ(n.client().read<Health>(c).hp, 12);
    EXPECT_NEAR(n.client().read<Transform>(c).position().x(), 2, 1e-4);
}

TEST(Replication, HostTeleportSnapsTheRenderPose)
{
    Net n;
    const Entity e = n.host->create();
    for (int i = 0; i < 20; ++i)
    {
        n.host->get<Transform>(e).setPosition(Vec3(0.1 * i, 0, 0));
        n.step();
    }
    const Entity c = n.onClient(e);
    n.host->get<Transform>(e).teleport(Vec3(100, 0, 0), Quat::Identity());
    bool snapped = false, interpolatedAcross = false;
    for (int i = 0; i < 20; ++i)
    {
        n.step();
        const RenderPose& p = n.client().read<RenderPose>(c);
        snapped             = snapped || (p.flags & RenderPose::kTeleported);
        // Never between the old and the new pose
        interpolatedAcross = interpolatedAcross || (p.position.x() > 3 && p.position.x() < 99);
    }
    EXPECT_TRUE(snapped);
    EXPECT_FALSE(interpolatedAcross);
    EXPECT_NEAR(n.client().read<RenderPose>(c).position.x(), 100, 1e-4);
    EXPECT_TRUE(n.client().read<Transform>(c).wasTeleportedLastStep() ||
                n.client().read<Transform>(c).position().x() > 99);
}

TEST(Replication, OwnerStateOnTheHost)
{
    Net n;
    const Entity p = makePlayer(*n.host, 1);
    n.step(2);
    const Entity c = n.onClient(p);
    ASSERT_TRUE(c.valid());
    // The owner walks (CharacterIntent from the presence list, player_control over ownedBy)
    ASSERT_TRUE(n.client().has<CharacterIntent>(c));
    n.client().get<PlayerInput>(c).move = Vec3(1, 0, 0);
    n.step(60);
    const double clientX = n.client().read<Transform>(c).position().x();
    const double hostX   = n.host->read<Transform>(p).position().x();
    EXPECT_GT(clientX, 4.0);
    EXPECT_NEAR(hostX, clientX, 0.5);
    EXPECT_EQ(n.host->read<PlayerInput>(p).move, Vec3(1, 0, 0));
    // The host renders it from its buffer
    EXPECT_TRUE(n.host->has<RenderPose>(p));
    // The owner stops sending (it does not step): the host extrapolates at most 6 steps
    for (int i = 0; i < 20; ++i) n.host->step();
    const double speed = n.host->read<Player>(p).moveSpeed;
    EXPECT_LE(n.host->read<Transform>(p).position().x(), clientX + 6 * speed / 60 + 1e-3);
    EXPECT_GT(n.host->read<Transform>(p).position().x(), clientX + 3 * speed / 60);
}

TEST(Replication, OwnerTeleportAndTheCounter)
{
    Net n(1, 2);
    const Entity p = makePlayer(*n.host, 1);
    // Before the spawn: a plain teleport (no command, no counter)
    n.host->ownerTeleport(p, Vec3(5, 0, 0), Quat::Identity());
    EXPECT_EQ(n.host->replication().commandCounter(p), 0);
    n.step(8);
    const Entity c = n.onClient(p);
    ASSERT_TRUE(c.valid());
    EXPECT_NEAR(n.client().read<Transform>(c).position().x(), 5, 1e-4);
    // The owner moves; states in flight carry the old counter when the host teleports
    n.client().get<PlayerInput>(c).move = Vec3(0, 0, 1);
    n.step(10);
    const uint8_t ownerSnap = n.client().replication().snapCounter(c);
    const uint8_t hostSnap  = n.host->replication().snapCounter(p);
    n.host->ownerTeleport(p, Vec3(-50, 0, 0), Quat::Identity());
    EXPECT_EQ(n.host->replication().commandCounter(p), 1);
    EXPECT_EQ(n.host->read<Transform>(p).position().x(), -50);
    n.step(1);
    // Older states arriving now are dropped: the host keeps the teleported pose
    EXPECT_NEAR(n.host->read<Transform>(p).position().x(), -50, 1e-9);
    n.step(10);
    EXPECT_GT(n.host->replication().stats().ownerStatesDropped, 0u);
    EXPECT_EQ(n.client().replication().commandCounter(c), 1);
    EXPECT_NEAR(n.client().read<Transform>(c).position().x(), -50, 1e-9);
    EXPECT_NEAR(n.host->read<Transform>(p).position().x(), -50, 1e-3);
    // The owner ignores the snapshot counter (its teleports come as commands)
    EXPECT_EQ(n.client().replication().snapCounter(c), ownerSnap);
    EXPECT_EQ(n.host->replication().snapCounter(p), uint8_t(hostSnap + 1));
    // The counter wraps 255 -> 0 and the owner's states are still accepted
    for (int i = 0; i < 255; ++i) n.host->ownerTeleport(p, Vec3(-60, 0, 0), Quat::Identity());
    EXPECT_EQ(n.host->replication().commandCounter(p), 0);
    n.step(10);
    EXPECT_EQ(n.client().replication().commandCounter(c), 0);
    const double z = n.host->read<Transform>(p).position().z();
    n.step(10);
    EXPECT_GT(n.host->read<Transform>(p).position().z(), z);  // the walking owner's states arrive again
}

TEST(Replication, OwnerTeleportSnapsAThirdClient)
{
    Net n(2);
    const Entity p = makePlayer(*n.host, 1);
    n.step(10);
    const Entity observer = n.onClient(p, 2);
    ASSERT_TRUE(observer.valid());
    EXPECT_FALSE(n.client(2).read<RenderPose>(observer).flags & RenderPose::kTeleported);
    n.host->ownerTeleport(p, Vec3(80, 0, 0), Quat::Identity());
    bool snapped = false;
    for (int i = 0; i < 15; ++i)
    {
        n.step();
        snapped = snapped || (n.client(2).read<RenderPose>(observer).flags & RenderPose::kTeleported);
    }
    EXPECT_TRUE(snapped);
    EXPECT_NEAR(n.client(2).read<Transform>(observer).position().x(), 80, 1e-3);
}

TEST(Replication, PlayerActionsExactlyOnceInOrder)
{
    Net n(2);
    std::vector<uint32_t> seen;
    const Entity p1 = makePlayer(*n.host, 1), p2 = makePlayer(*n.host, 2);
    n.host->addStepListener(
        [&]()
        {
            for (const ActionEvent& a : n.host->events.read<ActionEvent>()) seen.push_back(a.action);
            return true;
        });
    n.step(2);
    const Entity c1 = n.onClient(p1);
    n.client(1).queueAction(PlayerAction{c1, 1, Vec3::Zero()});
    n.client(1).queueAction(PlayerAction{c1, 2, Vec3::Zero()});
    n.client(1).queueAction(PlayerAction{c1, 3, Vec3::Zero()});
    // A client sends an action of a player it does not own: the host drops and counts it
    n.client(2).send(0, PlayerActionMessage{n.host->replication().netIdOf(p1), 99, Vec3::Zero()});
    n.host->step();  // received in phase 1, queued
    EXPECT_TRUE(seen.empty());
    n.host->step();  // delivered as ActionEvents in the next step
    EXPECT_EQ(seen, (std::vector<uint32_t>{1, 2, 3}));
    n.step(5);
    EXPECT_EQ(seen, (std::vector<uint32_t>{1, 2, 3}));
    EXPECT_EQ(n.host->replication().stats().actionsDropped, 1u);
    (void)p2;
}

TEST(Replication, PingPongAndTheHostClock)
{
    Net n(1, 2);
    makePlayer(*n.host, 0);
    n.step(100);
    const HostClock& clock = n.client().replication().clock();
    EXPECT_TRUE(clock.valid());
    EXPECT_GT(clock.roundTrip(), 2);
    // The estimate is near the host step (the client steps right after the host here)
    EXPECT_NEAR(clock.estimate(n.client().stepIndex()), double(n.host->stepIndex()), 4);
    EXPECT_NEAR(n.client().time(), clock.estimate(n.client().stepIndex()) * n.client().dt(), 1e-9);
}

TEST(HostClockTest, SlewJumpAndConvergence)
{
    HostClock c;
    c.observeSnapshot(100, 10);
    EXPECT_EQ(c.offset(), 90);   // the first estimate jumps
    c.observeSnapshot(110, 15);  // target 95: slews
    for (int i = 0; i < 8; ++i) c.update();
    EXPECT_NEAR(c.offset(), 91, 1e-12);  // 1/8 per step
    c.observeSnapshot(200, 16);          // error 93 > 30: a resync jump
    EXPECT_EQ(c.offset(), 184);
    // The asynchronous pattern of the scenario variant (0, 1 or 2 local steps per host step, the cumulative
    // difference bounded): the estimate stays within a few steps of the host step
    HostClock a;
    Rng rng(5);
    int64_t local = 0;
    double worst  = 0;
    for (int64_t host = 1; host <= 2000; ++host)
    {
        std::vector<int> allowed;
        for (int k = 0; k <= 2; ++k)
            if (std::abs(local + k - (host - 3)) <= 2) allowed.push_back(k);
        const int k = allowed[size_t(rng.range(0, int64_t(allowed.size()) - 1))];
        for (int i = 0; i < k; ++i)
        {
            local++;
            a.observeSnapshot(uint64_t(host), uint64_t(local));
            a.update();
            if (host > 100) worst = std::max(worst, std::abs(a.estimate(uint64_t(local)) - double(host)));
        }
    }
    // The client's steps per host step vary by up to 2 around the mean (the slew follows the mean)
    EXPECT_LT(worst, 4.5);
}

TEST(InterpolationBufferTest, SamplingDryHoldAndSnap)
{
    InterpolationBuffer b;
    Vec3 p;
    Quat q;
    bool snapped;
    EXPECT_FALSE(b.sample(0, p, q, snapped));
    b.push(10, Vec3(0, 0, 0), Quat::Identity(), 0);
    b.push(12, Vec3(2, 0, 0), Quat::Identity(), 0);
    ASSERT_TRUE(b.sample(11, p, q, snapped));
    EXPECT_NEAR(p.x(), 1, 1e-12);
    EXPECT_FALSE(snapped);
    // Dry: the last pose is held
    b.sample(30, p, q, snapped);
    EXPECT_NEAR(p.x(), 2, 1e-12);
    // A teleport counter change: no interpolation over it, a snap when reached
    b.push(14, Vec3(100, 0, 0), Quat::Identity(), 1);
    b.sample(13, p, q, snapped);
    EXPECT_NEAR(p.x(), 2, 1e-12);
    EXPECT_FALSE(snapped);
    b.sample(14, p, q, snapped);
    EXPECT_NEAR(p.x(), 100, 1e-12);
    EXPECT_TRUE(snapped);
}

TEST(Replication, ResyncAfterAHostLoad)
{
    Net n(1, 1);
    n.host->global<Score>().points = 3;
    const Entity a                 = n.host->create();
    n.host->add<Health>(a, Health{11, 100});
    n.step(5);
    const std::vector<uint8_t> saved = n.host->save();
    const Entity b                   = n.host->create();
    n.host->add<Health>(b, Health{22, 100});
    n.step(5);
    const Entity worldState = n.client().worldStateEntity();
    EXPECT_EQ(n.client().replication().netIds().size(), 4u);  // world state, space state, a, b
    n.host->load(saved);
    EXPECT_EQ(n.host->replication().epoch(), 1u);
    n.step(6);
    EXPECT_EQ(n.client().replication().epoch(), 1u);
    // b is gone, a came again with a new NetId; the bound state entity is the same
    EXPECT_EQ(n.client().replication().netIds().size(), 3u);
    EXPECT_EQ(n.client().worldStateEntity(), worldState);
    EXPECT_EQ(n.client().readGlobal<Score>().points, 3);
    const Entity ca = n.client().replication().entityOfNetId(n.host->replication().netIds().back());
    ASSERT_TRUE(ca.valid());
    EXPECT_EQ(n.client().read<Health>(ca).hp, 11);
    EXPECT_GT(n.host->replication().netIds().back(), kFirstDynamicNetId + 1);  // never reused
    // A snapshot of the old epoch (in flight) is dropped
    std::vector<uint8_t> old{0, 0, 1, 100, 0, 1};  // flags, epoch 0, host step 1, sequence 100, part 0 of 1
    n.host->sendRaw(1, MessageId(EngineMessage::Snapshot), old.data(), old.size());
    n.step(2);
    EXPECT_EQ(n.client().replication().stats().snapshotsDroppedEpoch, 1u);
    EXPECT_EQ(n.client().netStats().rejected, 0u);
}

// An aborted phase 8 after the replication: the bookkeeping is rolled back, the spawn is sent in the next step
TEST(Replication, AbortedPhase8RollsBackTheBookkeeping)
{
    Net n;
    n.step();
    const Entity e                = n.host->create();
    bool fail                     = true;
    n.host->debugAfterReplication = [&]()
    {
        if (fail) throw std::runtime_error("after replication");
    };
    EXPECT_THROW(n.host->step(), std::runtime_error);
    EXPECT_FALSE(n.host->replication().spawned(e));
    EXPECT_EQ(n.host->replication().netIdOf(e), 0u);
    fail = false;
    n.step();
    EXPECT_TRUE(n.host->replication().spawned(e));
    EXPECT_EQ(n.host->replication().netIdOf(e), kFirstDynamicNetId);  // the NetId counter was rolled back
    EXPECT_TRUE(n.onClient(e).valid());
    EXPECT_EQ(n.client().netStats().rejected, 0u);
}

TEST(Replication, LocalEntities)
{
    Net n;
    // A client's own entity stays local
    const Entity mine = n.client().create();
    // A NetLocal host entity and its descendants are not replicated
    const Entity helper = n.host->create();
    n.host->add<NetLocal>(helper);
    const Entity below = n.host->create();
    n.host->add<Parent>(below, Parent{helper});
    const Entity normal = n.host->create();
    n.step(3);
    EXPECT_EQ(n.client().replication().netIdOf(mine), 0u);
    EXPECT_TRUE(n.client().alive(mine));
    EXPECT_FALSE(n.host->replication().spawned(helper));
    EXPECT_FALSE(n.host->replication().spawned(below));
    EXPECT_TRUE(n.onClient(normal).valid());
}

TEST(Replication, AHostWithoutPeersSerializesNothing)
{
    World w(baseSetup());
    const Entity e = w.create();
    w.step(5);
    EXPECT_TRUE(w.replication().spawned(e));  // spawned (a NetId) also without peers
    EXPECT_EQ(w.replication().stats().snapshotsSent, 0u);
    EXPECT_EQ(w.replication().stats().spawnsSent, 0u);
}

TEST(Replication, ALateJoinerGetsTheFullSpawnSet)
{
    Net n(2, 0, false);
    n.addClient(1);
    const Entity early = n.host->spawn<Probe>(Vec3(1, 0, 0));
    n.step(10);
    // Client 2's world starts later: the spawn set the host sent when the peer joined waits for it
    n.addClient(2);
    const Entity later = n.host->create();
    n.step(2);
    EXPECT_TRUE(n.onClient(early, 2).valid());
    EXPECT_TRUE(n.onClient(later, 2).valid());
    EXPECT_EQ(n.client(2).replication().netIds(), n.host->replication().netIds());
}

TEST(Replication, AThrowingOnReplicatedSpawnKeepsTheEntityMapped)
{
    Net n;
    const Entity e = n.host->spawn<Thrower>(Vec3::Zero());
    n.host->step();
    EXPECT_THROW(n.client().step(), std::runtime_error);
    const Entity c = n.onClient(e);
    ASSERT_TRUE(c.valid());
    EXPECT_FALSE(n.client().has<Behaviour>(c));
    n.step(2);  // stays in sync
    EXPECT_EQ(n.onClient(e), c);
}

TEST(Replication, AnimatedLocalTransformReachesClients)
{
    Net n;
    const Entity parent = n.host->create(), child = n.host->create();
    n.host->add<LocalTransform>(child);
    n.host->add<Parent>(child, Parent{parent});
    n.step();
    for (int i = 0; i < 10; ++i)
    {
        n.host->get<LocalTransform>(child).position = Vec3(0, 0.5 * i, 0);
        n.step();
        const Entity c = n.onClient(child);
        ASSERT_TRUE(c.valid());
        EXPECT_EQ(n.client().read<LocalTransform>(c).position, Vec3(0, 0.5 * i, 0));
    }
}

// Regression (local review, S16): the bookkeeping of a finished phase survives an abort later in the step and a later
// abort of the same phase in the next step (it was undone by the second abort while its OwnerCommand had been sent)
TEST(Replication, AnAbortKeepsTheCommandsOfFinishedPhases)
{
    Net n;
    const Entity p = makePlayer(*n.host, 1);
    n.step(3);
    g_teleportInStep = p;
    g_throwIn        = Phase::Transform;
    EXPECT_THROW(n.host->step(), std::runtime_error);  // phase 4 teleported (counter 1), phase 6 threw
    g_teleportInStep = Entity{};
    g_throwIn        = Phase::Gameplay;
    EXPECT_THROW(n.host->step(), std::runtime_error);
    g_throwIn = Phase::None;
    EXPECT_EQ(n.host->replication().commandCounter(p), 1);
    n.step(3);
    EXPECT_EQ(n.client().replication().commandCounter(n.onClient(p)), 1);
    EXPECT_NEAR(n.client().read<Transform>(n.onClient(p)).position().x(), 7, 1e-9);
}

// Regression (local review, S16): a throwing onReplicatedDestroy left the entity alive and mapped on the client
TEST(Replication, AThrowingOnReplicatedDestroyStillDestroysTheReplica)
{
    Net n;
    const Entity e = n.host->spawn<Probe>(Vec3::Zero());
    n.step(2);
    const Entity c = n.onClient(e);
    ASSERT_TRUE(c.valid());
    n.host->destroy(e);
    n.host->step();
    Probe::throwOnReplicatedDestroy = true;
    EXPECT_THROW(n.client().step(), std::runtime_error);
    Probe::throwOnReplicatedDestroy = false;
    EXPECT_FALSE(n.client().alive(c));
    n.step(3);
    EXPECT_EQ(n.client().replication().netIds(), n.host->replication().netIds());
}

// Regression (local review, S16): an aborted ownerTeleport left its sample (with the new counter) in the host's
// interpolation buffer and erased the owner's last state
TEST(Replication, AnAbortedOwnerTeleportLeavesNoSample)
{
    Net n;
    const Entity p = makePlayer(*n.host, 1);
    n.step(5);
    g_teleportInStep = p;
    g_throwIn        = Phase::Gameplay;  // after test_teleport in the same phase
    EXPECT_THROW(n.host->step(), std::runtime_error);
    g_teleportInStep = Entity{};
    g_throwIn        = Phase::None;
    EXPECT_EQ(n.host->replication().commandCounter(p), 0);
    InterpolationBuffer* b = n.host->replication().buffer(p);
    ASSERT_NE(b, nullptr);
    Vec3 position;
    Quat rotation;
    bool snapped = false;
    ASSERT_TRUE(b->sample(1e9, position, rotation, snapped));
    EXPECT_LT(position.norm(), 1.0);
    EXPECT_FALSE(snapped);
}

// Regression (local review, S16): an owner state dropped as stale (before the last OwnerCommand) counted as received
TEST(Replication, AStaleOwnerStateDoesNotCountAsReceived)
{
    Net n(1, 2);
    const Entity p = makePlayer(*n.host, 1);
    n.step(10);
    const uint64_t before  = n.host->replication().ownerStateSequence(1);
    const uint64_t dropped = n.host->replication().stats().ownerStatesDropped;
    n.host->ownerTeleport(p, Vec3(3, 0, 0), Quat::Identity());
    n.host->step();
    ASSERT_GT(n.host->replication().stats().ownerStatesDropped, dropped);
    EXPECT_EQ(n.host->replication().ownerStateSequence(1), before);
    n.step(10);
    EXPECT_GT(n.host->replication().ownerStateSequence(1), before);
}

// Regression (local review, S16): the receivers of ComponentChange and Hierarchy did not check for foreign writes
TEST(Replication, TheWriteDetectorSeesAClientWriteOfAHostReliableComponent)
{
    Net n;
    const Entity e = n.host->create();
    n.host->add<Inventory>(e);
    n.step(2);
    std::vector<std::string> reports;
    n.client().replication().setWriteDetector([&](const std::string& r) { reports.push_back(r); });
    n.step();
    const Entity c = n.onClient(e);
    ASSERT_TRUE(c.valid());
    n.host->get<Inventory>(e).items = {1};
    n.step();
    EXPECT_TRUE(reports.empty());
    n.client().get<Inventory>(c).items = {2};  // a foreign write
    n.host->get<Inventory>(e).items    = {3};
    n.step();
    ASSERT_FALSE(reports.empty());
    EXPECT_NE(reports[0].find("component change receiver"), std::string::npos) << reports[0];
}
