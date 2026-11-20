/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The extract and the render sink (docs/engine_design.md §10.1, §10.3; stage S19b), without a device
// (RecordingRenderSink): the name tables and their forwarding, the render instances from the structural log, the pose
// pairs, the catch-up between steps, aborted steps, remote entities (RenderPose over loopback), saves by mesh name,
// interpolatedPose.

#include "saiga/engine/api/WorldApi.h"
#include "saiga/engine/components/CharacterIntent.h"
#include "saiga/engine/components/LocalTransform.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/PlayerInput.h"
#include "saiga/engine/components/RenderInstance.h"
#include "saiga/engine/components/RenderPose.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/net/LoopbackTransport.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/sync/ExtractSystem.h"
#include "saiga/engine/sync/RecordingRenderSink.h"
#include "saiga/engine/world/World.h"

#include "gtest/gtest.h"

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
    Mover = kFirstGameComponent,
    LateMover,
    Count
};

// Moved in phase 4 (Gameplay) by velocity per second, turned by spin (rad/s) about y
struct Mover
{
    static constexpr const char* name = "Mover";
    static constexpr ComponentId id   = ComponentId(TestComponent::Mover);
    static constexpr uint32_t version = 1;
    Vec3 velocity                     = Vec3::Zero();
    double spin                       = 0;
    template <class V>
    void visit(V& v)
    {
        v("velocity", velocity);
        v("spin", spin);
    }
};
// Moved in phase 7 (Streaming) on every third step: idle in phase 6 of that step
struct LateMover
{
    static constexpr const char* name = "LateMover";
    static constexpr ComponentId id   = ComponentId(TestComponent::LateMover);
    static constexpr uint32_t version = 1;
    double dx                         = 0;
    template <class V>
    void visit(V& v)
    {
        v("dx", dx);
    }
};

void moverSystem(World& w, double dt)
{
    w.view<Transform, const Mover>().each(
        [&](Entity, Transform& t, const Mover& m)
        {
            t.setPosition(t.position() + m.velocity * dt);
            if (m.spin != 0) t.setRotation((t.rotation() * Quat(AngleAxis(m.spin * dt, Vec3(0, 1, 0)))).normalized());
        });
}

void lateMoverSystem(World& w, double)
{
    if (w.stepIndex() % 3 != 0) return;
    w.viewAll<Transform, const LateMover>().each([&](Entity, Transform& t, const LateMover& m)
                                                 { t.setPosition(t.position() + Vec3(m.dx, 0, 0)); });
}

std::shared_ptr<const MeshLods> triangle(float size)
{
    auto lods = std::make_shared<MeshLods>();
    Vulkan::MeshLod lod;
    lod.vertices.resize(3);
    lod.vertices[1].position = vec3(size, 0, 0);
    lod.vertices[2].position = vec3(0, size, 0);
    lod.indices              = {0, 1, 2};
    lods->lods.push_back(lod);
    lods->sphereRadius = size;
    return lods;
}

// The meshes as a game registers them (renderAssets): built once per process
std::shared_ptr<const MeshLods> boxLods()
{
    static auto l = triangle(1);
    return l;
}
std::shared_ptr<const MeshLods> ballLods()
{
    static auto l = triangle(2);
    return l;
}

void assetsBoxBall(World& w)
{
    const MaterialId m = w.registerMaterial("test/flat", MaterialData());
    w.registerMesh("test/box", boxLods(), m);
    w.registerMesh("test/ball", ballLods(), m);
}
// The same names in another order (the hash and the saves do not depend on it)
void assetsBallBox(World& w)
{
    const MaterialId m = w.registerMaterial("test/flat", MaterialData());
    w.registerMesh("test/ball", ballLods(), m);
    w.registerMesh("test/box", boxLods(), m);
}
// Without test/ball (a load of an entity that names it is rejected)
void assetsBoxOnly(World& w)
{
    const MaterialId m = w.registerMaterial("test/flat", MaterialData());
    w.registerMesh("test/box", boxLods(), m);
}
// test/box renamed to test/crate
void assetsRenamed(World& w)
{
    const MaterialId m = w.registerMaterial("test/flat", MaterialData());
    w.registerMesh("test/crate", boxLods(), m);
    w.registerMesh("test/ball", ballLods(), m);
    w.registerMeshAlias("test/box", "test/crate");
}

WorldSetup baseSetup(void (*assets)(World&) = assetsBoxBall)
{
    WorldSetup s;
    s.profiling  = false;
    s.components = [](Registry& r)
    {
        r.registerComponent<Mover>(Replicate::Host);
        r.registerComponent<LateMover>(Replicate::Host);
    };
    s.systems = [](SystemList& l)
    {
        l.add(Phase::Gameplay, "test_mover", moverSystem, RunOn::Host);
        l.add(Phase::Streaming, "test_late_mover", lateMoverSystem, RunOn::Host);
    };
    s.renderAssets = assets;
    return s;
}

Entity rendered(World& w, const std::string& mesh, const Vec3& p = Vec3::Zero())
{
    const Entity e = w.create();
    w.get<Transform>(e).setPosition(p);
    w.add<RenderInstance>(e, RenderInstance(mesh));
    return e;
}

// The world pose of a dynamic from the sink (its chunk's origin plus the chunk-relative pose)
WorldPose worldOf(const ChunkId& chunk, const DynamicPose& p)
{
    WorldPose w;
    w.space    = chunk.space;
    w.position = chunkOrigin(chunk) + p.position;
    w.rotation = p.rotation;
    w.scale    = p.scale;
    return w;
}

const RecordingRenderSink::Dynamic* dynamicOf(const World& w, const RecordingRenderSink& s, Entity e)
{
    return s.dynamic(w.extract().instanceOf(e));
}

// A sink that throws on demand (an aborted extract)
struct ThrowingSink : RecordingRenderSink
{
    int throwOnAdd = 0;  // the n-th next addDynamic throws (0: never)
    int throwOnSet = 0;  // the n-th next setDynamic throws
    InstanceHandle addDynamic(MeshId mesh) override
    {
        if (throwOnAdd > 0 && --throwOnAdd == 0) throw std::runtime_error("sink: addDynamic");
        return RecordingRenderSink::addDynamic(mesh);
    }
    void setDynamic(InstanceHandle h, const ChunkId& chunk, const DynamicPose& prev, const DynamicPose& curr,
                    uint32_t boneOffset) override
    {
        if (throwOnSet > 0 && --throwOnSet == 0) throw std::runtime_error("sink: setDynamic");
        RecordingRenderSink::setDynamic(h, chunk, prev, curr, boneOffset);
    }
};

// A host and clients over loopback (S16), each with its own recording sink
struct Net
{
    std::vector<std::unique_ptr<LoopbackTransport>> t;
    std::unique_ptr<World> host;
    std::vector<std::unique_ptr<World>> clients;
    RecordingRenderSink hostSink;
    std::vector<std::unique_ptr<RecordingRenderSink>> clientSinks;
    explicit Net(int delay = 0)
    {
        LoopbackOptions o;
        o.delayRounds = delay;
        t             = LoopbackTransport::createNetwork(2, o);
        WorldSetup s  = baseSetup();
        s.transport   = t[0].get();
        host          = std::make_unique<World>(s);
        host->setRenderSink(&hostSink);
        WorldSetup c = baseSetup();
        c.transport  = t[1].get();
        c.role       = NetRole::Client;
        c.localPeer  = 1;
        clients.push_back(std::make_unique<World>(c));
        clientSinks.push_back(std::make_unique<RecordingRenderSink>());
        clients[0]->setRenderSink(clientSinks[0].get());
    }
    ~Net()
    {
        for (auto& c : clients) c->setRenderSink(nullptr);
        host->setRenderSink(nullptr);
    }
    World& client() { return *clients[0]; }
    RecordingRenderSink& clientSink() { return *clientSinks[0]; }
    void step(int n = 1)
    {
        for (int k = 0; k < n; ++k)
        {
            host->step();
            for (auto& c : clients) c->step();
        }
    }
    Entity onClient(Entity h) { return client().replication().entityOfNetId(host->replication().netIdOf(h)); }
};
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// Name tables and the sink

TEST(ExtractSink, RegistrationsForwardedWhenTheSinkIsSetLate)
{
    World w(baseSetup());
    // The World assigns the IDs from kGameReservedIds upward (the game range below stays for the scene's own content)
    EXPECT_EQ(w.findMesh("test/box"), Vulkan::kGameReservedIds);
    EXPECT_EQ(w.findMesh("test/ball"), Vulkan::kGameReservedIds + 1);
    EXPECT_EQ(w.findMaterial("test/flat"), Vulkan::kGameReservedIds);
    EXPECT_EQ(w.findMesh("nope"), 0);
    EXPECT_EQ(w.meshName(Vulkan::kGameReservedIds + 1), "test/ball");
    const Entity e = rendered(w, "test/box");
    w.step();
    RecordingRenderSink s;
    w.setRenderSink(&s);
    // Everything registered before, then the existing entity's instance
    EXPECT_EQ(s.materials().size(), 1u);
    ASSERT_EQ(s.meshes().size(), 2u);
    EXPECT_EQ(s.meshes().at(Vulkan::kGameReservedIds), "test/box");
    EXPECT_EQ(s.meshMaterial(Vulkan::kGameReservedIds), Vulkan::kGameReservedIds);
    ASSERT_EQ(s.liveDynamics(), 1u);
    EXPECT_EQ(dynamicOf(w, s, e)->mesh, Vulkan::kGameReservedIds);
    // A registration after the sink is set is forwarded at once
    const MeshId later = w.registerMesh("test/later", boxLods(), w.findMaterial("test/flat"));
    EXPECT_EQ(s.meshes().at(later), "test/later");
    // The same sink again: identical registrations are no-ops, the instances are recreated
    const uint64_t registered = s.counters().registerMesh;
    w.setRenderSink(&s);
    EXPECT_EQ(s.counters().registerMesh, registered);
    EXPECT_EQ(s.liveDynamics(), 1u);
    w.setRenderSink(nullptr);
}

TEST(ExtractSink, ReplacementAndNullFreeTheHandlesAndTheConsumer)
{
    World w(baseSetup());
    rendered(w, "test/box");
    rendered(w, "test/ball", Vec3(3, 0, 0));
    RecordingRenderSink a, b;
    w.setRenderSink(&a);
    EXPECT_EQ(a.liveDynamics(), 2u);
    w.setRenderSink(&b);
    EXPECT_EQ(a.liveDynamics(), 0u);
    EXPECT_EQ(b.liveDynamics(), 2u);
    w.setRenderSink(nullptr);
    EXPECT_EQ(b.liveDynamics(), 0u);
    EXPECT_EQ(w.extract().trackedCount(), 0u);
    EXPECT_EQ(w.renderSink(), nullptr);
    // No consumer is left behind: the log is truncated after every step
    for (int i = 0; i < 5; ++i)
    {
        rendered(w, "test/box");
        w.step();
    }
    EXPECT_EQ(w.structuralLog().size(), 0u);
}

TEST(ExtractSink, HostAndClientTablesAreEqual)
{
    Net n;
    EXPECT_EQ(n.host->renderTableHash(), n.client().renderTableHash());
    World other(baseSetup(assetsBallBox));
    EXPECT_NE(other.renderTableHash(), n.host->renderTableHash());
}

// ---------------------------------------------------------------------------------------------------------------------
// Instances from the structural log

TEST(ExtractInstances, CreatedAndFreedFromTheStructuralLog)
{
    World w(baseSetup());
    RecordingRenderSink s;
    w.setRenderSink(&s);
    // A spawn between steps: created by the next extract with a snap at the current pose
    const Entity e = rendered(w, "test/box", Vec3(1, 2, 3));
    EXPECT_EQ(s.liveDynamics(), 0u);
    w.step();
    ASSERT_EQ(s.liveDynamics(), 1u);
    const RecordingRenderSink::Dynamic* d = dynamicOf(w, s, e);
    ASSERT_NE(d, nullptr);
    EXPECT_TRUE(d->set);
    EXPECT_EQ(d->prev.position, d->curr.position);
    EXPECT_EQ(worldOf(d->chunk, d->curr).position, Vec3(1, 2, 3));
    // remove / add
    w.remove<RenderInstance>(e);
    w.step();
    EXPECT_EQ(s.liveDynamics(), 0u);
    w.add<RenderInstance>(e, RenderInstance("test/box"));
    w.step();
    EXPECT_EQ(s.liveDynamics(), 1u);
    // setRenderMesh: the instance is recreated with the other mesh
    w.setRenderMesh(e, "test/ball");
    EXPECT_EQ(w.read<RenderInstance>(e).mesh, "test/ball");
    w.step();
    ASSERT_EQ(s.liveDynamics(), 1u);
    EXPECT_EQ(dynamicOf(w, s, e)->mesh, w.findMesh("test/ball"));
    // The world API's set_component of the name (an unknown name is an error, nothing changes)
    WorldApi api(w);
    const Json je = WorldApi::entityJson(e);
    ApiResult r   = api.call(
        "set_component",
        Json::object{{"entity", je}, {"component", "RenderInstance"}, {"values", Json::object{{"mesh", "nope"}}}});
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("not registered"), std::string::npos) << r.error;
    EXPECT_EQ(w.read<RenderInstance>(e).mesh, "test/ball");
    r = api.call(
        "set_component",
        Json::object{{"entity", je}, {"component", "RenderInstance"}, {"values", Json::object{{"mesh", "test/box"}}}});
    EXPECT_TRUE(r.ok) << r.error;
    w.step();
    EXPECT_EQ(dynamicOf(w, s, e)->mesh, w.findMesh("test/box"));
    // add_component and spawn with an unknown name are errors
    const Entity bare = w.create();
    EXPECT_FALSE(api.call("add_component", Json::object{{"entity", WorldApi::entityJson(bare)},
                                                        {"component", "RenderInstance"},
                                                        {"values", Json::object{{"mesh", "nope"}}}})
                     .ok);
    EXPECT_FALSE(w.has<RenderInstance>(bare));
    EXPECT_FALSE(api.call("spawn", Json::object{{"position", Json::array{0, 0, 0}},
                                                {"components",
                                                 Json::object{{"RenderInstance", Json::object{{"mesh", "nope"}}}}}})
                     .ok);
    // destroy
    w.destroy(e);
    w.step();
    EXPECT_EQ(s.liveDynamics(), 0u);
    EXPECT_EQ(w.extract().trackedCount(), 0u);
    w.setRenderSink(nullptr);
}

TEST(ExtractInstances, EmptyNameIsTrackedNotDrawn)
{
    World w(baseSetup());
    RecordingRenderSink s;
    w.setRenderSink(&s);
    const Entity e = w.create();
    w.add<RenderInstance>(e);
    w.step();
    EXPECT_EQ(w.extract().trackedCount(), 1u);
    EXPECT_EQ(w.extract().instanceCount(), 0u);
    EXPECT_EQ(s.liveDynamics(), 0u);
    WorldPose p;
    EXPECT_TRUE(w.extract().interpolatedPose(e, 0.5, p));
    // A name later: drawn
    w.setRenderMesh(e, "test/box");
    w.step();
    EXPECT_EQ(s.liveDynamics(), 1u);
    w.setRenderSink(nullptr);
}

// Refused in every build (a game's own save or a replicated entity stays usable): logged, stored empty
TEST(ExtractInstances, UnknownNameThroughAddIsStoredEmptyAndLogged)
{
    World w(baseSetup());
    const Entity e        = w.create();
    const uint64_t before = engineLog().lastSequence();
    w.add<RenderInstance>(e, RenderInstance("nope"));
    EXPECT_EQ(w.read<RenderInstance>(e).mesh, "");
    bool logged = false;
    for (const auto& l : engineLog().since(before))
        logged = logged || (l.message.find("nope") != std::string::npos && l.category == "render");
    EXPECT_TRUE(logged);
}

TEST(ExtractInstances, MeshAlias)
{
    World w(baseSetup(assetsRenamed));
    EXPECT_EQ(w.findMesh("test/box"), w.findMesh("test/crate"));
    const Entity e = rendered(w, "test/box");
    // The alias resolves to the new name (saves keep the new name from now on)
    EXPECT_EQ(w.read<RenderInstance>(e).mesh, "test/crate");
}

// ---------------------------------------------------------------------------------------------------------------------
// Pose pairs

TEST(ExtractPairs, MovingStopTeleportAndChunkBorder)
{
    World w(baseSetup());
    RecordingRenderSink s;
    w.setRenderSink(&s);
    const Entity e = rendered(w, "test/box", Vec3(250, 0, 0));
    w.add<Mover>(e, Mover{Vec3(60, 0, 0), 0});  // 1 m per step
    w.step();
    // The pair: prev = the old curr, curr = the new pose (the creation snapped at 250, the step moved it to 251)
    const RecordingRenderSink::Dynamic* d = dynamicOf(w, s, e);
    ASSERT_NE(d, nullptr);
    w.step();
    EXPECT_NEAR(worldOf(d->chunk, d->prev).position.x(), 251, 1e-9);
    EXPECT_NEAR(worldOf(d->chunk, d->curr).position.x(), 252, 1e-9);
    // Across the chunk border (256 m): both poses relative to the chunk of the current pose
    for (int i = 0; i < 4; ++i) w.step();
    EXPECT_EQ(d->chunk.x, 1);
    EXPECT_NEAR(d->prev.position.x(), 255 - 256, 1e-9);
    EXPECT_NEAR(d->curr.position.x(), 256 - 256, 1e-9);
    // Stopped: one more pair (curr, curr), then nothing
    w.remove<Mover>(e);
    w.step();
    EXPECT_EQ(d->prev.position, d->curr.position);
    const uint64_t updates = d->updates;
    w.step(3);
    EXPECT_EQ(d->updates, updates);
    // A teleport: a snap at the new pose
    w.get<Transform>(e).teleport(Vec3(-1000, 5, 0), Quat::Identity());
    w.step();
    EXPECT_EQ(d->updates, updates + 1);
    EXPECT_EQ(d->prev.position, d->curr.position);
    EXPECT_NEAR(worldOf(d->chunk, d->curr).position.x(), -1000, 1e-9);
    EXPECT_EQ(d->chunk.x, -4);
    bool teleported = false;
    WorldPose p;
    ASSERT_TRUE(w.extract().interpolatedPose(e, 0.5, p, &teleported));
    EXPECT_TRUE(teleported);
    w.setRenderSink(nullptr);
}

TEST(ExtractPairs, InterpolatedPoseMatchesTheGpuFormula)
{
    World w(baseSetup());
    RecordingRenderSink s;
    w.setRenderSink(&s);
    const Entity e = rendered(w, "test/box", Vec3(100, 1, -40));
    w.add<Mover>(e, Mover{Vec3(30, 6, -12), 2.5});
    w.step(3);
    const RecordingRenderSink::Dynamic* d = dynamicOf(w, s, e);
    ASSERT_NE(d, nullptr);
    EXPECT_NE(d->prev.position, d->curr.position);
    for (double alpha : {0.0, 0.37, 0.999})
    {
        // S18's CPU reference of the shaders on the sink's chunk-relative pair
        const DynamicPose ref = Vulkan::interpolatePose(d->prev, d->curr, alpha);
        WorldPose p;
        bool teleported = true;
        ASSERT_TRUE(w.extract().interpolatedPose(e, alpha, p, &teleported));
        EXPECT_FALSE(teleported);
        EXPECT_LT((p.position - (chunkOrigin(d->chunk) + ref.position)).norm(), 1e-6);
        EXPECT_LT(p.rotation.angularDistance(ref.rotation), 1e-6);
    }
    // Untracked: dead, without RenderInstance, created since the last extract
    const Entity plain = w.create();
    WorldPose p;
    p.position = Vec3(7, 7, 7);
    EXPECT_FALSE(w.extract().interpolatedPose(plain, 0.5, p));
    const Entity fresh = rendered(w, "test/box");
    EXPECT_FALSE(w.extract().interpolatedPose(fresh, 0.5, p));
    w.destroy(e);
    EXPECT_FALSE(w.extract().interpolatedPose(e, 0.5, p));
    EXPECT_EQ(p.position, Vec3(7, 7, 7));
    w.setRenderSink(nullptr);
}

// A step aborted by the sink: the unread structural record is delivered again, the pairs are not shifted twice
TEST(ExtractPairs, AbortedStep)
{
    World w(baseSetup());
    ThrowingSink s;
    w.setRenderSink(&s);
    const Entity a = rendered(w, "test/box");
    const Entity b = rendered(w, "test/box", Vec3(0, 0, 10));
    w.add<Mover>(a, Mover{Vec3(60, 0, 0), 0});
    w.add<Mover>(b, Mover{Vec3(60, 0, 0), 0});
    w.step(2);
    const RecordingRenderSink::Dynamic* da = dynamicOf(w, s, a);
    ASSERT_NE(da, nullptr);
    const double committed = worldOf(da->chunk, da->curr).position.x();
    // The second setDynamic of the next step throws: a's pair was sent, b's not
    s.throwOnSet       = 2;
    const uint64_t idx = w.stepIndex();
    EXPECT_THROW(w.step(), std::runtime_error);
    EXPECT_EQ(w.stepIndex(), idx);
    // The rerun (the movers moved in the aborted step's phase 4 and move again): prev is the committed curr
    w.step();
    EXPECT_NEAR(worldOf(da->chunk, da->prev).position.x(), committed, 1e-9);
    EXPECT_NEAR(worldOf(da->chunk, da->curr).position.x(), w.read<Transform>(a).position().x(), 1e-9);
    // A throwing addDynamic: the record is delivered again by the next extract
    const Entity c = rendered(w, "test/ball");
    s.throwOnAdd   = 1;
    EXPECT_THROW(w.step(), std::runtime_error);
    EXPECT_EQ(s.liveDynamics(), 2u);
    w.step();
    EXPECT_EQ(s.liveDynamics(), 3u);
    EXPECT_NE(dynamicOf(w, s, c), nullptr);
    w.setRenderSink(nullptr);
}

// The extract never changes the world: a world with a sink hashes like one without
TEST(ExtractPairs, RecordingSinkDoesNotChangeTheHash)
{
    World a(baseSetup()), b(baseSetup());
    RecordingRenderSink s;
    a.setRenderSink(&s);
    for (World* w : {&a, &b})
    {
        const Entity e = rendered(*w, "test/box");
        w->add<Mover>(e, Mover{Vec3(1, 0, 2), 0.5});
        rendered(*w, "");
        w->step(20);
    }
    EXPECT_EQ(a.hash(), b.hash());
    a.setRenderSink(nullptr);
}

// ---------------------------------------------------------------------------------------------------------------------
// Catch-up between steps

TEST(ExtractCatchUp, PausedEditsReachTheSinkOnce)
{
    World w(baseSetup());
    RecordingRenderSink s;
    w.setRenderSink(&s);
    const Entity e = rendered(w, "test/box");
    w.step();
    w.setPaused(true);
    // Frames without changes: nothing is sent (O(tracked) comparisons only)
    uint64_t sets = s.counters().setDynamic;
    for (int i = 0; i < 5; ++i) w.extract().catchUp();
    EXPECT_EQ(s.counters().setDynamic, sets);
    // An edit through the world API: one snap over many frames
    WorldApi api(w);
    ASSERT_TRUE(
        api.call("set_transform", Json::object{{"entity", WorldApi::entityJson(e)}, {"position", Json::array{5, 0, 0}}})
            .ok);
    for (int i = 0; i < 5; ++i) w.extract().catchUp();
    EXPECT_EQ(s.counters().setDynamic, sets + 1);
    const RecordingRenderSink::Dynamic* d = dynamicOf(w, s, e);
    EXPECT_NEAR(worldOf(d->chunk, d->curr).position.x(), 5, 1e-12);
    EXPECT_EQ(d->prev.position, d->curr.position);
    // A spawn: one instance, one snap
    ASSERT_TRUE(api.call("spawn", Json::object{{"position", Json::array{1, 0, 0}},
                                               {"components",
                                                Json::object{{"RenderInstance", Json::object{{"mesh", "test/ball"}}}}}})
                    .ok);
    sets = s.counters().setDynamic;
    for (int i = 0; i < 5; ++i) w.extract().catchUp();
    EXPECT_EQ(s.liveDynamics(), 2u);
    EXPECT_EQ(s.counters().setDynamic, sets + 1);
    // load_snapshot recreates the instances: one live handle per rendered entity
    ASSERT_TRUE(api.call("save_snapshot", Json::object{{"name", "a"}}).ok);
    rendered(w, "test/box", Vec3(9, 0, 0));
    w.extract().catchUp();
    EXPECT_EQ(s.liveDynamics(), 3u);
    ASSERT_TRUE(api.call("load_snapshot", Json::object{{"name", "a"}}).ok);
    w.extract().catchUp();
    EXPECT_EQ(s.liveDynamics(), w.registry().poolById(RenderInstance::id)->size());
    EXPECT_EQ(s.liveDynamics(), 2u);
    w.setRenderSink(nullptr);
}

// The catch-up also runs at time scale 0 (no steps), and not in frames between running steps
TEST(ExtractCatchUp, TriggerAtTimeScaleZero)
{
    World w(baseSetup());
    RecordingRenderSink s;
    w.setRenderSink(&s);
    const Entity e = rendered(w, "test/box");
    w.step();
    // Running: no catch-up (the next step's extract sends the change)
    EXPECT_FALSE(w.extract().catchUp());
    w.get<Transform>(e).setPosition(Vec3(0, 3, 0));
    EXPECT_FALSE(w.extract().catchUp());
    w.setTimeScale(0);
    EXPECT_TRUE(w.extract().catchUp());
    EXPECT_NEAR(worldOf(dynamicOf(w, s, e)->chunk, dynamicOf(w, s, e)->curr).position.y(), 3, 1e-12);
    w.setRenderSink(nullptr);
}

// Phase 7-8 changes enter the next step's pair; a catch-up after a tool call does not send them as edits
TEST(ExtractCatchUp, LateMoversAreNotSentByACatchUp)
{
    World w(baseSetup());
    RecordingRenderSink s;
    w.setRenderSink(&s);
    const Entity e = rendered(w, "test/box");
    w.add<LateMover>(e, LateMover{1});
    w.step();
    // Until a step that moves it in phase 7 (stepIndex % 3 == 0 inside the step)
    while ((w.stepIndex() % 3) != 0) w.step();
    w.step();
    const double moved                    = w.read<Transform>(e).position().x();
    const RecordingRenderSink::Dynamic* d = dynamicOf(w, s, e);
    ASSERT_NE(d, nullptr);
    EXPECT_LT(worldOf(d->chunk, d->curr).position.x(), moved);  // one step late
    const uint64_t sets = s.counters().setDynamic;
    w.extract().markDirty();  // a tool call
    EXPECT_TRUE(w.extract().catchUp());
    EXPECT_EQ(s.counters().setDynamic, sets);
    // The next step's pair carries it
    w.step();
    EXPECT_NEAR(worldOf(d->chunk, d->curr).position.x(), moved, 1e-12);
    EXPECT_LT(worldOf(d->chunk, d->prev).position.x(), moved);
    w.setRenderSink(nullptr);
}

// ---------------------------------------------------------------------------------------------------------------------
// Remote entities (S16 over loopback)

TEST(ExtractRemote, ClientFrameWithoutAStepSendsNothing)
{
    Net n;
    const Entity e = rendered(*n.host, "test/box");
    n.host->add<Mover>(e, Mover{Vec3(6, 0, 0), 0});
    n.step(10);
    const Entity c = n.onClient(e);
    ASSERT_TRUE(c.valid());
    EXPECT_EQ(n.clientSink().liveDynamics(), 1u);
    EXPECT_EQ(n.client().read<RenderInstance>(c).mesh, "test/box");
    const uint64_t sets = n.clientSink().counters().setDynamic;
    for (int i = 0; i < 4; ++i) n.client().extract().catchUp();
    EXPECT_EQ(n.clientSink().counters().setDynamic, sets);
    // The client draws the remote entity from RenderPose (behind the host's Transform by the delay)
    n.step();
    const RecordingRenderSink::Dynamic* d = dynamicOf(n.client(), n.clientSink(), c);
    ASSERT_NE(d, nullptr);
    EXPECT_NEAR(worldOf(d->chunk, d->curr).position.x(), n.client().read<RenderPose>(c).position.x(), 1e-12);
    EXPECT_LT(worldOf(d->chunk, d->curr).position.x(), n.client().read<Transform>(c).position().x());
}

// A client's ComponentChange of the name recreates the instance there
TEST(ExtractRemote, ComponentChangeOfTheName)
{
    Net n;
    const Entity e = rendered(*n.host, "test/box");
    n.step(3);
    const Entity c = n.onClient(e);
    ASSERT_TRUE(c.valid());
    EXPECT_EQ(dynamicOf(n.client(), n.clientSink(), c)->mesh, n.client().findMesh("test/box"));
    n.host->setRenderMesh(e, "test/ball");
    n.step(3);
    EXPECT_EQ(n.client().read<RenderInstance>(c).mesh, "test/ball");
    ASSERT_EQ(n.clientSink().liveDynamics(), 1u);
    EXPECT_EQ(dynamicOf(n.client(), n.clientSink(), c)->mesh, n.client().findMesh("test/ball"));
}

// The host draws a client-owned root from RenderPose only (no pop back by the delay): not before its first owner
// state; its child (no RenderPose) composes the root's RenderPose with its LocalTransform
TEST(ExtractRemote, RemoteRootAfterItsFirstRenderPoseWithItsChild)
{
    Net n;
    const Entity p = n.host->create();
    Player pl;
    pl.peer = 1;
    n.host->add<Player>(p, pl);
    n.host->add<PlayerInput>(p);
    n.host->add<CharacterIntent>(p);
    n.host->add<RenderInstance>(p, RenderInstance("test/box"));
    n.host->setOwner(p, 1);
    const Entity child = n.host->create();
    LocalTransform local;
    local.position = Vec3(0, 2, 0);
    n.host->add<LocalTransform>(child, local);
    n.host->add<Parent>(child, Parent{p});
    n.host->add<RenderInstance>(child, RenderInstance("test/ball"));
    // Host steps only: the client sends no owner state yet
    n.host->step(3);
    EXPECT_EQ(n.host->extract().trackedCount(), 2u);
    EXPECT_EQ(n.hostSink.liveDynamics(), 0u);
    EXPECT_FALSE(n.host->has<RenderPose>(p));
    WorldPose wp;
    EXPECT_FALSE(n.host->extract().interpolatedPose(p, 0.5, wp));
    // The owner walks: its owner states give the host a RenderPose; root and child are drawn with it, with a snap
    n.step(2);
    const Entity c = n.onClient(p);
    ASSERT_TRUE(c.valid());
    n.client().get<PlayerInput>(c).move = Vec3(1, 0, 0);
    for (int i = 0; i < 40 && !n.host->has<RenderPose>(p); ++i) n.step();
    ASSERT_TRUE(n.host->has<RenderPose>(p));
    n.step();
    ASSERT_EQ(n.hostSink.liveDynamics(), 2u);
    n.step(20);
    const RenderPose& rp                     = n.host->read<RenderPose>(p);
    const RecordingRenderSink::Dynamic* root = dynamicOf(*n.host, n.hostSink, p);
    const RecordingRenderSink::Dynamic* kid  = dynamicOf(*n.host, n.hostSink, child);
    ASSERT_TRUE(root && kid);
    EXPECT_LT((worldOf(root->chunk, root->curr).position - rp.position).norm(), 1e-9);
    EXPECT_LT((worldOf(kid->chunk, kid->curr).position - (rp.position + rp.rotation * Vec3(0, 2, 0))).norm(), 1e-9);
    // Behind the host's own Transform (the owner's poses are interpolated with the delay)
    EXPECT_LT(rp.position.x(), n.host->read<Transform>(p).position().x());
}

// ---------------------------------------------------------------------------------------------------------------------
// Saves and the world API

TEST(ExtractSave, ByNameIndependentOfRegistrationOrder)
{
    World a(baseSetup());
    const Entity e = rendered(a, "test/ball", Vec3(1, 0, 0));
    rendered(a, "test/box");
    rendered(a, "");
    a.step();
    const std::vector<uint8_t> blob = a.save();
    EXPECT_NE(a.dumpJson(e).find("test/ball"), std::string::npos);
    // Another registration order: other IDs, the same names, the same hash
    World b(baseSetup(assetsBallBox));
    EXPECT_NE(a.findMesh("test/ball"), b.findMesh("test/ball"));
    ASSERT_TRUE(b.tryLoad(blob).ok());
    EXPECT_EQ(a.hash(), b.hash());
    EXPECT_EQ(b.read<RenderInstance>(e).mesh, "test/ball");
    EXPECT_EQ(b.read<RenderInstance>(e).cachedMesh, b.findMesh("test/ball"));
    // A missing mesh is a load error naming the entity (the world is unchanged)
    World c(baseSetup(assetsBoxOnly));
    const LoadResult r = c.tryLoad(blob);
    EXPECT_FALSE(r.committed);
    EXPECT_NE(r.error.find("entity index " + std::to_string(e.index)), std::string::npos) << r.error;
    EXPECT_NE(r.error.find("test/ball"), std::string::npos) << r.error;
    // A renamed mesh through an alias
    World d(baseSetup(assetsRenamed));
    ASSERT_TRUE(d.tryLoad(blob).ok());
    bool renamed = false;
    d.registry().view<const RenderInstance>().each([&](Entity, const RenderInstance& ri)
                                                   { renamed = renamed || ri.mesh == "test/crate"; });
    EXPECT_TRUE(renamed);
}

TEST(ExtractSave, NonUniformScaleOfARenderedEntityIsAnApiError)
{
    World w(baseSetup());
    WorldApi api(w);
    const Entity e    = rendered(w, "test/box");
    const Json je     = WorldApi::entityJson(e);
    const ApiResult r = api.call("set_transform", Json::object{{"entity", je}, {"scale", Json::array{1, 2, 1}}});
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("uniform"), std::string::npos) << r.error;
    EXPECT_TRUE(api.call("set_transform", Json::object{{"entity", je}, {"scale", Json::array{2, 2, 2}}}).ok);
    // Without RenderInstance it is allowed
    const Entity plain = w.create();
    EXPECT_TRUE(api.call("set_transform",
                         Json::object{{"entity", WorldApi::entityJson(plain)}, {"scale", Json::array{1, 2, 1}}})
                    .ok);
    // Game code: logged once per entity, rendered with scale.x
    RecordingRenderSink s;
    w.setRenderSink(&s);
    const uint64_t before = engineLog().lastSequence();
    w.get<Transform>(e).setScale(Vec3(3, 1, 1));
    w.step(3);
    int lines = 0;
    for (const auto& l : engineLog().since(before)) lines += l.message.find("non-uniform") != std::string::npos;
    EXPECT_EQ(lines, 1);
    EXPECT_EQ(dynamicOf(w, s, e)->curr.scale, 3);
    w.setRenderSink(nullptr);
}
