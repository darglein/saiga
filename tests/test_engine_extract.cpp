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
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/gameplay/ClassList.h"
#include "saiga/engine/lighting/EnvironmentExtract.h"
#include "saiga/engine/net/LoopbackTransport.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/sync/ExtractSystem.h"
#include "saiga/engine/sync/RecordingRenderSink.h"
#include "saiga/engine/world/World.h"

#include "gtest/gtest.h"

#include <limits>
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

enum class TestClass : uint16_t
{
    Crate = kFirstGameClass,
    Relabel,
    Stretched,
    Keep,
};
// Adds its own RenderInstance in onSpawn (an agent's spawn may name another mesh as an initial component)
class Crate final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(TestClass::Crate);
    static constexpr const char* name = "Crate";
    void onSpawn() override
    {
        if (!has<RenderInstance>()) add<RenderInstance>(RenderInstance("test/box"));
    }
};

// Changes its saved mesh in onLoad (refused: a load hook must not change saved state)
class Relabel final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(TestClass::Relabel);
    static constexpr const char* name = "Relabel";
    void onSpawn() override
    {
        if (!has<RenderInstance>()) add<RenderInstance>(RenderInstance("test/box"));
    }
    void onLoad() override { world().setRenderMesh(entity(), "test/ball"); }
};

// Sets a non-uniform scale in onSpawn (no RenderInstance of its own)
class Stretched final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(TestClass::Stretched);
    static constexpr const char* name = "Stretched";
    void onSpawn() override { transformWrite().setScale(Vec3(1, 2, 1)); }
};
// Sets its saved mesh to the same name in onLoad (no change: allowed)
class Keep final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(TestClass::Keep);
    static constexpr const char* name = "Keep";
    void onSpawn() override
    {
        if (!has<RenderInstance>()) add<RenderInstance>(RenderInstance("test/box"));
    }
    void onLoad() override { world().setRenderMesh(entity(), "test/box"); }
};

void aliasInStepSystem(World& w, double)
{
    w.registerMeshAlias("old/in_step", "test/box");
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
    s.classes = [](ClassList& c)
    {
        c.add<Crate>();
        c.add<Relabel>();
        c.add<Stretched>();
        c.add<Keep>();
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

// Space 0 and the enclosed space 1 (S21)
std::vector<SpaceTemplate> twoSpaces()
{
    return {SpaceTemplate(), SpaceTemplate::enclosed("cellar")};
}

// A sink that throws on demand (an aborted extract)
struct ThrowingSink : RecordingRenderSink
{
    int throwOnAdd    = 0;  // the n-th next addDynamic throws (0: never)
    int throwOnSet    = 0;  // the n-th next setDynamic throws
    int throwOnRemove = 0;  // the n-th next removeDynamic throws (before removing)
    void removeDynamic(InstanceHandle h) override
    {
        if (throwOnRemove > 0 && --throwOnRemove == 0) throw std::runtime_error("sink: removeDynamic");
        RecordingRenderSink::removeDynamic(h);
    }
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
    explicit Net(int delay = 0, bool spaces = false)
    {
        LoopbackOptions o;
        o.delayRounds = delay;
        t             = LoopbackTransport::createNetwork(2, o);
        WorldSetup s  = baseSetup();
        if (spaces) s.config.spaceTemplates = twoSpaces();
        s.transport = t[0].get();
        host        = std::make_unique<World>(s);
        host->setRenderSink(&hostSink);
        WorldSetup c = baseSetup();
        if (spaces) c.config.spaceTemplates = twoSpaces();
        c.transport = t[1].get();
        c.role      = NetRole::Client;
        c.localPeer = 1;
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

// S36: the recording sink keeps the last environment and counts the calls (the app's per-frame call; the World does
// not forward one)
TEST(RecordingRenderSink, EnvironmentAndLightsRecorded)
{
    WorldSetup s;
    s.environment.enabled = true;
    World w(s);
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    w.step(3);
    EXPECT_EQ(sink.environment(), nullptr);
    EXPECT_EQ(sink.counters().setEnvironment, 0u);
    Vulkan::EnvironmentSettings settings;
    settings.exposure.mode  = Vulkan::ExposureMode::Fixed;
    settings.exposure.ev100 = 7;
    sink.setEnvironment(environmentState(w, 0, w.alpha(), settings, w.environmentSetup()));
    sink.setEnvironment(environmentState(w, 0, w.alpha(), settings, w.environmentSetup()));
    ASSERT_NE(sink.environment(), nullptr);
    EXPECT_EQ(sink.counters().setEnvironment, 2u);
    EXPECT_EQ(sink.environment()->ev100, 7.0f);
    EXPECT_EQ(sink.environment()->exposureSource, Vulkan::ExposureSource::Fixed);
    w.setRenderSink(nullptr);
}

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

// An unknown name through add is logged and kept (tracked, not drawn); a later registration of the mesh draws it.
// Regression (S19b review): the name was stored empty, so a mesh registered after an entity named it was never drawn
// (ExtractSystem::meshRegistered could not find it)
TEST(ExtractInstances, UnknownNameIsKeptAndDrawnWhenRegisteredLater)
{
    World w(baseSetup(assetsBoxOnly));
    RecordingRenderSink s;
    w.setRenderSink(&s);
    const uint64_t before = engineLog().lastSequence();
    const Entity e        = rendered(w, "test/ball", Vec3(3, 0, 0));
    EXPECT_EQ(w.read<RenderInstance>(e).mesh, "test/ball");
    bool logged = false;
    for (const auto& l : engineLog().since(before))
        logged = logged || (l.message.find("test/ball") != std::string::npos && l.category == "render");
    EXPECT_TRUE(logged);
    w.step();
    EXPECT_EQ(w.extract().trackedCount(), 1u);
    EXPECT_EQ(s.liveDynamics(), 0u);
    // Registered later (between steps): the sink gets the mesh and the instance at the entity's pose
    const MeshId ball = w.registerMesh("test/ball", ballLods(), w.findMaterial("test/flat"));
    ASSERT_EQ(s.liveDynamics(), 1u);
    const RecordingRenderSink::Dynamic* d = dynamicOf(w, s, e);
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(d->mesh, ball);
    EXPECT_NEAR(worldOf(d->chunk, d->curr).position.x(), 3, 1e-12);
    w.setRenderSink(nullptr);
}

// Regression (S19b review): the spawn's merge of an initial RenderInstance into the one the class added in onSpawn
// bypassed renderMeshWritten: the component named test/ball while the cached ID (and the sink) kept test/box
TEST(ExtractInstances, SpawnMergeOfAnInitialRenderInstanceIntoTheClassOne)
{
    World w(baseSetup());
    RecordingRenderSink s;
    w.setRenderSink(&s);
    WorldApi api(w);
    const ApiResult r = api.call(
        "spawn", Json::object{{"class", "Crate"},
                              {"position", Json::array{1, 0, 0}},
                              {"components", Json::object{{"RenderInstance", Json::object{{"mesh", "test/ball"}}}}}});
    ASSERT_TRUE(r.ok) << r.error;
    Entity e;
    std::string error;
    ASSERT_TRUE(WorldApi::parseEntity(r.value["entity"], e, error)) << error;
    w.extract().catchUp();
    EXPECT_EQ(w.read<RenderInstance>(e).mesh, "test/ball");
    EXPECT_EQ(w.read<RenderInstance>(e).cachedMesh, w.findMesh("test/ball"));
    ASSERT_EQ(s.liveDynamics(), 1u);
    EXPECT_EQ(dynamicOf(w, s, e)->mesh, w.findMesh("test/ball"));
    w.step();
    EXPECT_EQ(dynamicOf(w, s, e)->mesh, w.findMesh("test/ball"));
    w.setRenderSink(nullptr);
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
// Regression (S19b review): an entity moved in phase 7 got the pair (old, new) in the next step and kept it (it was in
// phase 6's list of that step only), so it was drawn between both poses forever
TEST(ExtractPairs, LateMoverPairSettles)
{
    World w(baseSetup());
    RecordingRenderSink s;
    w.setRenderSink(&s);
    const Entity e = rendered(w, "test/box");
    w.add<LateMover>(e, LateMover{1});
    w.step();
    while ((w.stepIndex() % 3) != 0) w.step();
    w.step();  // moved in phase 7
    w.remove<LateMover>(e);
    const double moved                    = w.read<Transform>(e).position().x();
    const RecordingRenderSink::Dynamic* d = dynamicOf(w, s, e);
    ASSERT_NE(d, nullptr);
    w.step();
    EXPECT_NEAR(worldOf(d->chunk, d->curr).position.x(), moved, 1e-12);
    EXPECT_LT(worldOf(d->chunk, d->prev).position.x(), moved);
    // Two steps after the move: (curr, curr), and nothing after that
    w.step();
    EXPECT_EQ(d->prev.position, d->curr.position);
    EXPECT_NEAR(worldOf(d->chunk, d->curr).position.x(), moved, 1e-12);
    const uint64_t updates = d->updates;
    w.step(3);
    EXPECT_EQ(d->updates, updates);
    WorldPose p;
    ASSERT_TRUE(w.extract().interpolatedPose(e, 0.3, p));
    EXPECT_NEAR(p.position.x(), moved, 1e-12);
    w.setRenderSink(nullptr);
}

// A teleported spawn snaps once. Regression (S19b review): an entity created and teleported before the extract tracked
// it (in a step, or before a sink was set) snapped again in the next step (its teleport looked late or pending), and a
// teleport the catch-up sent followed by an idle step snapped again when the entity moved
TEST(ExtractPairs, TeleportSnapsOnce)
{
    World w(baseSetup());
    RecordingRenderSink s;
    w.setRenderSink(&s);
    w.step();
    // Created and teleported between the steps of a running world, tracked by the step's extract: one snap
    const Entity a = w.create();
    w.get<Transform>(a).teleport(Vec3(5, 0, 0), Quat::Identity());
    w.add<RenderInstance>(a, RenderInstance("test/box"));
    w.add<Mover>(a, Mover{Vec3(60, 0, 0), 0});
    w.step();
    const RecordingRenderSink::Dynamic* da = dynamicOf(w, s, a);
    ASSERT_NE(da, nullptr);
    EXPECT_EQ(da->prev.position, da->curr.position);
    w.step();
    EXPECT_NE(da->prev.position, da->curr.position);
    // Tracked by a catch-up (a tool's spawn): the catch-up's snap only
    const Entity b = w.create();
    w.get<Transform>(b).teleport(Vec3(20, 0, 0), Quat::Identity());
    w.add<RenderInstance>(b, RenderInstance("test/box"));
    w.add<Mover>(b, Mover{Vec3(60, 0, 0), 0});
    w.extract().markDirty();
    ASSERT_TRUE(w.extract().catchUp());
    w.step();
    const RecordingRenderSink::Dynamic* db = dynamicOf(w, s, b);
    ASSERT_NE(db, nullptr);
    EXPECT_NEAR(worldOf(db->chunk, db->prev).position.x(), 20, 1e-12);
    EXPECT_NEAR(worldOf(db->chunk, db->curr).position.x(), 21, 1e-12);
    // A teleport sent by the paused catch-up, an idle step, then motion: interpolated
    const Entity c = rendered(w, "test/box");
    w.step();
    w.setPaused(true);
    WorldApi api(w);
    ASSERT_TRUE(api.call("set_transform",
                         Json::object{{"entity", WorldApi::entityJson(c)}, {"position", Json::array{0, 0, 50}}})
                    .ok);
    w.extract().catchUp();
    w.setPaused(false);
    w.step();
    w.add<Mover>(c, Mover{Vec3(60, 0, 0), 0});
    w.step();
    const RecordingRenderSink::Dynamic* dc = dynamicOf(w, s, c);
    EXPECT_NEAR(worldOf(dc->chunk, dc->prev).position.x(), 0, 1e-12);
    EXPECT_NEAR(worldOf(dc->chunk, dc->curr).position.x(), 1, 1e-12);
    w.setRenderSink(nullptr);
    // A sink set between a teleport and the next step: its creation snap only
    World v(baseSetup());
    const Entity f = v.create();
    v.get<Transform>(f).teleport(Vec3(40, 0, 0), Quat::Identity());
    v.add<RenderInstance>(f, RenderInstance("test/box"));
    v.add<Mover>(f, Mover{Vec3(60, 0, 0), 0});
    RecordingRenderSink late;
    v.setRenderSink(&late);
    v.step();
    const RecordingRenderSink::Dynamic* df = dynamicOf(v, late, f);
    ASSERT_NE(df, nullptr);
    EXPECT_NEAR(worldOf(df->chunk, df->prev).position.x(), 40, 1e-12);
    EXPECT_NEAR(worldOf(df->chunk, df->curr).position.x(), 41, 1e-12);
    v.setRenderSink(nullptr);
}

// snapSequence changes once per snap and stays while the entity is idle (the games' camera reset reacts once).
// Regression (S19b review): the platformer reset its camera from the level of interpolatedPose's teleported, every
// frame while the hero stood still after its spawn, so Q/E did not turn the camera
TEST(ExtractPairs, SnapSequenceChangesOncePerTeleport)
{
    World w(baseSetup());
    RecordingRenderSink s;
    w.setRenderSink(&s);
    const Entity e = rendered(w, "test/box");
    EXPECT_EQ(w.extract().snapSequence(e), 0u);  // not tracked yet
    w.step();
    const uint64_t created = w.extract().snapSequence(e);
    EXPECT_NE(created, 0u);
    w.step(3);
    EXPECT_EQ(w.extract().snapSequence(e), created);
    // A teleport: a new value once, then the same while idle (the pair's teleported level stays set)
    w.get<Transform>(e).teleport(Vec3(30, 0, 0), Quat::Identity());
    w.step();
    const uint64_t teleported = w.extract().snapSequence(e);
    EXPECT_NE(teleported, created);
    w.step(3);
    EXPECT_EQ(w.extract().snapSequence(e), teleported);
    bool level = false;
    WorldPose p;
    ASSERT_TRUE(w.extract().interpolatedPose(e, 0.5, p, &level));
    EXPECT_TRUE(level);
    // Moving (no snap): unchanged
    w.add<Mover>(e, Mover{Vec3(60, 0, 0), 0});
    w.step(3);
    EXPECT_EQ(w.extract().snapSequence(e), teleported);
    w.setRenderSink(nullptr);
}

// Regression (S19b review): a setDynamic that threw on an instance's creation snap (after addDynamic succeeded) left
// the handle; the re-delivered record found it and sent nothing, so an idle entity's instance never got a pose
TEST(ExtractPairs, FailedFirstSetDynamicIsRetried)
{
    World w(baseSetup());
    ThrowingSink s;
    w.setRenderSink(&s);
    w.step();
    // In a step's extract
    const Entity a = rendered(w, "test/box", Vec3(4, 0, 0));
    s.throwOnSet   = 1;
    EXPECT_THROW(w.step(), std::runtime_error);
    w.step();
    const RecordingRenderSink::Dynamic* da = dynamicOf(w, s, a);
    ASSERT_NE(da, nullptr);
    EXPECT_TRUE(da->set);
    EXPECT_NEAR(worldOf(da->chunk, da->curr).position.x(), 4, 1e-12);
    EXPECT_EQ(s.liveDynamics(), 1u);
    // In a catch-up (paused; the first call after the step does not run)
    w.setPaused(true);
    w.extract().catchUp();
    const Entity b = rendered(w, "test/box", Vec3(6, 0, 0));
    s.throwOnSet   = 1;
    EXPECT_THROW(w.extract().catchUp(), std::runtime_error);
    w.extract().catchUp();
    const RecordingRenderSink::Dynamic* db = dynamicOf(w, s, b);
    ASSERT_NE(db, nullptr);
    EXPECT_TRUE(db->set);
    EXPECT_NEAR(worldOf(db->chunk, db->curr).position.x(), 6, 1e-12);
    EXPECT_EQ(s.liveDynamics(), 2u);
    w.setRenderSink(nullptr);
}

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

// Regression (S19b review): after a step with a phase 7 move the world was paused: the catch-up skipped the late mover
// (its baseline was set at the commit), so it was drawn at its old pose for the whole pause
TEST(ExtractCatchUp, PausedAfterALateMove)
{
    World w(baseSetup());
    RecordingRenderSink s;
    w.setRenderSink(&s);
    const Entity e = rendered(w, "test/box");
    w.add<LateMover>(e, LateMover{1});
    w.step();
    while ((w.stepIndex() % 3) != 0) w.step();
    w.step();  // moved in phase 7
    const double moved                    = w.read<Transform>(e).position().x();
    const RecordingRenderSink::Dynamic* d = dynamicOf(w, s, e);
    ASSERT_NE(d, nullptr);
    const double before = worldOf(d->chunk, d->curr).position.x();
    EXPECT_LT(before, moved);
    w.setPaused(true);
    for (int i = 0; i < 3; ++i) w.extract().catchUp();
    // The pair the next step would send (interpolated at the frozen alpha like every other mover)
    EXPECT_NEAR(worldOf(d->chunk, d->curr).position.x(), moved, 1e-12);
    EXPECT_NEAR(worldOf(d->chunk, d->prev).position.x(), before, 1e-12);
    w.setPaused(false);
    w.remove<LateMover>(e);
    w.step();
    EXPECT_EQ(d->prev.position, d->curr.position);
    EXPECT_NEAR(worldOf(d->chunk, d->curr).position.x(), moved, 1e-12);
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
    // A missing mesh: the load succeeds and keeps the name (ExtractSave.UnknownNameLoadsAndIsDrawnWhenRegistered)
    World c(baseSetup(assetsBoxOnly));
    ASSERT_TRUE(c.tryLoad(blob).ok());
    EXPECT_EQ(c.read<RenderInstance>(e).mesh, "test/ball");
    EXPECT_EQ(c.read<RenderInstance>(e).cachedMesh, 0);
    // A renamed mesh through an alias
    World d(baseSetup(assetsRenamed));
    ASSERT_TRUE(d.tryLoad(blob).ok());
    bool renamed = false;
    d.registry().view<const RenderInstance>().each([&](Entity, const RenderInstance& ri)
                                                   { renamed = renamed || ri.mesh == "test/crate"; });
    EXPECT_TRUE(renamed);
}

// A save of an entity whose mesh name is not registered (kept by the add path) loads into a world without that mesh
// (logged once), and a later registration draws it. Regression (S19b review): the load rejected the unknown name, so a
// game's own save of such an entity could not be loaded
TEST(ExtractSave, UnknownNameLoadsAndIsDrawnWhenRegistered)
{
    World a(baseSetup(assetsBoxOnly));
    const Entity e = rendered(a, "test/ball", Vec3(7, 0, 0));
    rendered(a, "test/ball", Vec3(8, 0, 0));
    rendered(a, "test/box");
    a.step();
    const std::vector<uint8_t> blob = a.save();
    World b(baseSetup(assetsBoxOnly));
    RecordingRenderSink s;
    b.setRenderSink(&s);
    const uint64_t before = engineLog().lastSequence();
    const LoadResult r    = b.tryLoad(blob);
    ASSERT_TRUE(r.ok()) << r.error;
    int lines = 0;
    for (const auto& l : engineLog().since(before))
        if (l.category == "render" && l.message.find("test/ball") != std::string::npos) lines++;
    EXPECT_EQ(lines, 1);
    EXPECT_EQ(b.read<RenderInstance>(e).mesh, "test/ball");
    EXPECT_EQ(b.hash(), a.hash());
    b.extract().catchUp();
    EXPECT_EQ(b.extract().trackedCount(), 3u);
    EXPECT_EQ(s.liveDynamics(), 1u);  // the box
    // Registered later: both balls are drawn at their poses
    const MeshId ball = b.registerMesh("test/ball", ballLods(), b.findMaterial("test/flat"));
    ASSERT_EQ(s.liveDynamics(), 3u);
    const RecordingRenderSink::Dynamic* d = dynamicOf(b, s, e);
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(d->mesh, ball);
    EXPECT_NEAR(worldOf(d->chunk, d->curr).position.x(), 7, 1e-12);
    b.setRenderSink(nullptr);
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

// ---------------------------------------------------------------------------------------------------------------------
// CodeRabbit review (S19b, PR #65)

// Regression: a chunk coordinate beyond the int32 range was converted with undefined behavior
TEST(ExtractReview, ChunkCoordinatesAreClamped)
{
    WorldPose p;
    p.position        = Vec3(1e12, 0, -1e12);
    const ChunkPose c = toChunkPose(p);
    EXPECT_EQ(c.chunk.x, std::numeric_limits<int32_t>::max());
    EXPECT_EQ(c.chunk.z, std::numeric_limits<int32_t>::min());
    p.position = Vec3(-300, 0, 600);
    EXPECT_EQ(toChunkPose(p).chunk.x, -2);
    EXPECT_EQ(toChunkPose(p).chunk.z, 2);
}

// Regression: a sink that threw for the first existing entity while it was attached left the others untracked (the
// consumer starts after their records)
TEST(ExtractReview, SinkAttachedDespiteAThrowingAdd)
{
    World w(baseSetup());
    for (int i = 0; i < 3; ++i) rendered(w, "test/box", Vec3(i, 0, 0));
    w.step();
    ThrowingSink s;
    s.throwOnAdd = 1;
    EXPECT_THROW(w.setRenderSink(&s), std::runtime_error);
    EXPECT_EQ(w.extract().trackedCount(), 3u);
    w.extract().catchUp();
    EXPECT_EQ(s.liveDynamics(), 3u);
    for (const auto& d : s.dynamics()) EXPECT_TRUE(d.second.set);
    w.setRenderSink(nullptr);
    EXPECT_EQ(s.liveDynamics(), 0u);
}

// Regression: a removeDynamic that threw after the entity was untracked lost the handle (the instance stayed drawn)
TEST(ExtractReview, FailedRemovalIsRetried)
{
    World w(baseSetup());
    ThrowingSink s;
    w.setRenderSink(&s);
    const Entity e = rendered(w, "test/box");
    w.step();
    ASSERT_EQ(s.liveDynamics(), 1u);
    w.destroy(e);
    s.throwOnRemove = 1;
    EXPECT_THROW(w.step(), std::runtime_error);
    w.step();
    EXPECT_EQ(s.liveDynamics(), 0u);
    w.setRenderSink(nullptr);
}

// Regression: a Parent record consumed before an addDynamic that threw lost its source re-evaluation, so the child of
// a RenderPose root stayed on its own Transform
TEST(ExtractReview, SourceReevaluationSurvivesAThrowingSink)
{
    World w(baseSetup());
    ThrowingSink s;
    w.setRenderSink(&s);
    const Entity root = w.create();
    RenderPose rp;
    rp.position = Vec3(5, 0, 0);
    w.add<RenderPose>(root, rp);
    const Entity child = rendered(w, "test/box");
    w.step();
    const RecordingRenderSink::Dynamic* before = dynamicOf(w, s, child);
    ASSERT_NE(before, nullptr);
    EXPECT_NEAR(worldOf(before->chunk, before->curr).position.x(), 0, 1e-12);
    // The Parent record first, then an add whose addDynamic throws
    w.add<Parent>(child, Parent{root});
    rendered(w, "test/ball", Vec3(9, 0, 0));
    s.throwOnAdd = 1;
    EXPECT_THROW(w.step(), std::runtime_error);
    w.step(2);
    // Composed from the root's RenderPose (x = 5), not the child's Transform (x = 0)
    const RecordingRenderSink::Dynamic* d = dynamicOf(w, s, child);
    ASSERT_NE(d, nullptr);
    EXPECT_NEAR(worldOf(d->chunk, d->curr).position.x(), 5, 1e-12);
    EXPECT_EQ(s.liveDynamics(), 2u);
    w.setRenderSink(nullptr);
}

// Regression: every paused catch-up that consumed a RenderPose, Parent or NetOwner record appended every tracked
// remote again (only a step deduplicated the list)
TEST(ExtractReview, RemoteListHasOneEntryPerEntity)
{
    World w(baseSetup());
    RecordingRenderSink s;
    w.setRenderSink(&s);
    for (int i = 0; i < 10; ++i)
    {
        const Entity e = rendered(w, "test/box", Vec3(i, 0, 0));
        w.add<RenderPose>(e);
    }
    w.step();
    EXPECT_EQ(w.extract().remoteQueued(), 10u);
    w.setPaused(true);
    w.extract().catchUp();
    for (int i = 0; i < 20; ++i)
    {
        const Entity a = w.create(), b = w.create();
        w.add<Parent>(b, Parent{a});
        w.extract().catchUp();
    }
    EXPECT_EQ(w.extract().remoteQueued(), 10u);
    w.setRenderSink(nullptr);
}

// Regression: an addDynamic that threw when a pending remote root got its first RenderPose left it without an instance
// (no retry: hasPose was set, the creation was not pending)
TEST(ExtractReview, FirstRemotePoseWithAThrowingAdd)
{
    // Before n: Net's teardown detaches the host's sink also when an assertion returns early
    ThrowingSink s;
    Net n;
    n.host->setRenderSink(&s);
    const Entity p = n.host->create();
    Player pl;
    pl.peer = 1;
    n.host->add<Player>(p, pl);
    n.host->add<PlayerInput>(p);
    n.host->add<CharacterIntent>(p);
    n.host->add<RenderInstance>(p, RenderInstance("test/box"));
    n.host->setOwner(p, 1);
    // The first addDynamic (at the root's first RenderPose, owner states without input: idle) throws
    s.throwOnAdd = 1;
    bool threw   = false;
    for (int i = 0; i < 40 && !threw; ++i)
    {
        try
        {
            n.step();
        }
        catch (const std::runtime_error&)
        {
            threw = true;
        }
    }
    ASSERT_TRUE(threw);
    EXPECT_TRUE(n.host->has<RenderPose>(p));
    EXPECT_EQ(s.liveDynamics(), 0u);
    n.step(5);
    EXPECT_EQ(s.liveDynamics(), 1u);
    const RecordingRenderSink::Dynamic* d = dynamicOf(*n.host, s, p);
    ASSERT_NE(d, nullptr);
    EXPECT_TRUE(d->set);
    n.host->setRenderSink(&n.hostSink);
}

// Regression: a second teleport after the catch-up had sent the first one (before the next step) was suppressed: the
// step interpolated from the first teleport's pose to the second's
TEST(ExtractReview, SecondTeleportBeforeTheStepSnaps)
{
    World w(baseSetup());
    RecordingRenderSink s;
    w.setRenderSink(&s);
    const Entity e = rendered(w, "test/box");
    w.step();
    w.get<Transform>(e).teleport(Vec3(10, 0, 0), Quat::Identity());
    w.extract().markDirty();
    ASSERT_TRUE(w.extract().catchUp());
    w.get<Transform>(e).teleport(Vec3(100, 0, 0), Quat::Identity());
    w.step();
    const RecordingRenderSink::Dynamic* d = dynamicOf(w, s, e);
    ASSERT_NE(d, nullptr);
    EXPECT_NEAR(worldOf(d->chunk, d->curr).position.x(), 100, 1e-12);
    EXPECT_NEAR(worldOf(d->chunk, d->prev).position.x(), 100, 1e-12);
    // One teleport the catch-up sent: still no second snap in the step (it moves from there)
    w.get<Transform>(e).teleport(Vec3(200, 0, 0), Quat::Identity());
    w.extract().markDirty();
    ASSERT_TRUE(w.extract().catchUp());
    const uint64_t seq = w.extract().snapSequence(e);
    w.add<Mover>(e, Mover{Vec3(60, 0, 0), 0});
    w.step();
    EXPECT_EQ(w.extract().snapSequence(e), seq);
    EXPECT_NEAR(worldOf(d->chunk, d->prev).position.x(), 200, 1e-12);
    EXPECT_NEAR(worldOf(d->chunk, d->curr).position.x(), 201, 1e-12);
    w.setRenderSink(nullptr);
}

// Regression: a paused edit whose setDynamic threw already had its pair and baseline committed, so the next catch-up
// skipped it and the sink kept the old pose
TEST(ExtractReview, PausedEditWithAThrowingSetIsRetried)
{
    World w(baseSetup());
    ThrowingSink s;
    w.setRenderSink(&s);
    const Entity e = rendered(w, "test/box");
    w.step();
    w.setPaused(true);
    w.extract().catchUp();
    w.get<Transform>(e).setPosition(Vec3(0, 7, 0));
    s.throwOnSet = 1;
    EXPECT_THROW(w.extract().catchUp(), std::runtime_error);
    w.extract().catchUp();
    const RecordingRenderSink::Dynamic* d = dynamicOf(w, s, e);
    ASSERT_NE(d, nullptr);
    EXPECT_NEAR(worldOf(d->chunk, d->curr).position.y(), 7, 1e-12);
    w.setRenderSink(nullptr);
}

// Regression: an alias registered after an entity named the old name (kept by a load or the add path) did not draw it
TEST(ExtractReview, AliasRegisteredLaterDrawsTheEntity)
{
    World w(baseSetup());
    RecordingRenderSink s;
    w.setRenderSink(&s);
    const Entity e = rendered(w, "old/box", Vec3(2, 0, 0));
    w.step();
    EXPECT_EQ(s.liveDynamics(), 0u);
    w.registerMeshAlias("old/box", "test/box");
    ASSERT_EQ(s.liveDynamics(), 1u);
    EXPECT_EQ(dynamicOf(w, s, e)->mesh, w.findMesh("test/box"));
    w.setRenderSink(nullptr);
}

// Regression: onLoad could change an existing saved RenderInstance through setRenderMesh (the add branch was refused)
TEST(ExtractReview, SetRenderMeshInOnLoadIsRefused)
{
    World a(baseSetup());
    WorldApi api(a);
    ASSERT_TRUE(api.call("spawn", Json::object{{"class", "Relabel"}, {"position", Json::array{0, 0, 0}}}).ok);
    a.step();
    const std::vector<uint8_t> blob = a.save();
    World b(baseSetup());
    const LoadResult r = b.tryLoad(blob);
    ASSERT_TRUE(r.committed) << r.error;
    ASSERT_EQ(r.hookErrors.size(), 1u);
    EXPECT_NE(r.hookErrors[0].message.find("setRenderMesh"), std::string::npos) << r.hookErrors[0].message;
}

// Regression: the load marked the extract dirty only after its listeners ran, so a listener's catch-up (after a step)
// did not see the loaded entities
TEST(ExtractReview, LoadListenerCatchUpSeesTheLoad)
{
    World a(baseSetup());
    rendered(a, "test/box");
    rendered(a, "test/ball");
    a.step();
    const std::vector<uint8_t> blob = a.save();
    World b(baseSetup());
    RecordingRenderSink s;
    b.setRenderSink(&s);
    b.step();
    bool ran = false;
    b.addLoadListener([&](const LoadInfo&) { ran = b.extract().catchUp(); });
    ASSERT_TRUE(b.tryLoad(blob).ok());
    EXPECT_TRUE(ran);
    EXPECT_EQ(s.liveDynamics(), 2u);
    b.setRenderSink(nullptr);
}

// The world API refuses a non-uniform scale on every path that makes an entity rendered. Regression: add_component of
// RenderInstance on an entity with a non-uniform scale, and a class spawn whose onSpawn adds RenderInstance, accepted
// it
TEST(ExtractReview, NonUniformScaleOnEveryRenderedPath)
{
    World w(baseSetup());
    WorldApi api(w);
    const Entity e = w.create();
    w.get<Transform>(e).setScale(Vec3(1, 2, 1));
    ApiResult r = api.call("add_component", Json::object{{"entity", WorldApi::entityJson(e)},
                                                         {"component", "RenderInstance"},
                                                         {"values", Json::object{{"mesh", "test/box"}}}});
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("uniform"), std::string::npos) << r.error;
    EXPECT_FALSE(w.has<RenderInstance>(e));
    r = api.call("spawn",
                 Json::object{{"class", "Crate"}, {"position", Json::array{0, 0, 0}}, {"scale", Json::array{1, 3, 1}}});
    ASSERT_TRUE(r.ok) << r.error;
    Entity c;
    std::string error;
    ASSERT_TRUE(WorldApi::parseEntity(r.value["entity"], c, error)) << error;
    EXPECT_EQ(w.read<Transform>(c).scale(), Vec3::Ones());
    ASSERT_TRUE(r.value["skipped"].is_array()) << r.value.dump();
    EXPECT_EQ(r.value["skipped"][0]["component"].string_value(), "Transform");
    // Uniform scales stay valid
    r = api.call("spawn",
                 Json::object{{"class", "Crate"}, {"position", Json::array{0, 0, 0}}, {"scale", Json::array{2, 2, 2}}});
    ASSERT_TRUE(r.ok) << r.error;
    ASSERT_TRUE(WorldApi::parseEntity(r.value["entity"], c, error)) << error;
    EXPECT_EQ(w.read<Transform>(c).scale(), Vec3(2, 2, 2));
}

// ---------------------------------------------------------------------------------------------------------------------
// CodeRabbit follow-ups (S19b, PR #65)

// Regression: a name record (untrack and track of the same entity) between two commits appended the RenderPose entity
// to the remote list again on every paused catch-up
TEST(ExtractReview, RetrackedRemoteKeepsOneEntry)
{
    World w(baseSetup());
    RecordingRenderSink s;
    w.setRenderSink(&s);
    const Entity e = rendered(w, "test/box");
    w.add<RenderPose>(e);
    w.step();
    EXPECT_EQ(w.extract().remoteQueued(), 1u);
    w.setPaused(true);
    w.extract().catchUp();
    for (int i = 0; i < 10; ++i)
    {
        w.setRenderMesh(e, i % 2 ? "test/box" : "test/ball");
        w.extract().catchUp();
    }
    EXPECT_EQ(w.extract().remoteQueued(), 1u);
    EXPECT_EQ(s.liveDynamics(), 1u);
    w.setRenderSink(nullptr);
}

// Regression: a catch-up of a running world (only marked dirty by an edit) whose setDynamic threw had cleared the mark,
// so the next catch-up did not run and the sink kept the old pose until the next step
TEST(ExtractReview, FailedDirtyCatchUpRunsAgain)
{
    World w(baseSetup());
    ThrowingSink s;
    w.setRenderSink(&s);
    const Entity e = rendered(w, "test/box");
    w.step();
    w.extract().catchUp();
    WorldApi api(w);
    ASSERT_TRUE(
        api.call("set_transform", Json::object{{"entity", WorldApi::entityJson(e)}, {"position", Json::array{0, 4, 0}}})
            .ok);
    s.throwOnSet = 1;
    EXPECT_THROW(w.extract().catchUp(), std::runtime_error);
    EXPECT_TRUE(w.extract().catchUp());
    const RecordingRenderSink::Dynamic* d = dynamicOf(w, s, e);
    ASSERT_NE(d, nullptr);
    EXPECT_NEAR(worldOf(d->chunk, d->curr).position.y(), 4, 1e-12);
    w.setRenderSink(nullptr);
}

// Regression: a class spawn whose onSpawn set a non-uniform scale got a requested RenderInstance (the request had no
// scale, so the preflight saw [1, 1, 1])
TEST(ExtractReview, ClassScaleIsCheckedBeforeARequestedRenderInstance)
{
    World w(baseSetup());
    WorldApi api(w);
    ApiResult r = api.call(
        "spawn", Json::object{{"class", "Stretched"},
                              {"position", Json::array{0, 0, 0}},
                              {"components", Json::object{{"RenderInstance", Json::object{{"mesh", "test/box"}}}}}});
    ASSERT_TRUE(r.ok) << r.error;
    Entity e;
    std::string error;
    ASSERT_TRUE(WorldApi::parseEntity(r.value["entity"], e, error)) << error;
    EXPECT_FALSE(w.has<RenderInstance>(e));
    ASSERT_TRUE(r.value["skipped"].is_array()) << r.value.dump();
    EXPECT_EQ(r.value["skipped"][0]["component"].string_value(), "RenderInstance");
    // A uniform requested scale replaces the class's: rendered
    r = api.call("spawn",
                 Json::object{{"class", "Stretched"},
                              {"position", Json::array{0, 0, 0}},
                              {"scale", Json::array{3, 3, 3}},
                              {"components", Json::object{{"RenderInstance", Json::object{{"mesh", "test/box"}}}}}});
    ASSERT_TRUE(r.ok) << r.error;
    ASSERT_TRUE(WorldApi::parseEntity(r.value["entity"], e, error)) << error;
    EXPECT_TRUE(w.has<RenderInstance>(e));
    EXPECT_TRUE(r.value["skipped"].is_null()) << r.value.dump();
}

// setRenderMesh with the unchanged name in onLoad is no change (allowed; a different name is refused,
// SetRenderMeshInOnLoadIsRefused)
TEST(ExtractReview, SetRenderMeshInOnLoadWithTheSameName)
{
    World a(baseSetup());
    WorldApi api(a);
    ASSERT_TRUE(api.call("spawn", Json::object{{"class", "Keep"}, {"position", Json::array{0, 0, 0}}}).ok);
    a.step();
    World b(baseSetup());
    const LoadResult r = b.tryLoad(a.save());
    ASSERT_TRUE(r.committed) << r.error;
    EXPECT_TRUE(r.hookErrors.empty());
    EXPECT_EQ(b.hash(), a.hash());
}

#ifdef SAIGA_ASSERTS
// Regression: registerMeshAlias was accepted inside a step (registerMesh and registerMaterial assert), where it could
// not create the instances of entities that name the old name
TEST(ExtractReviewDeathTest, RegisterMeshAliasInsideAStep)
{
    WorldSetup st = baseSetup();
    st.systems    = [](SystemList& l) { l.add(Phase::Gameplay, "test_alias", aliasInStepSystem, RunOn::Host); };
    World w(st);
    EXPECT_DEATH(w.step(), "registerMeshAlias inside a step");
}
#endif

// ---------------------------------------------------------------------------------------------------------------------
// Spaces (S21)

// A space change is a teleport pair (prev = curr) in the new space's chunk, also when the position stays
TEST(ExtractSpaces, MoveIsATeleportInTheNewSpace)
{
    WorldSetup setup            = baseSetup();
    setup.config.spaceTemplates = twoSpaces();
    World w(setup);
    RecordingRenderSink s;
    w.setRenderSink(&s);
    const Entity a = rendered(w, "test/box", Vec3(5, 0, 5));
    w.add<Mover>(a, Mover{Vec3(60, 0, 0), 0});
    w.step(3);
    const RecordingRenderSink::Dynamic* d = dynamicOf(w, s, a);
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(d->chunk.space, 0u);
    EXPECT_NE(d->prev.position, d->curr.position);
    w.moveToSpace(a, 1, Vec3(640, 0, 640), Quat::Identity());
    w.step();
    EXPECT_EQ(d->chunk.space, 1u);
    EXPECT_EQ(worldOf(d->chunk, d->prev).position, worldOf(d->chunk, d->curr).position);
    EXPECT_NEAR(worldOf(d->chunk, d->curr).position.x(), 641, 1e-9);  // moved by the step's mover
    w.step();
    EXPECT_NE(d->prev.position, d->curr.position);
    // Back to space 0 at the current position (only the space changes): a snap too
    w.remove<Mover>(a);
    w.step();
    const Vec3 here = w.read<Transform>(a).position();
    w.moveToSpace(a, 0, here, Quat::Identity());
    w.step();
    EXPECT_EQ(d->chunk.space, 0u);
    EXPECT_EQ(d->prev.position, d->curr.position);
    w.setRenderSink(nullptr);
}

// A client draws a remote entity in its RenderPose's space (delayed like the pose), not in its Transform's
TEST(ExtractSpaces, ClientUsesTheRenderPoseSpace)
{
    Net n(0, true);
    const Entity e = rendered(*n.host, "test/box", Vec3(1, 0, 1));
    n.host->add<Mover>(e, Mover{Vec3(6, 0, 0), 0});
    n.step(10);
    const Entity c = n.onClient(e);
    ASSERT_TRUE(c.valid());
    const RecordingRenderSink::Dynamic* d = dynamicOf(n.client(), n.clientSink(), c);
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(d->chunk.space, 0u);
    n.host->moveToSpace(e, 1, Vec3(640, 0, 640), Quat::Identity());
    bool sawDelay = false, blended = false;
    for (int i = 0; i < 20; ++i)
    {
        n.step();
        const SpaceId transformSpace = n.client().read<Transform>(c).space();
        const RenderPose& rp         = n.client().read<RenderPose>(c);
        EXPECT_EQ(d->chunk.space, rp.space);
        sawDelay = sawDelay || (transformSpace == 1 && d->chunk.space == 0);
        if (d->chunk.space == 1)
            blended = blended || worldOf(d->chunk, d->prev).position.x() < 600 ||
                      worldOf(d->chunk, d->curr).position.x() < 600;
    }
    EXPECT_TRUE(sawDelay);
    EXPECT_FALSE(blended);
    EXPECT_EQ(d->chunk.space, 1u);
}

// The World forwards one setSpaceSettings per SpaceId at setRenderSink (also to a sink set later): open world or
// enclosed with the template's background
TEST(ExtractSpaces, SpaceSettingsForwarded)
{
    WorldSetup setup            = baseSetup();
    setup.config.spaceTemplates = twoSpaces();
    setup.config.spaceTemplates.push_back(SpaceTemplate::enclosed("vault", 2));
    setup.config.spaceTemplates[2].background = Vec3(0.5, 0.25, 0.125);
    World w(setup);
    RecordingRenderSink s;
    w.setRenderSink(&s);
    ASSERT_EQ(s.spaceSettingsCalls().size(), 3u);
    for (const auto& [space, calls] : s.spaceSettingsCalls()) EXPECT_EQ(calls, 1) << space;
    EXPECT_TRUE(s.spaceSettings().at(0).openWorld);
    EXPECT_FALSE(s.spaceSettings().at(1).openWorld);
    EXPECT_FALSE(s.spaceSettings().at(2).openWorld);
    EXPECT_EQ(s.spaceSettings().at(1).background, SpaceTemplate().background.cast<float>());
    EXPECT_EQ(s.spaceSettings().at(2).background, vec3(0.5f, 0.25f, 0.125f));
    w.step();
    w.setRenderSink(nullptr);
    // A sink set later gets them too
    RecordingRenderSink later;
    w.setRenderSink(&later);
    EXPECT_EQ(later.counters().setSpaceSettings, 3u);
    EXPECT_EQ(s.counters().setSpaceSettings, 3u);
    w.setRenderSink(nullptr);
}
