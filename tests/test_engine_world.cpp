/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The World and its step loop (docs/engine_design.md §7.6-§7.10, §8.1-§8.3, §8.6; stage S08): phases, systems,
// commands, the structural log, events, the Transform hierarchy, advance/step, parallelEach and the hash.

#include "saiga/engine/components/LocalTransform.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/world/World.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Saiga;
using namespace Saiga::Engine;

namespace
{
enum class TestComponent : uint8_t
{
    Health = kFirstGameComponent,
    Velocity,
    Derived,
    Count
};

struct Health
{
    static constexpr const char* name = "Health";
    static constexpr ComponentId id   = ComponentId(TestComponent::Health);
    static constexpr uint32_t version = 1;
    double hp                         = 100;
    template <class V>
    void visit(V& v)
    {
        v("hp", hp);
    }
};

struct Velocity
{
    static constexpr const char* name = "Velocity";
    static constexpr ComponentId id   = ComponentId(TestComponent::Velocity);
    static constexpr uint32_t version = 1;
    Vec3 v                            = Vec3::Zero();
    template <class V>
    void visit(V& vis)
    {
        vis("v", v);
    }
};

// Not saved (rebuilt after loading): not part of the hash
struct Derived
{
    static constexpr const char* name = "Derived";
    static constexpr ComponentId id   = ComponentId(TestComponent::Derived);
    static constexpr uint32_t version = 1;
    double cache                      = 0;
    template <class V>
    void visit(V& v)
    {
        v("cache", cache);
    }
};

enum class TestEvent : uint16_t
{
    Damage = 64,
    Count
};

struct DamageEvent
{
    static constexpr const char* name = "DamageEvent";
    static constexpr EventId id       = EventId(TestEvent::Damage);
    static constexpr uint32_t version = 1;
    Entity target;
    double amount = 0;
    template <class V>
    void visit(V& v)
    {
        v("target", target);
        v("amount", amount);
    }
};

void testComponents(Registry& r)
{
    r.registerComponent<Health>(Replicate::Host);
    r.registerComponent<Velocity>(Replicate::OwnerAuthoritative);
    r.registerComponent<Derived>(Replicate::Never, Saved::No);
}

void testEvents(EventRegistry& e)
{
    e.add<DamageEvent>();
}

WorldSetup testSetup(void (*systems)(SystemList&) = nullptr)
{
    WorldSetup s;
    s.components = testComponents;
    s.events     = testEvents;
    s.systems    = systems;
    return s;
}

// A trace of system calls shared by the test systems
std::vector<std::string> g_trace;
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// Registration and phases

TEST(World, EngineListsFirstAndRegistration)
{
    World w(testSetup());
    EXPECT_TRUE(w.registry().registered(Transform::id));
    EXPECT_TRUE(w.registry().registered(Parent::id));
    EXPECT_TRUE(w.registry().registered(LocalTransform::id));
    EXPECT_TRUE(w.registry().registered(Health::id));
    EXPECT_TRUE(w.events.registered(DamageEvent::id));
    EXPECT_EQ(w.dt(), 1.0 / 60.0);
    EXPECT_EQ(w.stepIndex(), 0u);
    EXPECT_EQ(w.currentPhase(), Phase::None);
}

namespace
{
void sysInputA(World& w, double)
{
    g_trace.push_back(std::string("input_a@") + phaseName(w.currentPhase()));
}
void sysInputB(World&, double)
{
    g_trace.push_back("input_b");
}
void sysGameplay(World&, double)
{
    g_trace.push_back("gameplay");
}
void sysExtract(World&, double)
{
    g_trace.push_back("extract");
}
void sysClientOnly(World&, double)
{
    g_trace.push_back("client");
}
void sysBoth(World&, double)
{
    g_trace.push_back("both");
}
void orderSystems(SystemList& s)
{
    // Registered out of phase order: they run in phase order, inside a phase in registration order
    s.add(Phase::Extract, "extract", sysExtract);
    s.add(Phase::Gameplay, "gameplay", sysGameplay);
    s.add(Phase::Input, "input_a", sysInputA);
    s.add(Phase::Input, "input_b", sysInputB);
    s.add(Phase::Gameplay, "client_only", sysClientOnly, RunOn::Client);
    s.add(Phase::Gameplay, "both", sysBoth, RunOn::Both);
}
}  // namespace

TEST(World, SystemsRunInPhaseAndRegistrationOrder)
{
    g_trace.clear();
    World w(testSetup(orderSystems));
    w.step();
    // The host does not run RunOn::Client systems (before S16 every world is the host)
    EXPECT_EQ(g_trace, (std::vector<std::string>{"input_a@input", "input_b", "gameplay", "both", "extract"}));
    EXPECT_EQ(w.stepIndex(), 1u);
}

#ifdef SAIGA_ASSERTS
TEST(WorldDeathTest, DuplicateSystemName)
{
    SystemList s;
    s.add(Phase::Gameplay, "a", sysGameplay);
    EXPECT_DEATH(s.add(Phase::Input, "a", sysInputA), "already used");
}
#endif

// ---------------------------------------------------------------------------------------------------------------------
// Commands

namespace
{
Entity g_created;
bool g_sawReserved        = false;
size_t g_countInSamePhase = 0, g_countInNextPhase = 0;

void sysCreateInGameplay(World& w, double)
{
    w.view<Health>().each(
        [&](Entity, Health&)
        {
            if (g_created.valid()) return;
            g_created = w.create();
            w.add<Velocity>(g_created, Vec3(1, 2, 3));
            g_sawReserved = !w.alive(g_created);
        });
    g_countInSamePhase = w.viewAll<Velocity>().count();
}
void sysCountInPhysics(World& w, double)
{
    g_countInNextPhase = w.view<Velocity>().count();
}
void createSystems(SystemList& s)
{
    s.add(Phase::Gameplay, "create", sysCreateInGameplay);
    s.add(Phase::Physics, "count", sysCountInPhysics);
}
}  // namespace

TEST(World, CreateDuringAStepIsAppliedAfterThePhase)
{
    g_created     = Entity{};
    g_sawReserved = false;
    World w(testSetup(createSystems));
    Entity a = w.create();
    w.add<Health>(a);
    w.step();
    ASSERT_TRUE(g_created.valid());
    EXPECT_TRUE(g_sawReserved);         // reserved, not alive inside the phase
    EXPECT_EQ(g_countInSamePhase, 0u);  // views skip the reserved entity and its pending add
    EXPECT_EQ(g_countInNextPhase, 1u);  // applied at the end of the gameplay phase
    EXPECT_TRUE(w.alive(g_created));
    EXPECT_EQ(w.read<Velocity>(g_created).v, Vec3(1, 2, 3));
}

namespace
{
void sysDestroyAll(World& w, double)
{
    // Destroy and remove during a view are deferred, so the view is not disturbed
    w.view<Health>().each(
        [&](Entity e, Health& h)
        {
            if (h.hp < 50)
                w.destroy(e);
            else
                w.remove<Health>(e);
        });
}
void destroySystems(SystemList& s)
{
    s.add(Phase::Gameplay, "destroy", sysDestroyAll);
}
}  // namespace

TEST(World, DestroyAndRemoveDuringAViewAreDeferred)
{
    World w(testSetup(destroySystems));
    std::vector<Entity> e;
    for (int i = 0; i < 10; ++i)
    {
        e.push_back(w.create());
        w.add<Health>(e.back(), double(i * 10));
    }
    w.step();
    for (int i = 0; i < 10; ++i)
    {
        if (i < 5)
            EXPECT_FALSE(w.alive(e[size_t(i)])) << i;
        else
        {
            EXPECT_TRUE(w.alive(e[size_t(i)])) << i;
            EXPECT_FALSE(w.has<Health>(e[size_t(i)])) << i;
        }
    }
}

namespace
{
std::vector<std::string> g_order;
void sysRecordingOrder(World& w, double)
{
    if (w.stepIndex() != 0) return;
    Entity e = w.create();
    w.add<Health>(e, 1.0);
    w.remove<Health>(e);
    w.add<Health>(e, 2.0);  // applied in recording order: the final value is 2
    g_created = e;
    // A second destroy of the same entity is dropped (it is gone when the command runs)
    Entity d = w.create();
    w.destroy(d);
    w.destroy(d);
}
void orderCommandSystems(SystemList& s)
{
    s.add(Phase::Gameplay, "order", sysRecordingOrder);
}
}  // namespace

TEST(World, CommandsApplyInRecordingOrder)
{
    World w(testSetup(orderCommandSystems));
    w.step();
    EXPECT_EQ(w.read<Health>(g_created).hp, 2.0);
    EXPECT_EQ(w.registry().aliveCount(), 1u);
}

TEST(World, ChangesBetweenStepsApplyImmediately)
{
    World w(testSetup());
    Entity e = w.create();
    EXPECT_TRUE(w.alive(e));
    w.add<Health>(e, 5.0);
    EXPECT_EQ(w.read<Health>(e).hp, 5.0);
    w.remove<Health>(e);
    EXPECT_FALSE(w.has<Health>(e));
    w.destroy(e);
    EXPECT_FALSE(w.alive(e));
}

namespace
{
bool g_referenceStable = false;
void sysCreateWhileHoldingTransform(World& w, double)
{
    w.view<Transform, Health>().each(
        [&](Entity, Transform& t, Health&)
        {
            const Transform* before = &t;
            // Many reservations: the slot and Transform arrays grow by pages, existing ones do not move
            for (int i = 0; i < 10000; ++i) w.create();
            t.setPosition(Vec3(7, 7, 7));
            g_referenceStable = (before == &t);
        });
}
void stableSystems(SystemList& s)
{
    s.add(Phase::Gameplay, "stable", sysCreateWhileHoldingTransform);
}
}  // namespace

// Regression (S07 notes): a Transform& from a view must survive create() in the callback (reserving grows the slots)
TEST(World, TransformReferenceSurvivesCreateInAView)
{
    World w(testSetup(stableSystems));
    Entity e = w.create();
    w.add<Health>(e);
    w.step();
    EXPECT_TRUE(g_referenceStable);
    EXPECT_EQ(w.read<Transform>(e).position(), Vec3(7, 7, 7));
    EXPECT_EQ(w.registry().aliveCount(), 10001u);
}

#ifdef SAIGA_ASSERTS
namespace
{
void sysGetReserved(World& w, double)
{
    Entity e = w.create();
    w.get<Transform>(e);
}
void reservedSystems(SystemList& s)
{
    s.add(Phase::Gameplay, "get_reserved", sysGetReserved);
}
}  // namespace

TEST(WorldDeathTest, GetOnAReservedEntityAsserts)
{
    World w(testSetup(reservedSystems));
    EXPECT_DEATH(w.step(), "reserved");
}
#endif

// ---------------------------------------------------------------------------------------------------------------------
// Structural log

TEST(World, StructuralLogRecordsAndConsumers)
{
    World w(testSetup());
    StructuralLog& log = w.structuralLog();
    int a              = log.addConsumer();
    Entity e           = w.create();
    w.add<Health>(e);
    w.add<Velocity>(e);
    w.remove<Health>(e);
    int b = log.addConsumer();  // starts at the current end
    w.destroy(e);

    std::vector<StructuralLog::Record> ra, rb;
    log.consume(a, [&](const StructuralLog::Record& r) { ra.push_back(r); });
    log.consume(b, [&](const StructuralLog::Record& r) { rb.push_back(r); });
    // create (Transform added), add Health, add Velocity, remove Health, destroy: Transform and Velocity removed
    ASSERT_EQ(ra.size(), 6u);
    EXPECT_TRUE(ra[0].added && ra[0].component == Transform::id);
    EXPECT_TRUE(ra[1].added && ra[1].component == Health::id);
    EXPECT_TRUE(ra[2].added && ra[2].component == Velocity::id);
    EXPECT_TRUE(!ra[3].added && ra[3].component == Health::id);
    EXPECT_TRUE(!ra[4].added && ra[4].component == Transform::id);
    EXPECT_TRUE(!ra[5].added && ra[5].component == Velocity::id);
    ASSERT_EQ(rb.size(), 2u);
    EXPECT_EQ(rb[0].component, Transform::id);

    // Truncated at the end of a step up to the smallest cursor
    Entity f = w.create();
    EXPECT_EQ(log.unread(a), 1u);
    w.step();
    EXPECT_EQ(log.size(), 1u);  // a and b have not read the last record
    log.consume(a, [](const StructuralLog::Record&) {});
    w.step();
    EXPECT_EQ(log.size(), 1u);  // b still has not
    log.consume(b, [](const StructuralLog::Record&) {});
    w.step();
    EXPECT_EQ(log.size(), 0u);
    (void)f;
}

TEST(World, StructuralLogWithoutConsumersIsTruncated)
{
    World w(testSetup());
    for (int i = 0; i < 10; ++i) w.create();
    EXPECT_EQ(w.structuralLog().size(), 10u);
    w.step();
    EXPECT_EQ(w.structuralLog().size(), 0u);
}

// ---------------------------------------------------------------------------------------------------------------------
// Events

namespace
{
std::vector<std::pair<uint64_t, double>> g_events;  // (step, amount) read
void sysEmitAndRead(World& w, double)
{
    for (const DamageEvent& d : w.events.read<DamageEvent>()) g_events.push_back({w.stepIndex(), d.amount});
    if (w.stepIndex() == 1) w.events.emit(DamageEvent{Entity{}, 10});
    if (w.stepIndex() == 1) w.events.emit(DamageEvent{Entity{}, 11});
}
void sysReadLater(World& w, double)
{
    // A later phase of the same step sees the same readable events
    for (const DamageEvent& d : w.events.read<DamageEvent>()) g_events.push_back({w.stepIndex() + 100, d.amount});
}
void eventSystems(SystemList& s)
{
    s.add(Phase::Gameplay, "emit_read", sysEmitAndRead);
    s.add(Phase::Extract, "read_later", sysReadLater);
}
}  // namespace

TEST(World, EventsAreReadableInTheNextStepOnly)
{
    g_events.clear();
    World w(testSetup(eventSystems));
    w.events.emit(DamageEvent{Entity{}, 1});  // before the first step: readable in step 0 (the first step)
    w.step(4);
    // Step index during a step is the index of that step (0-based): emitted between steps -> read in step 0;
    // emitted in step 1 -> read in step 2 (emit order), then cleared
    std::vector<std::pair<uint64_t, double>> expected = {{0, 1}, {100, 1}, {2, 10}, {2, 11}, {102, 10}, {102, 11}};
    EXPECT_EQ(g_events, expected);
}

#ifdef SAIGA_ASSERTS
namespace
{
struct UnregisteredEvent
{
    static constexpr const char* name = "UnregisteredEvent";
    static constexpr EventId id       = 70;
    static constexpr uint32_t version = 1;
    template <class V>
    void visit(V&)
    {
    }
};
}  // namespace

TEST(WorldDeathTest, UnregisteredEventAsserts)
{
    World w(testSetup());
    EXPECT_DEATH(w.events.emit(UnregisteredEvent{}), "not registered");
}
#endif

// ---------------------------------------------------------------------------------------------------------------------
// Transform hierarchy

TEST(World, ChildFollowsParentInDepthOrder)
{
    World w(testSetup());
    Entity root = w.create(), child = w.create(), grandchild = w.create();
    // The grandchild is created before its parent is attached: the depth order, not the index order, counts
    w.add<Parent>(grandchild, Parent{child});
    w.add<LocalTransform>(grandchild, LocalTransform{Vec3(0, 0, 1), Quat::Identity(), Vec3::Ones()});
    w.add<Parent>(child, Parent{root});
    w.add<LocalTransform>(child, LocalTransform{Vec3(1, 0, 0), Quat::Identity(), Vec3(2, 2, 2)});

    // Root at (10, 0, 0), rotated 90 degrees about y
    Quat rot = Quat(AngleAxis(3.14159265358979323846 / 2, Vec3::UnitY()));
    w.get<Transform>(root).setPosition(Vec3(10, 0, 0));
    w.get<Transform>(root).setRotation(rot);
    w.step();

    // child = root.p + root.r * (1, 0, 0) = (10, 0, -1); grandchild = child.p + child.r * (child.s * (0, 0, 1))
    EXPECT_TRUE(w.read<Transform>(child).position().isApprox(Vec3(10, 0, -1), 1e-12));
    EXPECT_TRUE(w.read<Transform>(child).scale().isApprox(Vec3(2, 2, 2)));
    EXPECT_TRUE(w.read<Transform>(grandchild).position().isApprox(Vec3(12, 0, -1), 1e-12));
    EXPECT_TRUE(w.read<Transform>(grandchild).rotation().isApprox(rot, 1e-12));
}

namespace
{
std::vector<uint32_t> g_changed;
bool g_teleportedChild = false;
Entity g_child;
void sysRecordChanged(World& w, double)
{
    g_changed         = w.changedTransforms();
    g_teleportedChild = g_child.valid() && w.read<Transform>(g_child).wasTeleported();
}
void changedSystems(SystemList& s)
{
    // After the engine's transform system (phase 6), before the flags shift
    s.add(Phase::Extract, "record_changed", sysRecordChanged);
}
}  // namespace

TEST(World, ChangedListTeleportAndFlagShift)
{
    World w(testSetup(changedSystems));
    Entity parent = w.create(), child = w.create(), idle = w.create();
    w.add<Parent>(child, Parent{parent});
    g_child = child;
    w.step();  // a new Transform has no change flag: nothing is listed
    EXPECT_TRUE(g_changed.empty());

    w.get<Transform>(parent).setPosition(Vec3(1, 0, 0));
    w.step();
    // Changed this step: the parent (set between steps) and the child (moved by the hierarchy)
    EXPECT_EQ(g_changed, (std::vector<uint32_t>{parent.index, child.index}));
    EXPECT_FALSE(g_teleportedChild);

    w.step();
    // Changed in the last step only: still listed once more (changedLastStep), the idle entity never
    EXPECT_EQ(g_changed, (std::vector<uint32_t>{parent.index, child.index}));
    w.step();
    EXPECT_TRUE(g_changed.empty());

    // A teleported parent teleports its children
    w.get<Transform>(parent).teleport(Vec3(50, 0, 0), Quat::Identity());
    w.step();
    EXPECT_TRUE(g_teleportedChild);
    EXPECT_EQ(w.read<Transform>(child).position(), Vec3(50, 0, 0));
    // The flags were shifted at the end of the step
    EXPECT_FALSE(w.read<Transform>(child).changedInThisStep());
    EXPECT_TRUE(w.read<Transform>(child).changedInLastStep());
    EXPECT_FALSE(w.read<Transform>(child).wasTeleported());
    (void)idle;
    g_child = Entity{};
}

TEST(World, DestroyingAParentDestroysItsDescendants)
{
    World w(testSetup());
    Entity root = w.create(), a = w.create(), b = w.create(), c = w.create(), other = w.create();
    w.add<Parent>(a, Parent{root});
    w.add<Parent>(b, Parent{a});
    w.add<Parent>(c, Parent{root});
    w.step();
    w.destroy(root);
    EXPECT_FALSE(w.alive(root));
    EXPECT_FALSE(w.alive(a));
    EXPECT_FALSE(w.alive(b));
    EXPECT_FALSE(w.alive(c));
    EXPECT_TRUE(w.alive(other));
    w.step();  // the hierarchy list is rebuilt without them
    EXPECT_EQ(w.registry().aliveCount(), 1u);
}

// ---------------------------------------------------------------------------------------------------------------------
// Time

TEST(World, AdvanceRunsWholeStepsAndDropsTheBacklog)
{
    World w(testSetup());
    const double dt = w.dt();
    w.advance(dt * 2.5);
    EXPECT_EQ(w.stepIndex(), 2u);
    EXPECT_NEAR(w.alpha(), 0.5, 1e-9);
    // A long frame: at most 4 steps, the rest is dropped and counted
    w.advance(dt * 10.25);
    EXPECT_EQ(w.stepIndex(), 6u);
    EXPECT_GT(w.droppedSteps(), 0u);
    EXPECT_GE(w.alpha(), 0.0);
    EXPECT_LT(w.alpha(), 1.0);
    // Pause and time scale
    w.setPaused(true);
    w.advance(dt * 3);
    EXPECT_EQ(w.stepIndex(), 6u);
    w.setPaused(false);
    w.setTimeScale(2.0);
    uint64_t before = w.stepIndex();
    w.advance(dt * 1.0);
    EXPECT_GE(w.stepIndex(), before + 2);
    // step(n) runs exactly n steps
    w.step(3);
    EXPECT_NEAR(w.time(), double(w.stepIndex()) * dt, 1e-12);
}

// ---------------------------------------------------------------------------------------------------------------------
// parallelEach and Simulated

namespace
{
void sysIntegrate(World& w, double dt)
{
    w.view<Transform, const Velocity>().parallelEach(
        [dt](Entity, Transform& t, const Velocity& v) { t.setPosition(t.position() + v.v * dt); }, 7);
}
void integrateSystems(SystemList& s)
{
    s.add(Phase::Gameplay, "integrate", sysIntegrate);
}
}  // namespace

TEST(World, ParallelEachVisitsEveryEntityOnceForAnyThreadCount)
{
    std::vector<uint64_t> hashes;
    for (int threads : {1, 2, 4})
    {
        WorldSetup s    = testSetup(integrateSystems);
        s.workerThreads = threads;
        World w(s);
        for (int i = 0; i < 1000; ++i)
        {
            Entity e = w.create();
            if (i % 3 != 0) w.add<Velocity>(e, Vec3(i, 1, 0));
        }
        w.step(5);
        hashes.push_back(w.hash());
        // Every entity with a velocity moved 5 steps
        size_t moved = 0;
        w.viewAll<const Transform, const Velocity>().each(
            [&](Entity, const Transform& t, const Velocity& v)
            {
                moved++;
                EXPECT_NEAR(t.position()(1), 5 * w.dt() * v.v(1), 1e-12);
            });
        EXPECT_EQ(moved, 666u);
    }
    // The result does not depend on the thread count
    EXPECT_EQ(hashes[0], hashes[1]);
    EXPECT_EQ(hashes[0], hashes[2]);
}

#ifdef SAIGA_ASSERTS
namespace
{
void sysParallelCommand(World& w, double)
{
    w.view<Health>().parallelEach([&](Entity e, Health&) { w.destroy(e); });
}
void sysParallelEmit(World& w, double)
{
    w.view<Health>().parallelEach([&](Entity, Health&) { w.events.emit(DamageEvent{}); });
}
void sysParallelRandom(World& w, double)
{
    w.view<Health>().parallelEach([&](Entity, Health& h) { h.hp = w.random().uniform(); });
}
void sysParallelCreate(World& w, double)
{
    w.view<Health>().parallelEach([&](Entity, Health&) { w.create(); });
}
}  // namespace

TEST(WorldDeathTest, ParallelEachForbidsCommandsEventsAndRandom)
{
    auto run = [](void (*sys)(World&, double))
    {
        static void (*current)(World&, double) = nullptr;
        current                                = sys;
        WorldSetup s                           = testSetup([](SystemList& l) { l.add(Phase::Gameplay, "p", current); });
        s.workerThreads                        = 4;  // the assert fires on a worker thread too
        World w(s);
        for (int i = 0; i < 63; ++i) w.add<Health>(w.create());
        w.add<Health>(w.create());
        w.step();
    };
    EXPECT_DEATH(run(sysParallelCommand), "inside parallelEach");
    EXPECT_DEATH(run(sysParallelEmit), "inside parallelEach");
    EXPECT_DEATH(run(sysParallelRandom), "inside parallelEach");
    EXPECT_DEATH(run(sysParallelCreate), "inside parallelEach");
}
#endif

namespace
{
size_t g_simulatedSeen = 0, g_allSeen = 0, g_extractSeen = 0;
void sysCountSimulated(World& w, double)
{
    g_simulatedSeen = w.view<Health>().count();
    g_allSeen       = w.viewAll<Health>().count();
}
void sysCountInExtract(World& w, double)
{
    g_extractSeen = w.view<Health>().count();
}
void simulatedSystems(SystemList& s)
{
    s.add(Phase::Gameplay, "count_simulated", sysCountSimulated);
    s.add(Phase::Extract, "count_extract", sysCountInExtract);
}
}  // namespace

TEST(World, ViewSkipsNonSimulatedEntitiesInPhases2To5)
{
    World w(testSetup(simulatedSystems));
    Entity a = w.create(), b = w.create();
    w.add<Health>(a);
    w.add<Health>(b);
    w.registry().setFlags(b, 0);  // frozen (streaming clears the flag from S24)
    w.step();
    EXPECT_EQ(g_simulatedSeen, 1u);
    EXPECT_EQ(g_allSeen, 2u);
    EXPECT_EQ(g_extractSeen, 2u);  // phase 8 is not filtered
}

// ---------------------------------------------------------------------------------------------------------------------
// Hash

namespace
{
void sysRandomWalk(World& w, double)
{
    w.view<Health>().each([&](Entity, Health& h) { h.hp += w.random().uniform(-1, 1); });
    if (w.stepIndex() % 3 == 0) w.events.emit(DamageEvent{Entity{}, double(w.stepIndex())});
    if (w.stepIndex() % 5 == 0) w.add<Health>(w.create(), 1.0);
}
void walkSystems(SystemList& s)
{
    s.add(Phase::Gameplay, "walk", sysRandomWalk);
}
}  // namespace

TEST(World, HashIsEqualForTwoRunsAndSensitiveToState)
{
    WorldSetup s = testSetup(walkSystems);
    s.seed       = 7;
    World a(s), b(s);
    for (World* w : {&a, &b})
        for (int i = 0; i < 10; ++i) w->add<Health>(w->create(), double(i));
    for (int i = 0; i < 30; ++i)
    {
        a.step();
        b.step();
        ASSERT_EQ(a.hash(), b.hash()) << "step " << i;
    }

    uint64_t h = a.hash();
    // An unsaved component does not change the hash
    a.add<Derived>(Entity{1, 0}, 3.0);
    EXPECT_EQ(a.hash(), h);
    // A changed field, the RNG state, a pending event and the free-list order do
    a.get<Health>(Entity{1, 0}).hp += 1;
    EXPECT_NE(a.hash(), h);
    a.get<Health>(Entity{1, 0}).hp -= 1;
    EXPECT_EQ(a.hash(), h);
    a.random().nextU32();
    EXPECT_NE(a.hash(), b.hash());

    World c(s), d(s);
    for (World* w : {&c, &d})
        for (int i = 0; i < 3; ++i) w->create();
    EXPECT_EQ(c.hash(), d.hash());
    c.events.emit(DamageEvent{});
    EXPECT_NE(c.hash(), d.hash());

    World e(s), f(s);
    for (World* w : {&e, &f})
        for (int i = 0; i < 3; ++i) w->create();
    e.destroy(Entity{1, 0});
    e.destroy(Entity{2, 0});
    f.destroy(Entity{2, 0});
    f.destroy(Entity{1, 0});
    EXPECT_NE(e.hash(), f.hash());  // same alive set, different free-list order
}

// ---------------------------------------------------------------------------------------------------------------------
// Review round 1 regressions

#ifdef SAIGA_ASSERTS
// Regression: a Parent cycle made destroy loop forever; add<Parent> rejects cycles and dead parents
TEST(WorldDeathTest, ParentCyclesAndDeadParentsAreRejected)
{
    World w(testSetup());
    Entity a = w.create(), b = w.create(), c = w.create();
    EXPECT_DEATH(w.add<Parent>(a, Parent{a}), "a cycle");
    w.add<Parent>(b, Parent{a});
    w.add<Parent>(c, Parent{b});
    EXPECT_DEATH(w.add<Parent>(a, Parent{c}), "a cycle");
    Entity dead = w.create();
    w.destroy(dead);
    Entity d = w.create();
    EXPECT_DEATH(w.add<Parent>(d, Parent{dead}), "the parent is not alive");
}

TEST(WorldDeathTest, RecordTimeChecksOfAddAndRemove)
{
    auto run = [](void (*sys)(World&, double))
    {
        static void (*current)(World&, double) = nullptr;
        current                                = sys;
        World w(testSetup([](SystemList& l) { l.add(Phase::Gameplay, "s", current); }));
        Entity e = w.create();
        w.add<Health>(e);
        w.step();
    };
    EXPECT_DEATH(run([](World& w, double) { w.remove<Health>(Entity{999, 0}); }), "the entity is not alive");
    // Whether the entity has the component is checked when the command is applied (re-parenting needs remove + add)
    EXPECT_DEATH(run([](World& w, double) { w.remove<Velocity>(Entity{1, 0}); }), "has no Velocity");
    EXPECT_DEATH(run([](World& w, double) { w.add<Health>(Entity{1, 0}); }), "already has Health");
}
#endif

namespace
{
Entity g_parentToKill, g_orphan;
void sysDestroyParentThenAttach(World& w, double)
{
    if (w.stepIndex() != 0) return;
    // In one phase: destroy P, then attach C to P. The flush destroys P first; C follows its parent.
    w.destroy(g_parentToKill);
    w.add<Parent>(g_orphan, Parent{g_parentToKill});
}
void orphanSystems(SystemList& s)
{
    s.add(Phase::Gameplay, "orphan", sysDestroyParentThenAttach);
}
}  // namespace

// Regression: the add<Parent> was applied with a destroyed parent and phase 6 aborted
TEST(World, AttachingToAParentDestroyedInTheSameFlushDestroysTheChild)
{
    World w(testSetup(orphanSystems));
    g_parentToKill = w.create();
    g_orphan       = w.create();
    Entity other   = w.create();
    w.step();
    EXPECT_FALSE(w.alive(g_parentToKill));
    EXPECT_FALSE(w.alive(g_orphan));
    EXPECT_TRUE(w.alive(other));
}

namespace
{
Entity g_root;
void sysDestroyRootInFlush(World& w, double)
{
    if (w.stepIndex() == 1) w.destroy(g_root);
}
void cascadeSystems(SystemList& s)
{
    s.add(Phase::Gameplay, "cascade", sysDestroyRootInFlush);
}
}  // namespace

TEST(World, DestroyCascadeDuringAFlush)
{
    World w(testSetup(cascadeSystems));
    g_root = w.create();
    std::vector<Entity> chain{g_root};
    for (int i = 0; i < 50; ++i)
    {
        Entity c = w.create();
        w.add<Parent>(c, Parent{chain.back()});
        chain.push_back(c);
    }
    Entity sibling = w.create();
    w.add<Parent>(sibling, Parent{g_root});
    Entity unrelated = w.create();
    w.step(2);
    for (Entity e : chain) EXPECT_FALSE(w.alive(e));
    EXPECT_FALSE(w.alive(sibling));
    EXPECT_TRUE(w.alive(unrelated));
    EXPECT_EQ(w.registry().aliveCount(), 1u);
}

namespace
{
bool g_throw = false;
void sysCreateThenThrow(World& w, double)
{
    w.create();
    w.add<Health>(w.create(), 5.0);
    if (g_throw) throw std::runtime_error("system failed");
}
void throwingSystems(SystemList& s)
{
    s.add(Phase::Gameplay, "throws", sysCreateThenThrow);
}
}  // namespace

// Regression: a throwing system left its reserved entities and commands, applied in the next step
TEST(World, AThrowingSystemLeavesTheWorldUsable)
{
    World w(testSetup(throwingSystems));
    g_throw = true;
    EXPECT_THROW(w.step(), std::runtime_error);
    EXPECT_FALSE(w.inStep());
    EXPECT_EQ(w.stepIndex(), 0u);
    EXPECT_EQ(w.registry().aliveCount(), 0u);  // the reserved entities were released
    uint64_t h = w.hash();                     // allowed again (no pending commands)
    g_throw    = false;
    w.step();
    EXPECT_EQ(w.registry().aliveCount(), 2u);  // only this step's two entities
    EXPECT_NE(w.hash(), h);
}

namespace
{
Entity g_lateParent, g_lateChild, g_unrelatedChild;
bool g_childTeleportSeenByExtract = false;
std::vector<double> g_childXSeenByExtract;
void sysLateTeleport(World& w, double)
{
    if (w.stepIndex() != 1) return;
    w.get<Transform>(g_lateParent).teleport(Vec3(100, 0, 0), Quat::Identity());
    // A hierarchy change in the same phase must not cancel the propagation (streaming unloads parented entities)
    if (g_unrelatedChild.valid()) w.destroy(g_unrelatedChild);
}
void sysExtractReader(World& w, double)
{
    // Registered before the phase-8 teleport system: an extract that runs before a late teleport in the same phase
    g_childTeleportSeenByExtract |= w.read<Transform>(g_lateChild).wasTeleported();
    g_childXSeenByExtract.push_back(w.read<Transform>(g_lateChild).position()(0));
}
void lateStreamingSystems(SystemList& s)
{
    s.add(Phase::Streaming, "late_teleport", sysLateTeleport);
    s.add(Phase::Extract, "extract_reader", sysExtractReader);
}
void lateExtractSystems(SystemList& s)
{
    s.add(Phase::Extract, "extract_reader", sysExtractReader);
    s.add(Phase::Extract, "late_teleport", sysLateTeleport);
}

void setUpLateTeleport(World& w)
{
    g_lateParent     = w.create();
    g_lateChild      = w.create();
    Entity other     = w.create();
    g_unrelatedChild = w.create();
    w.add<Parent>(g_lateChild, Parent{g_lateParent});
    w.add<Parent>(g_unrelatedChild, Parent{other});
    g_childTeleportSeenByExtract = false;
    g_childXSeenByExtract.clear();
}
}  // namespace

// Regression: a teleport in phase 7 reached the children only after extract (and not at all if the hierarchy changed
// in the same phase), so extract interpolated the child across the jump
TEST(World, TeleportInPhase7ReachesTheChildrenBeforeExtract)
{
    World w(testSetup(lateStreamingSystems));
    setUpLateTeleport(w);
    w.step(2);
    EXPECT_TRUE(g_childTeleportSeenByExtract);
    ASSERT_EQ(g_childXSeenByExtract.size(), 2u);
    EXPECT_EQ(g_childXSeenByExtract[1], 100.0);  // the same step's extract sees the child at the target
    EXPECT_FALSE(w.alive(g_unrelatedChild));
}

// A teleport in phase 8 (after extract ran): the flag survives one step (teleportedLastStep) and the children are
// teleported in the next phase 6, so the next extract sees the jump flagged
TEST(World, TeleportInPhase8ReachesTheChildrenInTheNextStep)
{
    World w(testSetup(lateExtractSystems));
    setUpLateTeleport(w);
    w.step(2);
    EXPECT_FALSE(g_childTeleportSeenByExtract);
    EXPECT_TRUE(w.read<Transform>(g_lateParent).wasTeleportedLastStep());
    w.step();
    EXPECT_TRUE(g_childTeleportSeenByExtract);
    EXPECT_EQ(w.read<Transform>(g_lateChild).position(), Vec3(100, 0, 0));
    // Not teleported twice: in the following step the child moves normally again
    g_childTeleportSeenByExtract = false;
    w.step();
    EXPECT_FALSE(g_childTeleportSeenByExtract);
}

namespace
{
bool g_throwOnce    = false;
size_t g_eventsSeen = 0;
std::vector<double> g_amountsSeen;
Entity g_reservedByAbortedStep;
void sysReadThenThrow(World& w, double)
{
    auto events  = w.events.read<DamageEvent>();
    g_eventsSeen = events.size();
    g_amountsSeen.clear();
    for (auto& ev : events) g_amountsSeen.push_back(ev.amount);
    if (g_throwOnce)
    {
        g_throwOnce = false;
        w.events.emit(DamageEvent{Entity{}, 99});  // emitted by the aborted step: dropped with it
        g_reservedByAbortedStep = w.create();      // reserved by the aborted step: released
        throw std::runtime_error("once");
    }
}
void abortEventSystems(SystemList& s)
{
    s.add(Phase::Gameplay, "read_throw", sysReadThenThrow);
}
}  // namespace

// Regression: an aborted step consumed the events it had made readable; the repeated step saw none
TEST(World, AnAbortedStepKeepsItsEvents)
{
    World w(testSetup(abortEventSystems));
    w.events.emit(DamageEvent{Entity{}, 1});
    g_throwOnce = true;
    EXPECT_THROW(w.step(), std::runtime_error);
    EXPECT_EQ(w.registry().aliveCount(), 0u);
    // Released, not left reserved: the slot is free (its generation moved on)
    EXPECT_FALSE(w.registry().reserved(g_reservedByAbortedStep));
    EXPECT_EQ(w.registry().reservedCount(), 0u);
    EXPECT_FALSE(w.alive(g_reservedByAbortedStep));
    w.step();
    // The event of before the aborted step (amount 1), not the aborted step's own (99)
    EXPECT_EQ(g_amountsSeen, std::vector<double>{1});
    w.step();
    EXPECT_EQ(g_eventsSeen, 0u);
}

namespace
{
void sysCreateMany(World& w, double)
{
    for (int i = 0; i < 100000; ++i) w.create();
}
}  // namespace

// Regression: each create during a step reserved exactly one more entry for the abort cleanup, so every create
// reallocated and copied all entities reserved before it in the phase (quadratic)
TEST(World, ManyCreatesDuringAStepAreLinear)
{
    World w(testSetup([](SystemList& l) { l.add(Phase::Gameplay, "create_many", sysCreateMany); }));
    auto start = std::chrono::steady_clock::now();
    w.step();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    EXPECT_EQ(w.registry().aliveCount(), 100000u);
    // Generous bound (sanitizer builds); quadratic copying of 10^5 handles takes far longer
    EXPECT_LT(ms, 2000.0);
}

// Regression: every destroy sorted all Parent links; entities without children skip the cascade
TEST(World, DestroyWithoutChildrenIsCheapWithManyParents)
{
    World w(testSetup());
    Entity root = w.create();
    for (int i = 0; i < 20000; ++i) w.add<Parent>(w.create(), Parent{root});
    std::vector<Entity> loose;
    for (int i = 0; i < 2000; ++i) loose.push_back(w.create());
    auto start = std::chrono::steady_clock::now();
    for (Entity e : loose) w.destroy(e);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    // Generous bound (it was ~660 ms): a cascade per destroy would take far longer
    EXPECT_LT(ms, 200.0);
    // Destroying the root still destroys all children
    w.destroy(root);
    EXPECT_EQ(w.registry().aliveCount(), 0u);
}

#ifdef SAIGA_ASSERTS
TEST(WorldDeathTest, AttachToALongDeadParentDuringAStepAndInfiniteTimeScale)
{
    auto run = []()
    {
        World w(testSetup(
            [](SystemList& l)
            {
                l.add(Phase::Gameplay, "attach",
                      [](World& w, double) { w.add<Parent>(Entity{1, 0}, Parent{Entity{2, 0}}); });
            }));
        w.create();
        Entity dead = w.create();
        w.destroy(dead);
        w.step();
    };
    EXPECT_DEATH(run(), "the parent is not alive");
    World w(testSetup());
    EXPECT_DEATH(w.setTimeScale(std::numeric_limits<double>::infinity()), "finite");
}
#endif

namespace
{
Entity g_reparentChild, g_newParent;
void sysReparent(World& w, double)
{
    if (w.stepIndex() != 0) return;
    // Remove + add inside a system (the documented way to change a Parent)
    w.remove<Parent>(g_reparentChild);
    w.add<Parent>(g_reparentChild, Parent{g_newParent});
}
void reparentSystems(SystemList& s)
{
    s.add(Phase::Gameplay, "reparent", sysReparent);
}
}  // namespace

// Regression: the record-time has/has-not checks ignored the commands recorded earlier in the phase, so remove + add
// of the same component asserted
TEST(World, ReparentingInsideASystem)
{
    World w(testSetup(reparentSystems));
    Entity oldParent = w.create();
    g_newParent      = w.create();
    g_reparentChild  = w.create();
    w.add<Parent>(g_reparentChild, Parent{oldParent});
    w.get<Transform>(g_newParent).setPosition(Vec3(5, 0, 0));
    w.step();
    EXPECT_EQ(w.read<Parent>(g_reparentChild).entity, g_newParent);
    EXPECT_EQ(w.read<Transform>(g_reparentChild).position(), Vec3(5, 0, 0));
    // Destroying the old parent leaves the child alone, the new one takes it along
    w.destroy(oldParent);
    EXPECT_TRUE(w.alive(g_reparentChild));
    w.destroy(g_newParent);
    EXPECT_FALSE(w.alive(g_reparentChild));
}

namespace
{
Entity g_twiceParent, g_twiceChild;
Phase g_secondTeleport               = Phase::Streaming;
bool g_twiceChildTeleportedInExtract = false;
double g_twiceChildXInExtract        = 0;
void sysTeleportInGameplay(World& w, double)
{
    if (w.stepIndex() == 1) w.get<Transform>(g_twiceParent).teleport(Vec3(10, 0, 0), Quat::Identity());
}
void sysTeleportAgain(World& w, double)
{
    if (w.stepIndex() == 1 && w.currentPhase() == g_secondTeleport)
        w.get<Transform>(g_twiceParent).teleport(Vec3(100, 0, 0), Quat::Identity());
}
void sysTwiceReader(World& w, double)
{
    g_twiceChildTeleportedInExtract = w.read<Transform>(g_twiceChild).wasTeleported();
    g_twiceChildXInExtract          = w.read<Transform>(g_twiceChild).position()(0);
}
void twiceSystems(SystemList& s)
{
    s.add(Phase::Gameplay, "first", sysTeleportInGameplay);
    s.add(Phase::Streaming, "second_streaming", sysTeleportAgain);
    s.add(Phase::Extract, "reader", sysTwiceReader);
    s.add(Phase::Extract, "second_extract", sysTeleportAgain);
}
}  // namespace

// Regression: a second teleport of a parent in the same step (after the child followed the first) never reached the
// child as a teleport
TEST(World, SecondTeleportOfAParentInOneStep)
{
    for (Phase second : {Phase::Streaming, Phase::Extract})
    {
        g_secondTeleport = second;
        World w(testSetup(twiceSystems));
        g_twiceParent = w.create();
        g_twiceChild  = w.create();
        w.add<Parent>(g_twiceChild, Parent{g_twiceParent});
        w.step(2);
        if (second == Phase::Streaming)
        {
            // Phase 7: the extract of the same step sees the child at the second target, flagged
            EXPECT_TRUE(g_twiceChildTeleportedInExtract);
            EXPECT_EQ(g_twiceChildXInExtract, 100.0);
        }
        else
        {
            // Phase 8 (after the reader): the child follows as a teleport in the next step
            w.step();
            EXPECT_TRUE(g_twiceChildTeleportedInExtract);
            EXPECT_EQ(g_twiceChildXInExtract, 100.0);
        }
    }
}

// S11 fills the registry directly: rebuildHierarchyState recounts the children, so destroy cascades again
TEST(World, RebuildHierarchyStateAfterDirectRegistryChanges)
{
    World w(testSetup());
    Entity p = w.create(), c = w.create();
    w.registry().add<Parent>(c, Parent{p});  // bypasses the World (like a loader)
    w.rebuildHierarchyState();
    w.destroy(p);
    EXPECT_FALSE(w.alive(c));
}

TEST(World, RemovedConsumersDoNotHoldTheLog)
{
    World w(testSetup());
    StructuralLog& log = w.structuralLog();
    int a              = log.addConsumer();
    for (int i = 0; i < 5; ++i) w.create();
    w.step();
    EXPECT_EQ(log.size(), 5u);
    log.removeConsumer(a);
    w.step();
    EXPECT_EQ(log.size(), 0u);
}

TEST(World, AdvanceIgnoresNanAndCapsInfinity)
{
    World w(testSetup());
    w.advance(std::nan(""));
    EXPECT_EQ(w.stepIndex(), 0u);
    w.advance(w.dt() * 1.5);
    EXPECT_EQ(w.stepIndex(), 1u);
    EXPECT_FALSE(std::isnan(w.alpha()));
    w.advance(std::numeric_limits<double>::infinity());
    EXPECT_EQ(w.stepIndex(), 5u);  // at most 4 steps, the rest dropped
    EXPECT_GE(w.alpha(), 0.0);
    EXPECT_LT(w.alpha(), 1.0);
    // Dropped: exactly the whole steps beyond the 4 (not one more for the kept fraction)
    World v(testSetup());
    v.advance(v.dt() * 6.5);
    EXPECT_EQ(v.stepIndex(), 4u);
    EXPECT_EQ(v.droppedSteps(), 1u);
}

namespace
{
std::atomic<int> g_nested{0};
void sysNestedViewOnWorkers(World& w, double)
{
    w.view<const Health>().parallelEach(
        [&](Entity, const Health&) { g_nested += int(w.viewAll<const Velocity>().count()); }, 1);
}
void nestedSystems(SystemList& s)
{
    s.add(Phase::Gameplay, "nested", sysNestedViewOnWorkers);
}
}  // namespace

TEST(World, NestedViewsInsideParallelEachOnWorkers)
{
    WorldSetup s    = testSetup(nestedSystems);
    s.workerThreads = 4;
    World w(s);
    for (int i = 0; i < 64; ++i)
    {
        Entity e = w.create();
        w.add<Health>(e);
        if (i < 3) w.add<Velocity>(e);
    }
    g_nested = 0;
    w.step();
    EXPECT_EQ(g_nested.load(), 64 * 3);
    EXPECT_FALSE(w.registry().iterating());
}

// ---------------------------------------------------------------------------------------------------------------------
// CodeRabbit regressions

// A throwing consumer leaves the record unread (delivered again); a consumer removed in its callback stops
#ifdef SAIGA_ASSERTS
TEST(WorldDeathTest, ReentrantConsumeOfTheSameConsumer)
{
    World w(testSetup());
    StructuralLog& log = w.structuralLog();
    int c              = log.addConsumer();
    w.create();
    EXPECT_DEATH(
        log.consume(c, [&](const StructuralLog::Record&) { log.consume(c, [](const StructuralLog::Record&) {}); }),
        "reentrant");
}
#endif

TEST(World, NestedChangesDoNotUseTheOuterReservation)
{
    // prepare counts reservations: an outer prepare(1), then a nested prepare(1) + append, then the outer append
    StructuralLog log;
    {
        auto outer = log.prepare(1);
        {
            auto nested = log.prepare(1);
            nested.append(Entity{1, 0}, 0, true);
        }
        outer.append(Entity{2, 0}, 0, true);
    }
    EXPECT_EQ(log.size(), 2u);
    EXPECT_EQ(log.reservedCount(), 0u);
}

// Regression: prepare counted the reservation before the reserve could throw, and a mutation that failed after
// prepare left its reservation behind; each failure made later prepares reserve more
TEST(World, FailedPrepareOrMutationGivesTheReservationBack)
{
    StructuralLog log;
    // The reserve throws (more than max_size): nothing is reserved
    EXPECT_THROW(log.prepare(std::numeric_limits<size_t>::max() / 2), std::length_error);
    EXPECT_EQ(log.reservedCount(), 0u);
    // The mutation fails after prepare: the unused slot is given back
    auto failingMutation = [&]()
    {
        auto record = log.prepare(1);
        throw std::runtime_error("mutation failed");
    };
    EXPECT_THROW(failingMutation(), std::runtime_error);
    EXPECT_EQ(log.reservedCount(), 0u);
    // Partly used: only the slots not appended are given back
    {
        auto records = log.prepare(3);
        records.append(Entity{1, 0}, 0, false);
        EXPECT_EQ(log.reservedCount(), 2u);
    }
    EXPECT_EQ(log.reservedCount(), 0u);
    EXPECT_EQ(log.size(), 1u);
}

TEST(World, StructuralLogConsumerThrowsOrIsRemoved)
{
    World w(testSetup());
    StructuralLog& log = w.structuralLog();
    int c              = log.addConsumer();
    w.create();
    w.create();
    int calls = 0;
    EXPECT_THROW(log.consume(c,
                             [&](const StructuralLog::Record&)
                             {
                                 if (++calls == 2) throw std::runtime_error("consumer failed");
                             }),
                 std::runtime_error);
    EXPECT_EQ(log.unread(c), 1u);  // the second record was not consumed
    int seen = 0;
    log.consume(c, [&](const StructuralLog::Record&) { seen++; });
    EXPECT_EQ(seen, 1);

    int d = log.addConsumer();
    for (int i = 0; i < 5; ++i) w.create();
    int delivered = 0;
    log.consume(d,
                [&](const StructuralLog::Record&)
                {
                    delivered++;
                    log.removeConsumer(d);
                    log.truncate();
                });
    EXPECT_EQ(delivered, 1);
}

// Regression: depthOf stopped at depth 1000, so a valid deeper chain could update a child before its parent
TEST(World, DeepParentChainKeepsParentsBeforeChildren)
{
    World w(testSetup());
    std::vector<Entity> chain{w.create()};
    // Deeper than the old cutoff of 1000; created so the deepest entities have the lowest indices
    std::vector<Entity> nodes;
    for (int i = 0; i < 1500; ++i) nodes.push_back(w.create());
    for (int i = 1499; i >= 0; --i)
    {
        w.add<Parent>(nodes[size_t(i)], Parent{chain.back()});
        chain.push_back(nodes[size_t(i)]);
    }
    w.get<Transform>(chain.front()).setPosition(Vec3(3, 0, 0));
    w.step();
    EXPECT_EQ(w.read<Transform>(chain.back()).position(), Vec3(3, 0, 0));  // the leaf followed in the same step
}

TEST(World, RebuildHierarchyStateRejectsStaleParents)
{
    World w(testSetup());
    Entity p = w.create(), c = w.create(), dead = w.create();
    w.destroy(dead);
    w.add<Parent>(c, Parent{p});
    Entity x = w.create();
    w.registry().add<Parent>(x, Parent{dead});  // a broken direct fill
    EXPECT_FALSE(w.rebuildHierarchyState());
    w.registry().remove<Parent>(x);
    EXPECT_TRUE(w.rebuildHierarchyState());
    // A cycle from a direct fill is rejected too (A -> B -> A); a valid chain to a root is accepted
    Entity ca = w.create(), cb = w.create();
    w.registry().add<Parent>(ca, Parent{cb});
    EXPECT_TRUE(w.rebuildHierarchyState());
    w.registry().add<Parent>(cb, Parent{ca});
    EXPECT_FALSE(w.rebuildHierarchyState());
    w.registry().remove<Parent>(cb);
    EXPECT_TRUE(w.rebuildHierarchyState());
    w.destroy(p);
    EXPECT_FALSE(w.alive(c));
}

#ifdef SAIGA_ASSERTS
namespace
{
// Same ID, name and size as DamageEvent, but another C++ type (never registered)
struct DamageEventLookalike
{
    static constexpr const char* name = "DamageEvent";
    static constexpr EventId id       = EventId(TestEvent::Damage);
    static constexpr uint32_t version = 1;
    Entity target;
    double amount = 0;
    template <class V>
    void visit(V& v)
    {
        v("target", target);
        v("amount", amount);
    }
};
}  // namespace

// Regression: the queue lookup checked ID, name and size only, so a same-sized lookalike was downcast to the wrong type
TEST(WorldDeathTest, EventLookalikeWithTheSameIdNameAndSize)
{
    static_assert(sizeof(DamageEventLookalike) == sizeof(DamageEvent), "the test needs the same size");
    World w(testSetup());
    EXPECT_DEATH(w.events.read<DamageEventLookalike>(), "another C\\+\\+ type");
}

// Defined in engine_world_second_tu.cpp: reads the DamageEvent of that translation unit's unnamed namespace
size_t readDamageEventOfTheSecondTranslationUnit(World& w);

// Regression: the check compared typeid names, which GCC gives equal to types of the same name in the unnamed
// namespaces of two translation units; the other DamageEvent was accepted and downcast to the wrong type
TEST(WorldDeathTest, EventOfTheSameNameInAnotherTranslationUnit)
{
    World w(testSetup());
    EXPECT_DEATH(readDamageEventOfTheSecondTranslationUnit(w), "another C\\+\\+ type");
}
#endif

// Without the SAIGA_ASSERTS guard: the check uses SAIGA_EXIT_ERROR, active in every build
namespace
{
struct SameIdAsDamage
{
    static constexpr const char* name = "SameIdAsDamage";
    static constexpr EventId id       = EventId(TestEvent::Damage);
    static constexpr uint32_t version = 1;
    template <class V>
    void visit(V&)
    {
    }
};
}  // namespace

TEST(WorldDeathTest, DuplicateEventIdStopsInEveryBuild)
{
    World w(testSetup());
    EventRegistry r(w.events);
    EXPECT_DEATH(r.add<SameIdAsDamage>(), "already used by DamageEvent");
}
