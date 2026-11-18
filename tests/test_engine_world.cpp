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
#include <cmath>
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
    w.events.emit(DamageEvent{Entity{}, 1});  // between steps 0 and 1: belongs to step 0, readable in step 1
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
        World w(s);
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
