/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Game objects, classes, ticking, events, actions, the kinematic mover, world state and Persistent
// (docs/engine_design.md §7.8, §8.4, §9.2, §9.3, §9.6; stage S10)

#include "saiga/engine/components/CharacterIntent.h"
#include "saiga/engine/components/Health.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/PlayerInput.h"
#include "saiga/engine/gameplay/GameObject.h"
#include "saiga/engine/world/World.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Saiga;
using namespace Saiga::Engine;

namespace
{
std::vector<std::string> g_trace;

enum class TestClass : uint16_t
{
    Probe = kFirstGameClass,
    Spawner,
    Slow,
    Count
};

enum class TestComponent : uint8_t
{
    Flags = kFirstGameComponent,
    Zone,
    Count
};

// A world-state component
struct Flags
{
    static constexpr const char* name = "Flags";
    static constexpr ComponentId id   = ComponentId(TestComponent::Flags);
    static constexpr uint32_t version = 1;
    bool bossKilled                   = false;
    int32_t counter                   = 0;
    template <class V>
    void visit(V& v)
    {
        v("bossKilled", bossKilled);
        v("counter", counter);
    }
};

// A space-state component
struct Zone
{
    static constexpr const char* name = "Zone";
    static constexpr ComponentId id   = ComponentId(TestComponent::Zone);
    static constexpr uint32_t version = 1;
    double danger                     = 0;
    template <class V>
    void visit(V& v)
    {
        v("danger", danger);
    }
};

// Records its hooks; counts updates in its Health (state in components, R-STATE-1)
class Probe final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(TestClass::Probe);
    static constexpr const char* name = "Probe";

    void configure() override { g_trace.push_back("configure " + std::to_string(entity().index)); }
    void onSpawn() override
    {
        g_trace.push_back("onSpawn " + std::to_string(entity().index) + (has<Health>() ? " health" : ""));
    }
    void update(double dt) override
    {
        if (has<Health>())
        {
            write<Health>().hp += 1;
            write<Health>().maxHp = dt;  // the last dtSinceLastTick
        }
    }
    void onDamage(const DamageEvent& e) override
    {
        g_trace.push_back("damage " + std::to_string(entity().index) + " " + std::to_string(int(e.amount)));
    }
    void onInteract(const InteractEvent& e) override
    {
        g_trace.push_back("interact " + std::to_string(entity().index) + " " + std::to_string(e.action));
    }
    void onContact(const ContactEvent&) override { g_trace.push_back("contact " + std::to_string(entity().index)); }
    void onDestroy() override
    {
        g_trace.push_back("destroy " + std::to_string(entity().index));
        // A destroy of itself from the hook does not call it again
        destroy();
    }
};

// Spawns a Probe in its first update and destroys itself in its second
class Spawner final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(TestClass::Spawner);
    static constexpr const char* name = "Spawner";

    void update(double) override
    {
        Health& h = write<Health>();
        h.hp += 1;
        if (h.hp == 1)
        {
            Entity child = spawn<Probe>(Vec3(1, 0, 0), Health{0, 0});
            g_trace.push_back("spawned " + std::to_string(child.index) +
                              (world().alive(child) ? " alive" : " reserved"));
        }
        if (h.hp == 2) destroy();
    }
    void onDestroy() override
    {
        // Loot on death: spawning from onDestroy works
        spawn<Probe>(transform().position());
    }
};

// A class with its own policy: ticks every step up to ring 3
class Slow final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(TestClass::Slow);
    static constexpr const char* name = "Slow";
    const TickPolicy* tickPolicy() const override
    {
        static const TickPolicy p{32, 3, 4, 1.0, 2.0};
        return &p;
    }
};

void testClasses(ClassList& c)
{
    c.add<Probe>();
    c.add<Spawner>();
    c.add<Slow>();
}

void testComponents(Registry& r)
{
    r.registerGlobal<Flags>(Replicate::Host);
    r.registerSpaceState<Zone>(Replicate::Host);
}

WorldSetup testSetup(void (*systems)(SystemList&) = nullptr)
{
    WorldSetup s;
    s.components = testComponents;
    s.classes    = testClasses;
    s.systems    = systems;
    return s;
}

Entity spawnPlayer(World& w, const Vec3& p, uint8_t peer = 0)
{
    Entity e = w.create();
    w.get<Transform>(e).setPosition(p);
    Player player;
    player.peer        = peer;
    player.characterId = 7;
    w.add<Player>(e, player);
    w.add<PlayerInput>(e);
    w.add<CharacterIntent>(e);
    return e;
}
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// Lifecycle

TEST(GameObject, SpawnBetweenStepsRunsTheHooksWithTheInitialComponents)
{
    g_trace.clear();
    World w(testSetup());
    Entity e = w.spawn<Probe>(Vec3(3, 0, 4), Health{50, 100});
    ASSERT_TRUE(w.alive(e));
    EXPECT_EQ(w.read<Transform>(e).position(), Vec3(3, 0, 4));
    EXPECT_EQ(w.read<Behaviour>(e).classId, Probe::id);
    ASSERT_TRUE(w.read<Behaviour>(e).object);
    EXPECT_EQ(w.read<Behaviour>(e).object->entity(), e);
    EXPECT_EQ(w.read<Health>(e).hp, 50);
    const std::string idx = std::to_string(e.index);
    EXPECT_EQ(g_trace, (std::vector<std::string>{"configure " + idx, "onSpawn " + idx + " health"}));
}

TEST(GameObject, SpawnDuringAStepIsReservedUntilTheFlushAndDestroyFromUpdate)
{
    g_trace.clear();
    World w(testSetup());
    spawnPlayer(w, Vec3(0, 0, 0));
    Entity s = w.spawn<Spawner>(Vec3(0, 0, 0), Health{0, 0});
    w.step();
    // The spawn in update reserved the entity; at the end of phase 3 it is a Probe
    ASSERT_GE(g_trace.size(), 3u);
    EXPECT_EQ(g_trace[0].substr(g_trace[0].size() - 8), "reserved");
    EXPECT_EQ(g_trace[1].rfind("configure", 0), 0u);
    EXPECT_EQ(g_trace[2].rfind("onSpawn", 0), 0u);
    w.step();
    // Destroyed itself in the second update: onDestroy spawned the loot
    EXPECT_FALSE(w.alive(s));
    int probes = 0;
    w.viewAll<const Behaviour>().each([&](Entity, const Behaviour& b) { probes += b.classId == Probe::id; });
    EXPECT_EQ(probes, 2);
}

TEST(GameObject, OnDestroyRunsChildrenFirstOnceWithAllComponents)
{
    g_trace.clear();
    World w(testSetup());
    Entity parent = w.spawn<Probe>(Vec3(0, 0, 0));
    Entity child  = w.spawn<Probe>(Vec3(1, 0, 0));
    w.add<Parent>(child, Parent{parent});
    g_trace.clear();
    w.destroy(parent);
    EXPECT_EQ(g_trace, (std::vector<std::string>{"destroy " + std::to_string(child.index),
                                                 "destroy " + std::to_string(parent.index)}));
    EXPECT_FALSE(w.alive(parent));
    EXPECT_FALSE(w.alive(child));
}

#ifdef SAIGA_ASSERTS
namespace
{
class Unregistered final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(70);
    static constexpr const char* name = "Unregistered";
};
class Duplicate final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(TestClass::Probe);
    static constexpr const char* name = "Duplicate";
};
class Gap final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(66);
    static constexpr const char* name = "Gap";
};
}  // namespace

TEST(GameObjectDeathTest, ClassRegistration)
{
    World w(testSetup());
    EXPECT_DEATH(w.spawn<Unregistered>(Vec3::Zero()), "class Unregistered is not registered");
    EXPECT_DEATH(
        {
            ClassList c;
            c.add<Probe>();
            c.add<Duplicate>();
        },
        "already used by Probe");
    EXPECT_DEATH(
        {
            ClassList c;
            c.add<Gap>();
            c.validateRegistration();
        },
        "IDs must be contiguous");
}
#endif

// ---------------------------------------------------------------------------------------------------------------------
// Ticking

namespace
{
// Updates of the probe over n steps (Health::hp counts them)
double updatesOver(World& w, Entity e, int n)
{
    const double before = w.read<Health>(e).hp;
    w.step(n);
    return w.read<Health>(e).hp - before;
}
}  // namespace

TEST(Ticking, GridCellRingsAroundThePlayer)
{
    World w(testSetup());
    spawnPlayer(w, Vec3(5, 0, 5));                                        // cell (0, 0)
    Entity near0 = w.spawn<Probe>(Vec3(20, 0, 20), Health{0, 0});         // ring 0
    Entity near1 = w.spawn<Probe>(Vec3(-20, 0, 40), Health{0, 0});        // ring 1
    Entity mid   = w.spawn<Probe>(Vec3(6 * 32 + 1, 0, 0), Health{0, 0});  // ring 6
    Entity far   = w.spawn<Probe>(Vec3(7 * 32 + 1, 0, 0), Health{0, 0});  // ring 7
    EXPECT_EQ(w.read<Behaviour>(near0).tickInterval, 0);
    EXPECT_EQ(w.read<Behaviour>(near1).tickInterval, 0);
    EXPECT_EQ(w.read<Behaviour>(mid).tickInterval, 0.1);
    EXPECT_EQ(w.read<Behaviour>(far).tickInterval, 0.5);
    w.step();  // the staggered starts pass
    EXPECT_EQ(updatesOver(w, near0, 60), 60);
    EXPECT_EQ(updatesOver(w, near1, 60), 60);
    EXPECT_EQ(updatesOver(w, mid, 60), 10);
    EXPECT_EQ(updatesOver(w, far, 60), 2);
    // dtSinceLastTick is the time since the last update
    EXPECT_NEAR(w.read<Health>(far).maxHp, 0.5, 1e-9);
}

TEST(Ticking, AnObjectWithoutAPlayerOrInAnotherCellTicksFar)
{
    World w(testSetup());
    Entity e = w.spawn<Probe>(Vec3(0, 0, 0), Health{0, 0});
    EXPECT_EQ(w.read<Behaviour>(e).tickInterval, 0.5);
    // A player arrives: every object is re-evaluated in phase 6
    spawnPlayer(w, Vec3(0, 0, 0));
    w.step();
    EXPECT_EQ(w.read<Behaviour>(e).tickInterval, 0);
}

TEST(Ticking, APlayerCrossingACellBorderReTiersAndObjectsMovingReTier)
{
    World w(testSetup());
    Entity player = spawnPlayer(w, Vec3(0, 0, 0));
    Entity e      = w.spawn<Probe>(Vec3(2 * 32 + 1, 0, 0), Health{0, 0});  // ring 2: mid
    EXPECT_EQ(w.read<Behaviour>(e).tickInterval, 0.1);
    w.get<Transform>(player).setPosition(Vec3(33, 0, 0));  // cell 1: ring 1
    w.step();
    EXPECT_EQ(w.read<Behaviour>(e).tickInterval, 0);
    // The object moves away: re-evaluated from the changed list
    w.get<Transform>(e).setPosition(Vec3(1000, 0, 0));
    w.step();
    EXPECT_EQ(w.read<Behaviour>(e).tickInterval, 0.5);
}

TEST(Ticking, TheNearestOfTwoPlayersClassPolicyAndAlwaysTick)
{
    World w(testSetup());
    spawnPlayer(w, Vec3(0, 0, 0), 0);
    spawnPlayer(w, Vec3(10 * 32, 0, 0), 1);
    Entity between = w.spawn<Probe>(Vec3(9 * 32 + 1, 0, 0), Health{0, 0});  // ring 1 to the second player
    EXPECT_EQ(w.read<Behaviour>(between).tickInterval, 0);
    // Slow: every step up to ring 3, then 1 s up to ring 4, then 2 s
    Entity s3 = w.spawn<Slow>(Vec3(0, 0, 3 * 32 + 1));
    Entity s4 = w.spawn<Slow>(Vec3(0, 0, 4 * 32 + 1));
    Entity s9 = w.spawn<Slow>(Vec3(0, 0, 9 * 32 + 1));
    EXPECT_EQ(w.read<Behaviour>(s3).tickInterval, 0);
    EXPECT_EQ(w.read<Behaviour>(s4).tickInterval, 1.0);
    EXPECT_EQ(w.read<Behaviour>(s9).tickInterval, 2.0);
    // alwaysTickEveryStep overrides the rings
    Entity far                                = w.spawn<Probe>(Vec3(0, 0, 5000), Health{0, 0});
    w.get<Behaviour>(far).alwaysTickEveryStep = true;
    EXPECT_EQ(updatesOver(w, far, 10), 10);
}

TEST(Ticking, SameHistorySameTicks)
{
    auto run = []()
    {
        World w(testSetup());
        Entity p = spawnPlayer(w, Vec3(0, 0, 0));
        std::vector<Entity> objects;
        for (int i = 0; i < 200; ++i)
            objects.push_back(w.spawn<Probe>(Vec3(double(i % 20) * 25, 0, double(i / 20) * 25), Health{0, 0}));
        std::vector<double> counts;
        for (int s = 0; s < 120; ++s)
        {
            w.get<Transform>(p).setPosition(Vec3(double(s) * 2, 0, 0));
            w.step();
        }
        for (Entity e : objects) counts.push_back(w.read<Health>(e).hp);
        return std::make_pair(counts, w.hash());
    };
    auto a = run();
    auto b = run();
    EXPECT_EQ(a.first, b.first);
    EXPECT_EQ(a.second, b.second);
}

// ---------------------------------------------------------------------------------------------------------------------
// Events and actions

TEST(GameObject, EventsReachTheHandlersInTheNextStep)
{
    g_trace.clear();
    World w(testSetup());
    Entity a    = w.spawn<Probe>(Vec3(0, 0, 0));
    Entity b    = w.spawn<Probe>(Vec3(1, 0, 0));
    Entity dead = w.spawn<Probe>(Vec3(2, 0, 0));
    g_trace.clear();
    w.events.emit(DamageEvent{b, a, 5});
    w.events.emit(DamageEvent{a, b, 7});
    w.events.emit(DamageEvent{dead, a, 9});
    w.events.emit(InteractEvent{a, b, 3});
    w.events.emit(ContactEvent{a, b, Vec3::Zero(), Vec3::UnitY()});
    w.destroy(dead);
    g_trace.clear();
    w.step();
    const std::string ia = std::to_string(a.index), ib = std::to_string(b.index);
    EXPECT_EQ(g_trace, (std::vector<std::string>{"damage " + ib + " 5", "damage " + ia + " 7", "interact " + ia + " 3",
                                                 "contact " + ia, "contact " + ib}));
    g_trace.clear();
    w.step();
    EXPECT_TRUE(g_trace.empty());
}

namespace
{
std::vector<std::pair<uint32_t, uint32_t>> g_actions;  // (player index, action)
bool g_throwInGameplay = false;
void sysReadActions(World& w, double)
{
    for (const ActionEvent& a : w.events.read<ActionEvent>()) g_actions.push_back({a.player.index, a.action});
    if (g_throwInGameplay)
    {
        g_throwInGameplay = false;
        throw std::runtime_error("aborted");
    }
}
void actionSystems(SystemList& s)
{
    s.add(Phase::Gameplay, "read_actions", sysReadActions);
}
}  // namespace

TEST(GameObject, ActionsArriveInPeerSequenceOrderExactlyOnce)
{
    World w(testSetup(actionSystems));
    Entity p1 = spawnPlayer(w, Vec3::Zero(), 1);
    Entity p0 = spawnPlayer(w, Vec3::Zero(), 0);
    w.queueAction(PlayerAction{p1, 10});
    w.queueAction(PlayerAction{p0, 20});
    w.queueAction(PlayerAction{p1, 11});
    w.queueAction(PlayerAction{p0, 21});
    g_actions.clear();
    // An aborted step keeps them (they are events of the step)
    g_throwInGameplay = true;
    EXPECT_THROW(w.step(), std::runtime_error);
    g_actions.clear();
    w.step();
    EXPECT_EQ(g_actions, (std::vector<std::pair<uint32_t, uint32_t>>{
                             {p0.index, 20}, {p0.index, 21}, {p1.index, 10}, {p1.index, 11}}));
    g_actions.clear();
    w.step();
    EXPECT_TRUE(g_actions.empty());
}

// ---------------------------------------------------------------------------------------------------------------------
// Player control and the kinematic mover

TEST(GameObject, PlayerControlAndCharacterMove)
{
    World w(testSetup());
    Entity p                   = spawnPlayer(w, Vec3(0, 3, 0));
    w.get<PlayerInput>(p).move = Vec3(2, 0, 0);  // longer than 1: shortened
    w.step(60);
    EXPECT_NEAR(w.read<Transform>(p).position().x(), 5.0, 1e-9);
    EXPECT_EQ(w.read<Transform>(p).position().y(), 0);  // flat ground before S20
    EXPECT_EQ(w.read<CharacterIntent>(p).desiredVelocity, Vec3(5, 0, 0));
    // Idle: the transform is not written
    w.get<PlayerInput>(p).move = Vec3::Zero();
    w.step(2);
    w.step();
    EXPECT_FALSE(w.read<Transform>(p).changedInLastStep());
    // ownedBy: all entities before S16
    int owned = 0;
    w.ownedBy<const Player>(w.localPeer()).each([&](Entity, const Player&) { owned++; });
    EXPECT_EQ(owned, 1);
    EXPECT_TRUE(w.isHost());
}

// ---------------------------------------------------------------------------------------------------------------------
// World state

TEST(WorldState, GlobalAndSpaceStateEntities)
{
    World w(testSetup());
    ASSERT_TRUE(w.alive(w.worldStateEntity()));
    ASSERT_TRUE(w.alive(w.spaceStateEntity(0)));
    w.global<Flags>().bossKilled = true;
    w.global<Flags>().counter++;
    EXPECT_TRUE(w.readGlobal<Flags>().bossKilled);
    w.spaceState<Zone>(0).danger = 3;
    EXPECT_EQ(w.readSpaceState<Zone>(0).danger, 3);
    // Not in the grid, visible in views
    w.step();
    EXPECT_TRUE(w.queryRadius(0, Vec3::Zero(), 1).empty());
    int flags = 0;
    w.viewAll<const Flags>().each([&](Entity, const Flags&) { flags++; });
    EXPECT_EQ(flags, 1);
    // A world without world-state components has no such entities
    World plain(WorldSetup{});
    EXPECT_FALSE(plain.worldStateEntity().valid());
    EXPECT_EQ(plain.registry().aliveCount(), 0u);
}

#ifdef SAIGA_ASSERTS
TEST(WorldStateDeathTest, NotDestroyableAndOnlyRegisteredComponents)
{
    World w(testSetup());
    EXPECT_DEATH(w.destroy(w.worldStateEntity()), "cannot be destroyed");
    EXPECT_DEATH(w.global<Health>(), "not registered as a global");
    EXPECT_DEATH(w.spaceState<Zone>(1), "space 1 is not loaded");
}
#endif

// ---------------------------------------------------------------------------------------------------------------------
// Persistent

namespace
{
Entity g_target;
PersistentId g_madeInStep;
void sysMakePersistent(World& w, double)
{
    if (w.alive(g_target) && !g_madeInStep.valid()) g_madeInStep = w.makePersistent(g_target);
}
void persistentSystems(SystemList& s)
{
    s.add(Phase::Gameplay, "persistent", sysMakePersistent);
}
}  // namespace

TEST(Persistent, MakeResolveDestroyAndReuse)
{
    World w(testSetup(persistentSystems));
    Entity a              = w.create();
    const PersistentId id = w.makePersistent(a);
    EXPECT_TRUE(id.valid());
    EXPECT_EQ(w.makePersistent(a), id);  // idempotent
    EXPECT_EQ(w.resolve(id), a);
    w.destroy(a);
    EXPECT_FALSE(w.resolve(id).valid());
    // The reused slot is not the persistent entity
    Entity b = w.create();
    EXPECT_EQ(b.index, a.index);
    EXPECT_FALSE(w.resolve(id).valid());
    // During a step: the ID at once, resolvable after the flush
    g_target     = b;
    g_madeInStep = PersistentId{};
    w.step();
    ASSERT_TRUE(g_madeInStep.valid());
    EXPECT_NE(g_madeInStep, id);
    EXPECT_EQ(w.resolve(g_madeInStep), b);
}

// ---------------------------------------------------------------------------------------------------------------------
// Review regressions (S10 round 1)

namespace
{
enum class ReviewClass : uint16_t
{
    Persister = kFirstGameClass + 3,
    Adopter,
    SelfDestroyer,
    Cleaner,
    Thrower,
    Count
};

Entity g_spawnedPersistent;
PersistentId g_pidA, g_pidB, g_pidSpawned;
// Calls makePersistent twice on itself in one update and on an entity it spawns (reserved)
class Persister final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(ReviewClass::Persister);
    static constexpr const char* name = "Persister";
    void update(double) override
    {
        if (g_pidA.valid()) return;
        g_pidA              = world().makePersistent(entity());
        g_pidB              = world().makePersistent(entity());
        g_spawnedPersistent = spawn<Probe>(Vec3::Zero());
        g_pidSpawned        = world().makePersistent(g_spawnedPersistent);
    }
};

Entity g_adopted;
// Attaches a new entity to itself in onDestroy (it must be destroyed with it)
class Adopter final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(ReviewClass::Adopter);
    static constexpr const char* name = "Adopter";
    void onDestroy() override
    {
        g_adopted = world().create();
        world().add<Parent>(g_adopted, Parent{entity()});
    }
};

// Destroys its own entity in onSpawn and keeps using its members afterwards (the object must stay valid)
class SelfDestroyer final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(ReviewClass::SelfDestroyer);
    static constexpr const char* name = "SelfDestroyer";
    void onSpawn() override
    {
        const Entity self = entity();
        destroy();
        g_trace.push_back("after destroy " + std::to_string(entity().index == self.index));
    }
};

Entity g_effect;
// Destroys itself in update and spawns an effect afterwards; its onDestroy (at the flush) cleans the effect up
class Cleaner final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(ReviewClass::Cleaner);
    static constexpr const char* name = "Cleaner";
    void update(double) override
    {
        if (g_effect.valid()) return;
        destroy();
        g_effect = world().create();
    }
    void onDestroy() override { world().destroy(g_effect); }
};

bool g_throwInDestroy = true;
class Thrower final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(ReviewClass::Thrower);
    static constexpr const char* name = "Thrower";
    void onDestroy() override
    {
        g_trace.push_back("thrower destroy");
        if (g_throwInDestroy) throw std::runtime_error("onDestroy failed");
    }
};

void reviewClasses(ClassList& c)
{
    testClasses(c);
    c.add<Persister>();
    c.add<Adopter>();
    c.add<SelfDestroyer>();
    c.add<Cleaner>();
    c.add<Thrower>();
}
WorldSetup reviewSetup()
{
    WorldSetup s = testSetup();
    s.classes    = reviewClasses;
    return s;
}
}  // namespace

// Regression: a second makePersistent in the same phase assigned a second ID (the flush then asserted), and a reserved
// (just spawned) entity was refused
TEST(Persistent, IdempotentWithinAStepAndForSpawnedEntities)
{
    World w(reviewSetup());
    spawnPlayer(w, Vec3::Zero());
    w.spawn<Persister>(Vec3::Zero());
    g_pidA = g_pidB = g_pidSpawned = PersistentId{};
    w.step();
    ASSERT_TRUE(g_pidA.valid());
    EXPECT_EQ(g_pidA, g_pidB);
    ASSERT_TRUE(g_pidSpawned.valid());
    EXPECT_EQ(w.resolve(g_pidSpawned), g_spawnedPersistent);
}

// Regression: an entity attached to a parent in its onDestroy survived with a Parent naming a dead entity (a later
// destroy asserted on the child count)
TEST(GameObject, AnEntityAttachedInOnDestroyIsDestroyedWithTheParent)
{
    World w(reviewSetup());
    Entity a = w.spawn<Adopter>(Vec3::Zero());
    w.destroy(a);
    EXPECT_FALSE(w.alive(a));
    EXPECT_FALSE(w.alive(g_adopted));
    w.step();
}

// Regression: a hook that destroyed its own entity freed the running object (use after free under ASan)
TEST(GameObject, SelfDestroyInAHookKeepsTheObjectAliveUntilItReturns)
{
    g_trace.clear();
    World w(reviewSetup());
    Entity e = w.spawn<SelfDestroyer>(Vec3::Zero());
    EXPECT_FALSE(w.alive(e));
    ASSERT_FALSE(g_trace.empty());
    EXPECT_EQ(g_trace.back(), "after destroy 1");
}

// Regression: during a flush a hook destroyed an entity that was reserved later in the same phase (asserted: not
// alive); it follows the entity's activation now
TEST(GameObject, HooksDuringAFlushReachEntitiesReservedLaterInThePhase)
{
    World w(reviewSetup());
    spawnPlayer(w, Vec3::Zero());
    Entity c = w.spawn<Cleaner>(Vec3::Zero());
    g_effect = Entity{};
    w.step();
    EXPECT_FALSE(w.alive(c));
    ASSERT_TRUE(g_effect.valid());
    EXPECT_FALSE(w.alive(g_effect));
}

// Regression: the stagger of the first tick was passed to update() as elapsed time
TEST(Ticking, TheFirstUpdateGetsTheRealElapsedTime)
{
    World w(testSetup());
    std::vector<Entity> objects;
    for (int i = 0; i < 200; ++i) objects.push_back(w.spawn<Probe>(Vec3(5000, 0, 0), Health{0, 0}));  // far: 0.5 s
    for (int s = 0; s < 40; ++s)
    {
        w.step();
        const double elapsed = double(s + 1) / 60.0;
        for (Entity e : objects)
            if (w.read<Health>(e).hp > 0) EXPECT_LE(w.read<Health>(e).maxHp, elapsed + 1e-9);
    }
}

// An exception in onDestroy aborts the destroy: nothing is destroyed and the next destroy calls the hook again
TEST(GameObject, AThrowingOnDestroyLeavesTheEntityDestroyableAgain)
{
    g_trace.clear();
    World w(reviewSetup());
    Entity t         = w.spawn<Thrower>(Vec3::Zero());
    g_throwInDestroy = true;
    EXPECT_THROW(w.destroy(t), std::runtime_error);
    EXPECT_TRUE(w.alive(t));
    g_throwInDestroy = false;
    w.destroy(t);
    EXPECT_FALSE(w.alive(t));
    EXPECT_EQ(std::count(g_trace.begin(), g_trace.end(), "thrower destroy"), 2);
}

#ifdef SAIGA_ASSERTS
// Regression: world-state entities could be destroyed through a parent's cascade and their components removed
TEST(WorldStateDeathTest, NoParentAndNoRemovalOfWorldStateComponents)
{
    World w(testSetup());
    Entity p = w.create();
    EXPECT_DEATH(w.add<Parent>(w.worldStateEntity(), Parent{p}), "cannot have a parent");
    EXPECT_DEATH(w.remove<Flags>(w.worldStateEntity()), "world-state components cannot be removed");
}
#endif
