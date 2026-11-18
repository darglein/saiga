/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The ECS core (docs/engine_design.md §7.1-§7.5; stage S07): entities, registration, storage, views, visit.

#include "saiga/core/math/ChunkId.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/ecs/EntityLog.h"
#include "saiga/engine/ecs/FieldList.h"
#include "saiga/engine/ecs/Registry.h"
#include "saiga/engine/ecs/VisitBinary.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Saiga;
using namespace Saiga::Engine;

namespace
{
// Test components with game IDs (64..)
enum class TestComponent : uint8_t
{
    Health = kFirstGameComponent,
    Velocity,
    Tag,
    Inventory,
    Count
};

struct Health
{
    static constexpr const char* name = "Health";
    static constexpr ComponentId id   = ComponentId(TestComponent::Health);
    static constexpr uint32_t version = 1;
    double hp = 100, maxHp = 100;
    template <class V>
    void visit(V& v)
    {
        v("hp", hp);
        v("max_hp", maxHp);
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
        vis("v", v, Quantize{0.01});
    }
};

struct Tag
{
    static constexpr const char* name = "Tag";
    static constexpr ComponentId id   = ComponentId(TestComponent::Tag);
    static constexpr uint32_t version = 1;
    template <class V>
    void visit(V&)
    {
    }
};

enum class ItemKind : uint8_t
{
    Wood,
    Stone,
    Count
};
constexpr const char* ItemKindNames[] = {"wood", "stone"};

struct ItemStack
{
    static constexpr const char* name = "ItemStack";
    static constexpr uint32_t version = 1;
    ItemKind kind                     = ItemKind::Wood;
    int32_t count                     = 0;
    template <class V>
    void visit(V& v)
    {
        v("kind", kind, EnumNames{ItemKindNames});
        v("count", count);
    }
    bool operator==(const ItemStack& o) const { return kind == o.kind && count == o.count; }
};

// Every supported field type
struct Inventory
{
    static constexpr const char* name = "Inventory";
    static constexpr ComponentId id   = ComponentId(TestComponent::Inventory);
    static constexpr uint32_t version = 2;
    bool open                         = false;
    int8_t i8                         = -3;
    uint16_t u16                      = 65000;
    int64_t i64                       = -1234567890123ll;
    uint64_t u64                      = 0xFEDCBA9876543210ull;
    double d                          = -0.0;
    std::string label;
    Entity owner;
    PersistentId pid;
    ChunkId chunk;
    Vec2 v2                      = Vec2::Zero();
    Vec4 v4                      = Vec4::Zero();
    Quat q                       = Quat::Identity();
    Mat3 m3                      = Mat3::Identity();
    Mat4 m4                      = Mat4::Identity();
    ItemKind selected            = ItemKind::Wood;
    std::array<int32_t, 3> slots = {0, 0, 0};
    std::vector<ItemStack> items;
    std::vector<std::vector<double>> nested;
    template <class V>
    void visit(V& v)
    {
        v("open", open);
        v("i8", i8);
        v("u16", u16);
        v("i64", i64);
        v("u64", u64);
        v("d", d);
        v("label", label);
        v("owner", owner);
        v("pid", pid);
        v("chunk", chunk);
        v("v2", v2);
        v("v4", v4);
        v("q", q);
        v("m3", m3);
        v("m4", m4);
        v("selected", selected, EnumNames{ItemKindNames});
        v("slots", slots);
        v("items", items);
        v("nested", nested);
    }
};

// A render component may use float types
struct RenderThing
{
    static constexpr const char* name = "RenderThing";
    static constexpr ComponentId id   = 100;
    static constexpr uint32_t version = 1;
    static constexpr bool allowFloat  = true;
    vec3 color                        = vec3(1, 0.5f, 0.25f);
    float alpha                       = 1;
    template <class V>
    void visit(V& v)
    {
        v("color", color);
        v("alpha", alpha);
    }
};

// Kept Health's ID by mistake (never registered)
struct CopiedHealthId
{
    static constexpr const char* name = "Mana";
    static constexpr ComponentId id   = ComponentId(TestComponent::Health);
    static constexpr uint32_t version = 1;
    double mana                       = 0;
    template <class V>
    void visit(V& v)
    {
        v("mana", mana);
    }
};

// Health's name, ID and size, but another C++ type (never registered)
struct HealthLookalike
{
    static constexpr const char* name = "Health";
    static constexpr ComponentId id   = ComponentId(TestComponent::Health);
    static constexpr uint32_t version = 1;
    double a = 0, b = 0;
    template <class V>
    void visit(V& v)
    {
        v("a", a);
        v("b", b);
    }
};

// Another ID, but Health's name
struct SameNameAsHealth
{
    static constexpr const char* name = "Health";
    static constexpr ComponentId id   = 120;
    static constexpr uint32_t version = 1;
    double x                          = 0;
    template <class V>
    void visit(V& v)
    {
        v("x", x);
    }
};

void registerTest(Registry& r)
{
    r.registerComponent<Health>(Replicate::Host);
    r.registerComponent<Velocity>(Replicate::OwnerAuthoritative);
    r.registerComponent<Tag>();
    r.registerComponent<Inventory>(Replicate::HostReliable);
}

std::vector<uint32_t> indices(View<Health>&& view)
{
    std::vector<uint32_t> r;
    view.each([&](Entity e, Health&) { r.push_back(e.index); });
    return r;
}
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// Entities

TEST(Ecs, CreateDestroyAndStaleHandles)
{
    Registry r;
    EXPECT_FALSE(r.alive(Entity{}));
    Entity a = r.create(), b = r.create();
    EXPECT_EQ(a.index, 1u);
    EXPECT_EQ(b.index, 2u);
    EXPECT_TRUE(a.valid());
    EXPECT_TRUE(r.alive(a));
    EXPECT_EQ(r.aliveCount(), 2u);
    r.destroy(a);
    EXPECT_FALSE(r.alive(a));
    EXPECT_EQ(r.aliveCount(), 1u);
    // The slot is reused with a higher generation: the old handle stays stale
    Entity c = r.create();
    EXPECT_EQ(c.index, a.index);
    EXPECT_GT(c.generation, a.generation);
    EXPECT_FALSE(r.alive(a));
    EXPECT_TRUE(r.alive(c));
    EXPECT_FALSE(r.alive(Entity{999, 0}));
}

// A slot whose generation wraps around is retired: a stale handle with generation 0 can never match again
TEST(Ecs, GenerationWrapAroundRetiresTheSlot)
{
    Registry r;
    Entity a = r.create();
    r.destroy(a);
    r.debugSetGeneration(a.index, 0xFFFFFFFFu);
    Entity b = r.create();
    EXPECT_EQ(b.index, a.index);
    EXPECT_EQ(b.generation, 0xFFFFFFFFu);
    r.destroy(b);  // the generation wraps to 0: the slot is not reused
    Entity c = r.create();
    EXPECT_NE(c.index, a.index);
    EXPECT_FALSE(r.alive(Entity{a.index, 0}));
}

TEST(Ecs, IterationGuardSurvivesExceptionsAndNesting)
{
    Registry r;
    registerTest(r);
    Entity e = r.create();
    r.add<Health>(e);
    r.add<Tag>(e);
    EXPECT_THROW(r.view<Health>().each([](Entity, Health&) { throw std::runtime_error("x"); }), std::runtime_error);
    EXPECT_FALSE(r.iterating());
    // Nested views (reads inside a view are fine)
    int inner = 0;
    r.view<Health>().each([&](Entity, Health&) { r.view<const Tag>().each([&](Entity, const Tag&) { inner++; }); });
    EXPECT_EQ(inner, 1);
    EXPECT_FALSE(r.iterating());
    // Structural changes work again after the view
    r.remove<Tag>(e);
    EXPECT_FALSE(r.has<Tag>(e));
}

TEST(Ecs, FreeSlotsAreReusedInFifoOrder)
{
    Registry r;
    std::vector<Entity> e;
    for (int i = 0; i < 5; ++i) e.push_back(r.create());
    r.destroy(e[3]);
    r.destroy(e[1]);
    r.destroy(e[4]);
    EXPECT_EQ(r.create().index, e[3].index);
    EXPECT_EQ(r.create().index, e[1].index);
    EXPECT_EQ(r.create().index, e[4].index);
    EXPECT_EQ(r.create().index, 6u);
}

// Regression: the room for releasing a reserved slot was made by the World before reserving, and destroys appended to
// the free list afterwards could use it up; release could then allocate during an aborted step
TEST(Ecs, DestroysKeepTheRoomForReleasingReservedSlots)
{
    Registry r;
    std::vector<Entity> alive;
    for (int i = 0; i < 64; ++i) alive.push_back(r.create());
    Entity a = r.reserve();
    Entity b = r.reserve();
    EXPECT_EQ(r.reservedCount(), 2u);
    for (Entity e : alive)
    {
        r.destroy(e);
        EXPECT_TRUE(r.canReleaseWithoutAllocation());
    }
    r.release(a);
    EXPECT_TRUE(r.canReleaseWithoutAllocation());
    r.activate(b);
    EXPECT_EQ(r.reservedCount(), 0u);
    EXPECT_TRUE(r.alive(b));
    EXPECT_FALSE(r.reserved(a));
    EXPECT_EQ(r.aliveCount(), 1u);
}

TEST(Ecs, ManyCreateDestroyCycles)
{
    Registry r;
    registerTest(r);
    std::vector<Entity> live;
    for (int i = 0; i < 10000; ++i)
    {
        Entity e = r.create();
        r.add<Health>(e, double(i), 100.0);
        live.push_back(e);
        if (i % 3 == 0)
        {
            r.destroy(live.front());
            live.erase(live.begin());
        }
    }
    EXPECT_EQ(r.aliveCount(), uint32_t(live.size()));
    EXPECT_EQ(r.view<Health>().count(), live.size());
    for (Entity e : live) EXPECT_TRUE(r.alive(e));
}

// ---------------------------------------------------------------------------------------------------------------------
// Components

TEST(Ecs, AddGetReadRemove)
{
    Registry r;
    registerTest(r);
    Entity e = r.create();
    EXPECT_TRUE(r.has<Transform>(e));
    EXPECT_FALSE(r.has<Health>(e));
    EXPECT_EQ(r.tryGet<Health>(e), nullptr);
    r.add<Health>(e, 50.0, 80.0);
    EXPECT_TRUE(r.has<Health>(e));
    EXPECT_EQ(r.read<Health>(e).hp, 50.0);
    EXPECT_EQ(r.read<Health>(e).maxHp, 80.0);
    r.get<Health>(e).hp = 10;
    EXPECT_EQ(r.tryRead<Health>(e)->hp, 10.0);
    r.add<Tag>(e);
    EXPECT_TRUE(r.entityMask(e).containsAll(Registry::mask<Transform, Health, Tag>()));
    EXPECT_FALSE(r.entityMask(e).test(Velocity::id));
    r.remove<Health>(e);
    EXPECT_FALSE(r.has<Health>(e));
    EXPECT_TRUE(r.has<Tag>(e));
    // Transform defaults and the built-in registration
    EXPECT_EQ(r.read<Transform>(e).position(), Vec3::Zero());
    EXPECT_TRUE(r.registered(Transform::id));
    EXPECT_STREQ(r.componentInfo(Transform::id).name, "Transform");
    EXPECT_EQ(r.componentInfo(Health::id).replicate, Replicate::Host);
    EXPECT_EQ(r.componentInfo(Inventory::id).version, 2u);
}

// Regression: add reserved the pool storage before constructing, which invalidated an argument that refers to another
// component of the same pool when the pool was full
TEST(Ecs, AddCopiesFromAComponentOfTheSamePool)
{
    Registry r;
    registerTest(r);
    std::vector<Entity> e;
    for (int i = 0; i < 64; ++i)
    {
        e.push_back(r.create());
        // Each add copies from the previous entity's component, also when the pool grows
        if (i == 0)
            r.add<Health>(e.back(), 1.0, 7.0);
        else
            r.add<Health>(e.back(), r.read<Health>(e[size_t(i - 1)]));
    }
    for (Entity x : e) EXPECT_EQ(r.read<Health>(x).maxHp, 7.0);
}

TEST(Ecs, DestroyRemovesAllComponentsAndSwapAndPopKeepsOthers)
{
    Registry r;
    registerTest(r);
    std::vector<Entity> e;
    for (int i = 0; i < 4; ++i)
    {
        e.push_back(r.create());
        r.add<Health>(e.back(), double(i), 100.0);
    }
    r.add<Velocity>(e[1], Vec3(1, 2, 3));
    r.destroy(e[1]);
    // The last element was moved into the gap; every other entity still finds its own component
    EXPECT_EQ(r.read<Health>(e[0]).hp, 0.0);
    EXPECT_EQ(r.read<Health>(e[2]).hp, 2.0);
    EXPECT_EQ(r.read<Health>(e[3]).hp, 3.0);
    EXPECT_EQ(r.view<Velocity>().count(), 0u);
    // A new entity in the old slot starts without components
    Entity n = r.create();
    EXPECT_EQ(n.index, e[1].index);
    EXPECT_FALSE(r.has<Health>(n));
    EXPECT_FALSE(r.has<Velocity>(n));
}

TEST(Ecs, FlagsAndTransformDefaultsOfAReusedSlot)
{
    Registry r;
    Entity e = r.create();
    EXPECT_EQ(r.flags(e), Registry::kSimulated);  // every new entity is Simulated (§12.3)
    r.setFlags(e, 0x80);
    r.get<Transform>(e).setPosition(Vec3(1, 2, 3));
    r.destroy(e);
    Entity n = r.create();
    EXPECT_EQ(r.flags(n), Registry::kSimulated);  // the old flags are gone
    EXPECT_EQ(r.read<Transform>(n).position(), Vec3::Zero());
}

TEST(Ecs, ValidateRegistration)
{
    Registry ok;
    registerTest(ok);
    ok.validateRegistration();
    EXPECT_EQ(ok.registeredCount(), 5);
}

// The accepted side of the float rule: a Render* component with allowFloat registers and is usable
TEST(Ecs, RenderComponentWithFloatsRegisters)
{
    Registry r;
    r.registerComponent<RenderThing>(Replicate::Never, Saved::No);
    Entity e = r.create();
    r.add<RenderThing>(e);
    EXPECT_EQ(r.read<RenderThing>(e).alpha, 1.0f);
    EXPECT_EQ(r.componentInfo(RenderThing::id).saved, Saved::No);
}

#ifdef SAIGA_ASSERTS
TEST(EcsDeathTest, MisuseAsserts)
{
    Registry r;
    registerTest(r);
    Entity e = r.create();
    r.add<Health>(e);
    EXPECT_DEATH(r.add<Health>(e), "already has Health");
    EXPECT_DEATH(r.get<Velocity>(e), "has no Velocity");
    EXPECT_DEATH(r.remove<Velocity>(e), "has no Velocity");
    Entity stale = e;
    r.destroy(e);
    EXPECT_DEATH(r.get<Health>(stale), "stale");
    EXPECT_DEATH(r.destroy(stale), "not alive");
    EXPECT_DEATH(r.get<Health>(Entity{}), "invalid handle");
    // Unregistered component
    Registry bare;
    Entity b = bare.create();
    EXPECT_DEATH(bare.add<Health>(b), "not registered");
    // ID used twice
    EXPECT_DEATH(r.registerComponent<Health>(), "already used");
    // has on an unregistered component asserts like get (it is a registration bug, not "no component")
    EXPECT_DEATH(bare.has<Health>(b), "not registered");
    // A copy-pasted component that kept the ID of another one: the registered pool is not reinterpreted
    Entity c = r.create();
    r.add<Health>(c);
    EXPECT_DEATH(r.get<CopiedHealthId>(c), "registered as Health, not Mana");
    EXPECT_DEATH(r.has<CopiedHealthId>(c), "registered as Health, not Mana");
    // Same ID, name and size: the C++ type itself is checked before the downcast
    static_assert(sizeof(HealthLookalike) == sizeof(Health), "the test needs the same size");
    EXPECT_DEATH(r.get<HealthLookalike>(c), "another C\\+\\+ type");
    // A second component with an existing name (saves load by name)
    EXPECT_DEATH(r.registerComponent<SameNameAsHealth>(), "name Health is already used");
}

// A component whose destructor creates an entity in the same registry (reentrant structural change during destroy)
Registry* g_reentrantRegistry = nullptr;
struct CreatesOnDestruction
{
    static constexpr const char* name = "CreatesOnDestruction";
    static constexpr ComponentId id   = 121;
    static constexpr uint32_t version = 1;
    bool armed                        = false;
    CreatesOnDestruction()            = default;
    CreatesOnDestruction(CreatesOnDestruction&& o) noexcept : armed(o.armed) { o.armed = false; }
    CreatesOnDestruction& operator=(CreatesOnDestruction&& o) noexcept
    {
        armed   = o.armed;
        o.armed = false;
        return *this;
    }
    ~CreatesOnDestruction()
    {
        if (armed && g_reentrantRegistry) g_reentrantRegistry->create();
    }
    template <class V>
    void visit(V&)
    {
    }
};

TEST(EcsDeathTest, CreateFromAComponentDestructorDuringDestroy)
{
    Registry r;
    r.registerComponent<CreatesOnDestruction>();
    g_reentrantRegistry = &r;
    Entity e            = r.create();
    r.add<CreatesOnDestruction>(e);
    r.get<CreatesOnDestruction>(e).armed = true;
    // Without the check, create() got the slot that destroy was freeing
    EXPECT_DEATH(r.destroy(e), "while destroy removes the components");
    r.get<CreatesOnDestruction>(e).armed = false;
    g_reentrantRegistry                  = nullptr;
}

TEST(EcsDeathTest, StructuralChangeDuringIteration)
{
    Registry r;
    registerTest(r);
    Entity e = r.create();
    r.add<Health>(e);
    EXPECT_DEATH(r.view<Health>().each([&](Entity x, Health&) { r.add<Tag>(x); }), "while a view iterates");
    EXPECT_DEATH(r.view<Health>().each([&](Entity x, Health&) { r.destroy(x); }), "while a view iterates");
    EXPECT_DEATH(r.view<Health>().each([&](Entity, Health&) { r.create(); }), "while a view iterates");
    r.add<Tag>(e);
    EXPECT_DEATH(r.view<Health>().each([&](Entity x, Health&) { r.remove<Tag>(x); }), "while a view iterates");
}

TEST(EcsDeathTest, RegistrationGap)
{
    // Game IDs 64 and 66 without 65
    Registry r;
    r.registerComponent<Health>();
    r.registerComponent<Tag>();
    EXPECT_DEATH(r.validateRegistration(), "65 is not registered");
}
#endif

// ---------------------------------------------------------------------------------------------------------------------
// Views

TEST(Ecs, ViewsOfOneTwoAndThreeComponents)
{
    Registry r;
    registerTest(r);
    std::vector<Entity> e;
    for (int i = 0; i < 10; ++i)
    {
        e.push_back(r.create());
        r.add<Health>(e.back(), double(i), 100.0);
        if (i % 2 == 0) r.add<Velocity>(e.back(), Vec3(i, 0, 0));
        if (i % 3 == 0) r.add<Tag>(e.back());
    }
    EXPECT_EQ(r.view<Health>().count(), 10u);
    EXPECT_EQ((r.view<Health, Velocity>().count()), 5u);
    EXPECT_EQ((r.view<Health, Velocity, Tag>().count()), 2u);  // 0, 6
    EXPECT_EQ(r.view<Transform>().count(), 10u);

    // Writable and read-only access
    r.view<Health, const Velocity>().each([](Entity, Health& h, const Velocity& v) { h.hp += v.v(0); });
    EXPECT_EQ(r.read<Health>(e[4]).hp, 8.0);
    EXPECT_EQ(r.read<Health>(e[3]).hp, 3.0);

    // Transform-only view over the slots, in index order, without destroyed entities
    r.destroy(e[2]);
    std::vector<uint32_t> seen;
    r.view<Transform>().each([&](Entity x, Transform&) { seen.push_back(x.index); });
    EXPECT_EQ(seen.size(), 9u);
    EXPECT_TRUE(std::is_sorted(seen.begin(), seen.end()));
    EXPECT_EQ(std::count(seen.begin(), seen.end(), e[2].index), 0);

    // Transform combined with a component
    r.view<Transform, const Tag>().each([](Entity, Transform& t, const Tag&) { t.setPosition(Vec3(1, 1, 1)); });
    EXPECT_EQ(r.read<Transform>(e[3]).position(), Vec3(1, 1, 1));
    EXPECT_EQ(r.read<Transform>(e[4]).position(), Vec3::Zero());
}

TEST(Ecs, ViewOrderIsDeterministic)
{
    // The same sequence of structural changes gives the same iteration order (the dense order of the pool)
    auto build = []()
    {
        auto r = std::make_unique<Registry>();
        registerTest(*r);
        std::vector<Entity> e;
        for (int i = 0; i < 50; ++i)
        {
            e.push_back(r->create());
            r->add<Health>(e.back());
        }
        for (int i = 0; i < 50; i += 7) r->remove<Health>(e[size_t(i)]);
        for (int i = 3; i < 50; i += 11) r->destroy(e[size_t(i)]);
        for (int i = 0; i < 5; ++i) r->add<Health>(r->create());
        return r;
    };
    auto a = build(), b = build();
    auto oa = indices(a->view<Health>()), ob = indices(b->view<Health>());
    EXPECT_EQ(oa, ob);
    // Not sorted: the swap-and-pop order, which only depends on the history
    EXPECT_FALSE(std::is_sorted(oa.begin(), oa.end()));
}

TEST(Ecs, ViewSkipsEntitiesWithoutAllComponents)
{
    Registry r;
    registerTest(r);
    Entity a = r.create(), b = r.create();
    r.add<Health>(a);
    r.add<Velocity>(b);
    r.add<Velocity>(a);
    r.remove<Health>(a);
    EXPECT_EQ((r.view<Health, Velocity>().count()), 0u);
    EXPECT_EQ(r.view<Velocity>().count(), 2u);
}

// ---------------------------------------------------------------------------------------------------------------------
// visit

TEST(Ecs, VisitBinaryRoundTripOfEveryFieldType)
{
    Inventory in;
    in.open     = true;
    in.label    = "chest";
    in.owner    = Entity{7, 3};
    in.pid      = PersistentId{99};
    in.chunk    = ChunkId{2, -5, 17};
    in.v2       = Vec2(1.5, -2);
    in.v4       = Vec4(1, 2, 3, 4);
    in.q        = Quat(0.5, 0.5, 0.5, 0.5);
    in.m3(0, 1) = 7;
    in.m4(3, 2) = -9;
    in.selected = ItemKind::Stone;
    in.slots    = {4, 5, 6};
    in.items    = {{ItemKind::Stone, 3}, {ItemKind::Wood, 12}};
    in.nested   = {{1.0}, {}, {2.0, 3.0}};

    std::vector<uint8_t> bytes = toBytes(in);
    Inventory out;
    std::string error;
    ASSERT_TRUE(fromBytes(out, bytes, &error)) << error;
    EXPECT_EQ(out.open, true);
    EXPECT_EQ(out.i8, -3);
    EXPECT_EQ(out.u16, 65000);
    EXPECT_EQ(out.i64, -1234567890123ll);
    EXPECT_EQ(out.u64, 0xFEDCBA9876543210ull);
    EXPECT_TRUE(std::signbit(out.d));
    EXPECT_EQ(out.label, "chest");
    EXPECT_EQ(out.owner, (Entity{7, 3}));
    EXPECT_EQ(out.pid, PersistentId{99});
    EXPECT_EQ(out.chunk, (ChunkId{2, -5, 17}));
    EXPECT_EQ(out.v2, in.v2);
    EXPECT_EQ(out.v4, in.v4);
    EXPECT_EQ(out.q.coeffs(), in.q.coeffs());
    EXPECT_EQ(out.m3, in.m3);
    EXPECT_EQ(out.m4, in.m4);
    EXPECT_EQ(out.selected, ItemKind::Stone);
    EXPECT_EQ(out.slots, in.slots);
    EXPECT_EQ(out.items, in.items);
    EXPECT_EQ(out.nested, in.nested);
    // Deterministic bytes: writing again gives the same bytes
    EXPECT_EQ(toBytes(out), bytes);

    RenderThing rt;
    rt.alpha = 0.25f;
    RenderThing rt2;
    rt2.alpha = 0;
    ASSERT_TRUE(fromBytes(rt2, toBytes(rt), &error)) << error;
    EXPECT_EQ(rt2.alpha, 0.25f);
    EXPECT_EQ(rt2.color, rt.color);
}

TEST(Ecs, VisitBinaryLayoutIsFixed)
{
    // Little-endian integers, the double bit pattern, u64 lengths: a change here changes every save and hash
    Health h;
    h.hp                          = 1.0;
    h.maxHp                       = 2.0;
    std::vector<uint8_t> expected = {0, 0, 0, 0, 0, 0, 0xF0, 0x3F, 0, 0, 0, 0, 0, 0, 0, 0x40};
    EXPECT_EQ(toBytes(h), expected);
    ItemStack s{ItemKind::Stone, 258};
    EXPECT_EQ(toBytes(s), (std::vector<uint8_t>{1, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0}));
}

// Regression: the input checks were asserts, so a release build (SAIGA_ASSERTS=OFF) read past a truncated buffer.
// Saves and network packets are untrusted: fromBytes reports the error in every build.
TEST(Ecs, MalformedBytesAreRejected)
{
    std::string error;
    Inventory in;
    in.items   = {{ItemKind::Stone, 3}};
    auto bytes = toBytes(in);
    Inventory out;
    ASSERT_TRUE(fromBytes(out, bytes, &error)) << error;
    for (size_t cut : {size_t(1), size_t(5), bytes.size() / 2})
    {
        auto truncated = bytes;
        truncated.resize(bytes.size() - cut);
        Inventory o;
        EXPECT_FALSE(fromBytes(o, truncated, &error));
        EXPECT_NE(error.find("truncated"), std::string::npos) << error;
    }
    // An enum value without a name
    std::vector<uint8_t> badEnum = {7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ItemStack s;
    EXPECT_FALSE(fromBytes(s, badEnum, &error));
    EXPECT_NE(error.find("enum value 7 out of range"), std::string::npos) << error;
    // Bytes left over
    auto extra = toBytes(Health{});
    extra.push_back(0);
    Health h;
    EXPECT_FALSE(fromBytes(h, extra, &error));
    EXPECT_NE(error.find("left over"), std::string::npos) << error;
    // A vector length far beyond the input does not allocate: 2^40 elements of 12 bytes in 20 bytes of input
    std::vector<uint8_t> hugeVector = {0, 0, 0, 0, 0, 1, 0, 0};
    hugeVector.resize(28, 0);
    std::vector<ItemStack> items;
    EXPECT_FALSE(fromBytes(items, hugeVector, &error));
    EXPECT_NE(error.find("does not fit"), std::string::npos) << error;
    // A bool byte other than 0 or 1 (no second encoding of the same state)
    std::vector<uint8_t> badBool = {2};
    bool flag                    = false;
    EXPECT_FALSE(fromBytes(flag, badBool, &error));
    EXPECT_NE(error.find("neither 0 nor 1"), std::string::npos) << error;
    // A string longer than the input
    std::vector<uint8_t> longString = {100, 0, 0, 0, 0, 0, 0, 0, 'a'};
    std::string str;
    EXPECT_FALSE(fromBytes(str, longString, &error));
}

struct TagHolder
{
    static constexpr const char* name = "TagHolder";
    static constexpr uint32_t version = 1;
    std::vector<Tag> tags;
    template <class V>
    void visit(V& v)
    {
        v("tags", tags);
    }
};

// Regression: a vector of fieldless structs was written as a count only, and the reader rejected it (it required one
// byte per element)
TEST(Ecs, VectorOfEmptyStructsRoundTrips)
{
    using Holder = TagHolder;
    Holder h;
    h.tags.resize(3);
    Holder out;
    std::string error;
    ASSERT_TRUE(fromBytes(out, toBytes(h), &error)) << error;
    EXPECT_EQ(out.tags.size(), 3u);
    // The count of zero-byte elements is capped (a corrupt count cannot allocate without bound); the largest accepted
    // count round-trips, the writer refuses one more (death test below)
    std::vector<uint8_t> huge = {0, 0, 0, 0, 0, 1, 0, 0};
    EXPECT_FALSE(fromBytes(out, huge, &error));
    h.tags.resize(size_t(kMaxEmptyElements));
    ASSERT_TRUE(fromBytes(out, toBytes(h), &error)) << error;
    EXPECT_EQ(out.tags.size(), size_t(kMaxEmptyElements));
    // One more: an error in every build (not an assert), so no unreadable buffer is produced
    h.tags.resize(size_t(kMaxEmptyElements) + 1);
    std::vector<uint8_t> bytes;
    EXPECT_FALSE(toBytes(h, bytes, &error));
    EXPECT_NE(error.find("more than kMaxEmptyElements"), std::string::npos) << error;
    EXPECT_DEATH(toBytes(h), "more than kMaxEmptyElements");
}

// Regression: a struct that contains itself through a vector (a tree) made the field check of registerComponent
// recurse forever
struct TreeNode
{
    static constexpr const char* name = "TreeNode";
    static constexpr uint32_t version = 1;
    int32_t value                     = 0;
    std::vector<TreeNode> children;
    template <class V>
    void visit(V& v)
    {
        v("value", value);
        v("children", children);
    }
};
struct Tree
{
    static constexpr const char* name = "Tree";
    static constexpr ComponentId id   = 110;
    static constexpr uint32_t version = 1;
    TreeNode root;
    template <class V>
    void visit(V& v)
    {
        v("root", root);
    }
};

TEST(Ecs, SelfContainingStructRegistersAndRoundTrips)
{
    Registry r;
    r.registerComponent<Tree>();
    Tree t;
    t.root.value = 1;
    t.root.children.resize(2);
    t.root.children[1].value = 3;
    t.root.children[1].children.resize(1);
    t.root.children[1].children[0].value = 4;
    Tree out;
    std::string error;
    ASSERT_TRUE(fromBytes(out, toBytes(t), &error)) << error;
    ASSERT_EQ(out.root.children.size(), 2u);
    ASSERT_EQ(out.root.children[1].children.size(), 1u);
    EXPECT_EQ(out.root.children[1].children[0].value, 4);

    // The nesting limit: the writer refuses a tree deeper than kMaxStructDepth (no unreadable buffer) ...
    TreeNode deep;
    TreeNode* n = &deep;
    for (int i = 0; i < kMaxStructDepth + 5; ++i)
    {
        n->children.resize(1);
        n = &n->children[0];
    }
    std::vector<uint8_t> bytes;
    EXPECT_FALSE(toBytes(deep, bytes, &error));
    EXPECT_NE(error.find("nested deeper"), std::string::npos) << error;
    // ... and the reader rejects such input (a hand-built buffer: per level value 0 and one child, then no child)
    // instead of overflowing the stack
    std::vector<uint8_t> wire;
    for (int i = 0; i < kMaxStructDepth + 5; ++i)
    {
        wire.insert(wire.end(), {0, 0, 0, 0});              // value
        wire.insert(wire.end(), {1, 0, 0, 0, 0, 0, 0, 0});  // one child
    }
    wire.insert(wire.end(), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});  // the innermost node: value, no children
    TreeNode deepOut;
    EXPECT_FALSE(fromBytes(deepOut, wire, &error));
    EXPECT_NE(error.find("nested deeper"), std::string::npos) << error;
    // A chain of exactly kMaxStructDepth nested nodes (the root and 63 descendants) round-trips
    TreeNode ok;
    n = &ok;
    for (int i = 0; i < kMaxStructDepth - 1; ++i)
    {
        n->children.resize(1);
        n = &n->children[0];
    }
    ASSERT_TRUE(toBytes(ok, bytes, &error)) << error;
    TreeNode okOut;
    EXPECT_TRUE(fromBytes(okOut, bytes, &error)) << error;
}

TEST(Ecs, WriterRefusesValuesTheReaderRejects)
{
    std::string error;
    std::vector<uint8_t> bytes;
    // An enum value without a name (Count)
    ItemStack s{ItemKind::Count, 1};
    EXPECT_FALSE(toBytes(s, bytes, &error));
    EXPECT_NE(error.find("has no name"), std::string::npos) << error;
    ItemStack valid{ItemKind::Stone, 1};
    EXPECT_TRUE(toBytes(valid, bytes, &error)) << error;
}

// Regression: toBytes cleared its output before reading the value, so an output that is a field of the value was
// written as empty
struct ByteBlob
{
    static constexpr const char* name = "ByteBlob";
    static constexpr uint32_t version = 1;
    std::vector<uint8_t> data;
    template <class V>
    void visit(V& v)
    {
        v("data", data);
    }
};

TEST(Ecs, ToBytesIntoAFieldOfTheValue)
{
    ByteBlob b;
    b.data = {1, 2, 3};
    std::string error;
    ASSERT_TRUE(toBytes(b, b.data, &error)) << error;
    // u64 length 3, then the three bytes
    EXPECT_EQ(b.data, (std::vector<uint8_t>{3, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3}));
}

// Regression: destroy cleared the mask after removing all components, so a component destructor saw has() == true
// for a component that was already gone
Registry* g_observedRegistry = nullptr;
Entity g_observedEntity;
int g_hasHealthInDestructor = -1;
struct ObservesHealth
{
    static constexpr const char* name = "ObservesHealth";
    static constexpr ComponentId id   = 122;
    static constexpr uint32_t version = 1;
    bool armed                        = false;
    ObservesHealth()                  = default;
    ObservesHealth(ObservesHealth&& o) noexcept : armed(o.armed) { o.armed = false; }
    ObservesHealth& operator=(ObservesHealth&& o) noexcept
    {
        armed   = o.armed;
        o.armed = false;
        return *this;
    }
    ~ObservesHealth()
    {
        if (armed && g_observedRegistry)
            g_hasHealthInDestructor = g_observedRegistry->entityMask(g_observedEntity).test(Health::id) ? 1 : 0;
    }
    template <class V>
    void visit(V&)
    {
    }
};

TEST(Ecs, DestroyClearsEachMaskBitWithItsComponent)
{
    Registry r;
    registerTest(r);
    r.registerComponent<ObservesHealth>();
    Entity e = r.create();
    r.add<Health>(e);
    r.add<ObservesHealth>(e);
    r.get<ObservesHealth>(e).armed = true;
    g_observedRegistry             = &r;
    g_observedEntity               = e;
    r.destroy(e);  // Health (ID 64) is removed before ObservesHealth (ID 122)
    g_observedRegistry = nullptr;
    EXPECT_EQ(g_hasHealthInDestructor, 0);
}

TEST(Ecs, DescribeFields)
{
    auto f = describeFields<Inventory>();
    std::map<std::string, std::string> types;
    for (auto& x : f) types[x.name] = x.type;
    ASSERT_EQ(f.size(), 19u);
    EXPECT_EQ(f[0].name, "open");
    EXPECT_EQ(types["open"], "bool");
    EXPECT_EQ(types["i8"], "int8");
    EXPECT_EQ(types["u16"], "uint16");
    EXPECT_EQ(types["d"], "double");
    EXPECT_EQ(types["label"], "string");
    EXPECT_EQ(types["owner"], "Entity");
    EXPECT_EQ(types["chunk"], "ChunkId");
    EXPECT_EQ(types["v4"], "vec4d");
    EXPECT_EQ(types["q"], "quatd");
    EXPECT_EQ(types["m4"], "mat4d");
    EXPECT_EQ(types["selected"], "enum");
    auto selected = std::find_if(f.begin(), f.end(), [](const FieldInfo& x) { return x.name == "selected"; });
    ASSERT_NE(selected, f.end());
    EXPECT_EQ(selected->enumValues, (std::vector<std::string>{"wood", "stone"}));
    EXPECT_EQ(types["slots"], "array<int32,3>");
    EXPECT_EQ(types["items"], "vector<ItemStack>");
    EXPECT_EQ(types["nested"], "vector<vector<double>>");
    auto t = describeFields<Transform>();
    ASSERT_EQ(t.size(), 4u);
    EXPECT_EQ(t[0].name, "space");
    EXPECT_EQ(t[0].type, "uint16");
    EXPECT_EQ(t[2].type, "quatd");
    auto rf = describeFields<RenderThing>();
    EXPECT_EQ(rf[0].type, "vec3f");
    EXPECT_EQ(rf[1].type, "float");
}

// Compile-time field rules (docs/engine_design.md §7.2): the supported types pass checkFieldType; the rejected ones
// (float without allowFloat, raw pointers, std::map, enums without names) are static_asserts and cannot be tested at
// run time. This test documents the accepted set through the traits.
TEST(Ecs, FieldTypeTraits)
{
    using namespace VisitDetail;
    static_assert(IsVisitStruct<ItemStack>::value, "");
    static_assert(IsVisitStruct<ChunkId>::value, "");
    static_assert(!IsVisitStruct<Vec3>::value, "");
    static_assert(AllowsFloat<RenderThing>::value, "");
    static_assert(!AllowsFloat<Health>::value, "");
    static_assert(isInteger<int8_t> && isInteger<uint64_t> && !isInteger<bool> && !isInteger<char>, "");
    static_assert(isDoubleMath<Mat4> && isFloatMath<vec3> && isQuat<Quat>, "");
    SUCCEED();
}

// ---------------------------------------------------------------------------------------------------------------------
// Transform, value types, log

TEST(Ecs, TransformSettersChangeOnlyOnARealChange)
{
    Transform t;
    EXPECT_FALSE(t.changedInThisStep());
    t.setPosition(Vec3::Zero());
    t.setRotation(Quat::Identity());
    t.setScale(Vec3::Ones());
    EXPECT_FALSE(t.changedInThisStep());
    t.setPosition(Vec3(0, -0.0, 0));  // -0.0 differs bitwise
    EXPECT_TRUE(t.changedInThisStep());
    EXPECT_FALSE(t.wasTeleported());
    Transform u;
    u.teleport(Vec3(1, 2, 3), Quat::Identity());
    EXPECT_TRUE(u.wasTeleported());
    EXPECT_EQ(u.position(), Vec3(1, 2, 3));

    // visit round trip (the flags are not visited)
    Transform v;
    std::string error;
    ASSERT_TRUE(fromBytes(v, toBytes(u), &error)) << error;
    EXPECT_EQ(v.position(), Vec3(1, 2, 3));
    EXPECT_FALSE(v.changedInThisStep());
}

TEST(Ecs, ChunkRegionAndInstanceKeys)
{
    EXPECT_TRUE((ChunkId{0, 1, 2} == ChunkId{0, 1, 2}));
    EXPECT_TRUE((ChunkId{0, 1, 2} < ChunkId{0, 1, 3}));
    EXPECT_TRUE((ChunkId{0, 5, 0} < ChunkId{1, -5, 0}));
    EXPECT_TRUE((RegionId{0, 1, 0, 0} < RegionId{0, 2, -9, -9}));
    EXPECT_TRUE((InstanceKey{1, 9} < InstanceKey{2, 0}));
    RegionId r{3, 4, -1, 2}, r2;
    std::string error;
    ASSERT_TRUE(fromBytes(r2, toBytes(r), &error)) << error;
    EXPECT_EQ(r, r2);
}

TEST(Ecs, EntityLogPrefixesTheEntity)
{
    setLogEcho(false);
    uint64_t before = engineLog().lastSequence();
    SAIGA_LOG_E("ai", (Entity{12, 3}), "flees from %d", 7);
    SAIGA_LOG_E("ai", (Entity{1, 0}), "idle");
    setLogEcho(true);
    auto lines = engineLog().since(before);
    ASSERT_EQ(lines.size(), 2u);
    EXPECT_EQ(lines[0].message, "e12:3 flees from 7");
    EXPECT_EQ(lines[1].message, "e1:0 idle");
    EXPECT_EQ(lines[0].category, "ai");
}
