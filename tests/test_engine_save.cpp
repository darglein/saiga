/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Saves, loads, migrations and save comparison (docs/engine_design.md §15; stage S11)

#include "saiga/engine/components/Health.h"
#include "saiga/engine/components/LocalTransform.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/Persistent.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/gameplay/GameObject.h"
#include "saiga/engine/save/SaveDiff.h"
#include "saiga/engine/save/SaveFormat.h"
#include "saiga/engine/save/SaveReader.h"
#include "saiga/engine/save/SchemaTable.h"
#include "saiga/engine/world/World.h"
#include "saiga/saiga_buildconfig.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

using namespace Saiga;
using namespace Saiga::Engine;
using Json = json11::Json;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);

// ---- test types
// ------------------------------------------------------------------------------------------------------

enum class Mood
{
    Calm,
    Angry,
    Count
};
const char* const kMoodNames[] = {"Calm", "Angry"};

struct Weather
{
    static constexpr const char* name = "Weather";
    static constexpr ComponentId id   = ComponentId(kFirstGameComponent);
    static constexpr uint32_t version = 1;
    double rain                       = 0;
    template <class V>
    void visit(V& v)
    {
        v("rain", rain);
    }
};

// Refers to other entities (stale handles must survive a load unchanged)
struct Link
{
    static constexpr const char* name = "Link";
    static constexpr ComponentId id   = ComponentId(kFirstGameComponent + 1);
    static constexpr uint32_t version = 1;
    Entity target;
    PersistentId pid;
    template <class V>
    void visit(V& v)
    {
        v("target", target);
        v("pid", pid);
    }
};

// The state of the Walker game object (R-STATE-1): it moves along +x by speed per update
struct WalkerState
{
    static constexpr const char* name = "WalkerState";
    static constexpr ComponentId id   = ComponentId(kFirstGameComponent + 2);
    static constexpr uint32_t version = 1;
    double speed                      = 1;
    int32_t updates                   = 0;
    template <class V>
    void visit(V& v)
    {
        v("speed", speed);
        v("updates", updates);
    }
};

// A derived component that is not saved (rebuilt in onLoad)
struct WalkerRender
{
    static constexpr const char* name = "WalkerRender";
    static constexpr ComponentId id   = ComponentId(kFirstGameComponent + 3);
    static constexpr uint32_t version = 1;
    double brightness                 = 0;
    template <class V>
    void visit(V& v)
    {
        v("brightness", brightness);
    }
};

int g_configures = 0, g_onLoads = 0, g_onSpawns = 0;

class Walker final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(kFirstGameClass);
    static constexpr const char* name = "Walker";
    void configure() override
    {
        g_configures++;
        stride = 0.5;
    }
    void onSpawn() override
    {
        g_onSpawns++;
        add<WalkerState>();
        cache = read<WalkerState>().speed * 10;
    }
    void onLoad() override
    {
        g_onLoads++;
        // Robust against corrupted saves (the robustness test flips bytes)
        if (!has<WalkerState>()) return;
        cache = read<WalkerState>().speed * 10;
        if (!has<WalkerRender>()) add<WalkerRender>();
    }
    void update(double) override
    {
        if (!has<WalkerState>()) return;
        auto& s = write<WalkerState>();
        s.updates++;
        Vec3 p = transform().position();
        p.x() += s.speed * stride;
        transformWrite().setPosition(p);
    }
    double stride = 0;  // set in configure (constant)
    double cache  = 0;  // rebuilt in onSpawn / onLoad
};

class Sitter final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(kFirstGameClass + 1);
    static constexpr const char* name = "Sitter";
};

// Registered instead of Sitter in a later build
class Lounger final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(kFirstGameClass + 1);
    static constexpr const char* name = "Lounger";
};

bool g_throwOnLoad = false;
class Fragile final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(kFirstGameClass + 2);
    static constexpr const char* name = "Fragile";
    void onLoad() override
    {
        if (g_throwOnLoad) throw std::runtime_error("fragile broke");
    }
};

void baseComponents(Registry& r)
{
    r.registerGlobal<Weather>(Replicate::Host);
    r.registerComponent<Link>();
    r.registerComponent<WalkerState>();
    r.registerComponent<WalkerRender>(Replicate::Never, Saved::No);
}
void baseClasses(ClassList& c)
{
    c.add<Walker>();
    c.add<Sitter>();
    c.add<Fragile>();
}

WorldSetup baseSetup()
{
    WorldSetup s;
    s.components = baseComponents;
    s.classes    = baseClasses;
    return s;
}

// A world with every kind of saved state
std::vector<Entity> fill(World& w)
{
    std::vector<Entity> e;
    for (int i = 0; i < 6; ++i)
    {
        Entity x = w.create();
        w.get<Transform>(x).setPosition(Vec3(i * 3.0, 0, 1.5 * i));
        e.push_back(x);
    }
    w.add<Health>(e[0], 50.0, 100.0);
    w.add<Player>(e[1]);
    w.add<Parent>(e[2], Parent{e[1]});
    w.add<LocalTransform>(e[2], LocalTransform{Vec3(0, 1, 0), Quat::Identity(), Vec3::Ones()});
    w.makePersistent(e[3]);
    // A stale handle in a field
    Entity gone = w.create();
    w.destroy(gone);
    w.add<Link>(e[4], Link{gone, w.makePersistent(e[3])});
    w.destroy(e[5]);
    w.spawn<Walker>(Vec3(10, 0, 10));
    w.spawn<Sitter>(Vec3(-10, 0, 0));
    w.global<Weather>().rain = 0.25;
    w.events.emit(DamageEvent{e[0], e[1], 7});
    w.queueAction(PlayerAction{e[1], 3, Vec3(1, 2, 3)});
    w.random().nextU32();
    return e;
}

// The offset of a section's content in a save
size_t sectionOffset(const std::vector<uint8_t>& data, uint32_t tag)
{
    size_t pos = 8;
    while (pos + 12 <= data.size())
    {
        uint32_t t;
        uint64_t len;
        std::memcpy(&t, data.data() + pos, 4);
        std::memcpy(&len, data.data() + pos + 4, 8);
        if (t == tag) return pos + 12;
        pos += 12 + size_t(len);
    }
    return 0;
}
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// Round trips

TEST(Save, RoundTripRestoresEverySavedState)
{
    World a(baseSetup());
    std::vector<Entity> e           = fill(a);
    const std::vector<uint8_t> blob = a.save();
    // Deterministic
    EXPECT_EQ(blob, a.save());

    World b(baseSetup());
    b.spawn<Walker>(Vec3(5, 5, 5));  // replaced by the load
    const LoadResult r = b.tryLoad(blob);
    ASSERT_TRUE(r.ok()) << r.error;
    EXPECT_EQ(b.save(), blob);
    EXPECT_EQ(b.hash(), a.hash());
    EXPECT_EQ(World::diff(a, b), "");
    EXPECT_EQ(b.loadCount(), 1u);
    EXPECT_EQ(b.time(), a.time());
    // Handles keep their meaning
    EXPECT_TRUE(b.alive(e[2]));
    EXPECT_EQ(b.read<Parent>(e[2]).entity, e[1]);
    EXPECT_FALSE(b.alive(e[5]));
    EXPECT_FALSE(b.alive(b.read<Link>(e[4]).target));
    EXPECT_EQ(b.resolve(b.read<Link>(e[4]).pid), e[3]);
    EXPECT_DOUBLE_EQ(b.readGlobal<Weather>().rain, 0.25);
    // The next entities get the same indices and generations
    EXPECT_EQ(b.create(), a.create());
    // Grid queries work at once (the grid was rebuilt)
    EXPECT_EQ(b.queryRadius(0, Vec3(0, 0, 0), 0.5), std::vector<Entity>{e[0]});
    // Pending events and queued actions are delivered in the next step
    b.step();
    a.step();
    ASSERT_EQ(b.events.read<DamageEvent>().size(), 1u);
    EXPECT_EQ(b.events.read<DamageEvent>()[0].amount, 7);
    ASSERT_EQ(b.events.read<ActionEvent>().size(), 1u);
    EXPECT_EQ(b.events.read<ActionEvent>()[0].action, 3u);
}

TEST(Save, GameObjectsAreRecreatedWithConfigureAndOnLoad)
{
    World a(baseSetup());
    const Entity walker                          = a.spawn<Walker>(Vec3(0, 0, 0));
    a.get<WalkerState>(walker).speed             = 3;
    a.get<Behaviour>(walker).alwaysTickEveryStep = true;
    a.step(5);
    const auto blob = a.save();
    g_configures = g_onLoads = g_onSpawns = 0;
    World b(baseSetup());
    ASSERT_TRUE(b.tryLoad(blob).ok());
    EXPECT_EQ(g_configures, 1);
    EXPECT_EQ(g_onLoads, 1);
    EXPECT_EQ(g_onSpawns, 0);
    const Walker* o = static_cast<const Walker*>(b.read<Behaviour>(walker).object.get());
    ASSERT_TRUE(o);
    EXPECT_EQ(o->cache, 30);
    EXPECT_EQ(o->stride, 0.5);
    EXPECT_EQ(o->entity(), walker);
    // The Saved::No component was re-added by onLoad, and the save is unchanged by it
    EXPECT_TRUE(b.has<WalkerRender>(walker));
    EXPECT_EQ(b.save(), blob);
    // update continues from the saved state
    const int before = b.read<WalkerState>(walker).updates;
    b.step(3);
    EXPECT_EQ(b.read<WalkerState>(walker).updates, before + 3);
}

// A loaded world continues exactly like the original (the engine's systems and component-only game objects)
TEST(Save, LoadedWorldContinuesIdentically)
{
    World a(baseSetup());
    fill(a);
    for (int i = 0; i < 20; ++i) a.spawn<Walker>(Vec3(i * 7.0, 0, i * 3.0));
    a.step(10);
    World b(baseSetup());
    ASSERT_TRUE(b.tryLoad(a.save()).ok());
    for (int i = 0; i < 100; ++i)
    {
        a.step();
        b.step();
        ASSERT_EQ(a.hash(), b.hash()) << "step " << i << "\n" << World::diff(a, b);
    }
}

TEST(Save, ThrowingOnLoadRemovesThatObjectOnly)
{
    World a(baseSetup());
    const Entity fragile = a.spawn<Fragile>(Vec3(0, 0, 0));
    const Entity child   = a.create();
    a.add<Parent>(child, Parent{fragile});
    const Entity walker = a.spawn<Walker>(Vec3(1, 0, 0));
    g_throwOnLoad       = true;
    World b(baseSetup());
    const LoadResult r = b.tryLoad(a.save());
    g_throwOnLoad      = false;
    EXPECT_TRUE(r.committed);
    ASSERT_EQ(r.hookErrors.size(), 1u);
    EXPECT_EQ(r.hookErrors[0].entity, fragile);
    EXPECT_EQ(r.hookErrors[0].className, "Fragile");
    EXPECT_NE(r.hookErrors[0].message.find("fragile broke"), std::string::npos);
    EXPECT_FALSE(b.alive(fragile));
    EXPECT_FALSE(b.alive(child));
    EXPECT_TRUE(b.alive(walker));
    EXPECT_TRUE(b.read<Behaviour>(walker).object);
}

namespace
{
class SpawnsOnLoad final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(kFirstGameClass);
    static constexpr const char* name = "SpawnsOnLoad";
    void onLoad() override { add<Health>(); }
};
}  // namespace

#ifdef SAIGA_ASSERTS
// onLoad may only add Saved::No components of its own entity (a load followed by save() gives the saved bytes)
TEST(SaveDeathTest, StructuralChangesOfSavedComponentsInOnLoad)
{
    WorldSetup s;
    s.classes = [](ClassList& c) { c.add<SpawnsOnLoad>(); };
    World a(s);
    a.spawn<SpawnsOnLoad>(Vec3(0, 0, 0));
    const auto blob = a.save();
    EXPECT_DEATH(
        {
            World b(s);
            b.load(blob);
        },
        "during a load");
}
#endif

// ---------------------------------------------------------------------------------------------------------------------
// Builds that changed (two registrations of the same names in one test binary)

namespace v1
{
struct Item
{
    static constexpr const char* name = "Item";
    static constexpr uint32_t version = 1;
    std::string id;
    int32_t count = 0;
    template <class V>
    void visit(V& v)
    {
        v("id", id);
        v("count", count);
    }
};
struct Stats
{
    static constexpr const char* name = "Stats";
    static constexpr ComponentId id   = ComponentId(kFirstGameComponent + 4);
    static constexpr uint32_t version = 1;
    double hp                         = 0;
    int32_t level                     = 1;
    std::string title;  // removed in v2
    Mood mood = Mood::Calm;
    std::vector<Item> bag;
    template <class V>
    void visit(V& v)
    {
        v("hp", hp);
        v("level", level);
        v("title", title);
        v("mood", mood, EnumNames{kMoodNames});
        v("bag", bag);
    }
};
// Removed in v2 (several policies)
struct Old
{
    static constexpr const char* name = "Old";
    static constexpr ComponentId id   = ComponentId(kFirstGameComponent + 5);
    static constexpr uint32_t version = 1;
    double value                      = 0;
    template <class V>
    void visit(V& v)
    {
        v("value", value);
    }
};
void components(Registry& r)
{
    baseComponents(r);
    r.registerComponent<Stats>();
    r.registerComponent<Old>();
}
}  // namespace v1

namespace v2
{
enum class Mood
{
    Calm,
    Sleepy,  // added in the middle: by name
    Angry,
    Count
};
const char* const kMoodNames[] = {"Calm", "Sleepy", "Angry"};
struct Item
{
    static constexpr const char* name = "Item";
    static constexpr uint32_t version = 2;
    std::string id;
    int32_t amount = 0;  // renamed from count
    template <class V>
    void visit(V& v)
    {
        v("id", id);
        v("amount", amount);
    }
};
struct Stats
{
    static constexpr const char* name = "Stats";
    static constexpr ComponentId id   = ComponentId(kFirstGameComponent + 4);
    static constexpr uint32_t version = 2;
    double health                     = 0;  // hp * 10 (meaning changed)
    int32_t level                     = 1;
    double armor                      = 5;  // added: keeps its default
    Mood mood                         = Mood::Calm;
    std::vector<Item> bag;
    template <class V>
    void visit(V& v)
    {
        v("health", health);
        v("level", level);
        v("armor", armor);
        v("mood", mood, EnumNames{kMoodNames});
        v("bag", bag);
    }
};
// The replacement of Old
struct Fresh
{
    static constexpr const char* name = "Fresh";
    static constexpr ComponentId id   = ComponentId(kFirstGameComponent + 5);
    static constexpr uint32_t version = 1;
    double amount                     = 0;
    double other                      = -1;
    template <class V>
    void visit(V& v)
    {
        v("amount", amount);
        v("other", other);
    }
};
void components(Registry& r)
{
    baseComponents(r);
    r.registerComponent<Stats>();
    r.registerComponent<Fresh>();
}
void migrations(MigrationList& m)
{
    m.add<Stats>(1,
                 [](FieldMap& f)
                 {
                     f.set("health", Json(f.get("hp").number_value() * 10));
                     f.remove("hp");
                 });
    m.add<Item>(1, [](FieldMap& f) { f.rename("count", "amount"); });
    // The removed type's own migration runs before it is loaded as Fresh
    m.add("Old", 1, [](FieldMap& f) { f.rename("value", "amount"); });
    m.addRemoved("Old", RemovedPolicy::Replace, "Fresh");
}
}  // namespace v2

namespace
{
WorldSetup setupV1()
{
    WorldSetup s;
    s.components = v1::components;
    s.classes    = baseClasses;
    return s;
}
WorldSetup setupV2()
{
    WorldSetup s;
    s.components = v2::components;
    s.classes    = baseClasses;
    s.migrations = v2::migrations;
    return s;
}
}  // namespace

TEST(SaveMigration, ChangedComponentsAndNestedStructsAreMigrated)
{
    World a(setupV1());
    const Entity e = a.create();
    v1::Stats s;
    s.hp    = 4.5;
    s.level = 7;
    s.title = "gone";
    s.mood  = Mood::Angry;
    s.bag   = {v1::Item{"apple", 3}, v1::Item{"pear", 1}};
    a.add<v1::Stats>(e, s);
    const Entity o = a.create();
    a.add<v1::Old>(o, v1::Old{2.5});

    World b(setupV2());
    const LoadResult r = b.tryLoad(a.save());
    ASSERT_TRUE(r.ok()) << r.error;
    const v2::Stats& t = b.read<v2::Stats>(e);
    EXPECT_EQ(t.health, 45);
    EXPECT_EQ(t.level, 7);
    EXPECT_EQ(t.armor, 5);
    EXPECT_EQ(t.mood, v2::Mood::Angry);
    ASSERT_EQ(t.bag.size(), 2u);
    EXPECT_EQ(t.bag[0].id, "apple");
    EXPECT_EQ(t.bag[0].amount, 3);
    EXPECT_EQ(t.bag[1].amount, 1);
    // Replace after the removed type's own migration
    EXPECT_EQ(b.read<v2::Fresh>(o).amount, 2.5);
    EXPECT_EQ(b.read<v2::Fresh>(o).other, -1);
}

namespace
{
// v3: Stats version 3 without a migration from 2
struct StatsV3
{
    static constexpr const char* name = "Stats";
    static constexpr ComponentId id   = ComponentId(kFirstGameComponent + 4);
    static constexpr uint32_t version = 3;
    double health                     = 0;
    template <class V>
    void visit(V& v)
    {
        v("health", health);
    }
};
// The same version as v1 but a field changed its type (R-SAVE-1 violated)
struct StatsTypeChanged
{
    static constexpr const char* name = "Stats";
    static constexpr ComponentId id   = ComponentId(kFirstGameComponent + 4);
    static constexpr uint32_t version = 1;
    int64_t hp                        = 0;
    template <class V>
    void visit(V& v)
    {
        v("hp", hp);
    }
};
// The same version as v1, the mood enum lost "Angry"
enum class MoodShort
{
    Calm,
    Count
};
const char* const kMoodShortNames[] = {"Calm"};
struct StatsLostEnum
{
    static constexpr const char* name = "Stats";
    static constexpr ComponentId id   = ComponentId(kFirstGameComponent + 4);
    static constexpr uint32_t version = 1;
    double hp                         = 0;
    MoodShort mood                    = MoodShort::Calm;
    template <class V>
    void visit(V& v)
    {
        v("hp", hp);
        v("mood", mood, EnumNames{kMoodShortNames});
    }
};

template <class S>
LoadResult loadInto(const std::vector<uint8_t>& blob, void (*migrations)(MigrationList&) = nullptr)
{
    WorldSetup s;
    s.components = [](Registry& r)
    {
        baseComponents(r);
        r.registerComponent<S>();
        r.registerComponent<v1::Old>();
    };
    s.classes    = baseClasses;
    s.migrations = migrations;
    World w(s);
    const auto before = w.save();
    LoadResult r      = w.tryLoad(blob);
    if (!r.committed)
    {
        EXPECT_EQ(w.save(), before) << "a rejected load changed the world";
    }
    return r;
}

std::vector<uint8_t> v1Blob(Mood mood = Mood::Calm)
{
    World a(setupV1());
    const Entity e = a.create();
    v1::Stats s;
    s.hp   = 2;
    s.mood = mood;
    a.add<v1::Stats>(e, s);
    return a.save();
}
}  // namespace

TEST(SaveMigration, IncompatibleChangesAreRejected)
{
    LoadResult r = loadInto<StatsV3>(v1Blob(),
                                     [](MigrationList& m)
                                     {
                                         m.add("Stats", 1, [](FieldMap& f) { f.rename("hp", "health"); });
                                         m.add("Item", 1, [](FieldMap&) {});
                                     });
    EXPECT_FALSE(r.committed);
    EXPECT_NE(r.error.find("no migration from version 2"), std::string::npos) << r.error;

    r = loadInto<StatsTypeChanged>(v1Blob());
    EXPECT_FALSE(r.committed);
    EXPECT_NE(r.error.find("Stats.hp changed its type"), std::string::npos) << r.error;

    // A newer save than the build
    World a(setupV2());
    a.add<v2::Stats>(a.create());
    WorldSetup old = setupV1();
    World b(old);
    r = b.tryLoad(a.save());
    EXPECT_FALSE(r.committed);
    EXPECT_NE(r.error.find("newer than the registered version"), std::string::npos) << r.error;

    // An enum value that no longer exists: an error only when a value uses it
    EXPECT_TRUE(loadInto<StatsLostEnum>(v1Blob(Mood::Calm)).committed);
    r = loadInto<StatsLostEnum>(v1Blob(Mood::Angry));
    EXPECT_FALSE(r.committed);
    EXPECT_NE(r.error.find("Angry"), std::string::npos) << r.error;
}

TEST(SaveMigration, RemovedTypes)
{
    auto blobWithOld = []()
    {
        World a(setupV1());
        const Entity parent = a.create();
        a.add<v1::Old>(parent, v1::Old{1});
        const Entity child = a.create();
        a.add<Parent>(child, Parent{parent});
        const Entity other = a.create();
        a.add<Health>(other);
        return std::make_tuple(a.save(), parent, child, other);
    };
    auto [blob, parent, child, other] = blobWithOld();

    // Not registered and no entry: an error naming it
    {
        WorldSetup s;
        s.components = baseComponents;
        s.classes    = baseClasses;
        World w(s);
        const LoadResult r = w.tryLoad(blob);
        EXPECT_FALSE(r.committed);
        EXPECT_NE(r.error.find("Old"), std::string::npos) << r.error;
    }
    // DropComponent
    {
        WorldSetup s;
        s.components = baseComponents;
        s.classes    = baseClasses;
        s.migrations = [](MigrationList& m) { m.addRemoved("Old", RemovedPolicy::DropComponent); };
        World w(s);
        ASSERT_TRUE(w.tryLoad(blob).ok());
        EXPECT_TRUE(w.alive(parent));
        EXPECT_TRUE(w.alive(child));
    }
    // DropEntity: with its descendants; their handles are stale, their slots reused in index order
    {
        WorldSetup s;
        s.components = baseComponents;
        s.classes    = baseClasses;
        s.migrations = [](MigrationList& m) { m.addRemoved("Old", RemovedPolicy::DropEntity); };
        World w(s);
        ASSERT_TRUE(w.tryLoad(blob).ok());
        EXPECT_FALSE(w.alive(parent));
        EXPECT_FALSE(w.alive(child));
        EXPECT_TRUE(w.alive(other));
        // Deterministic: a second load gives the same world
        World w2(s);
        ASSERT_TRUE(w2.tryLoad(blob).ok());
        EXPECT_EQ(w.save(), w2.save());
    }
}

TEST(SaveMigration, RemovedClassesAndSavedNoComponents)
{
    World a(baseSetup());
    const Entity sitter = a.spawn<Sitter>(Vec3(0, 0, 0));
    const Entity walker = a.spawn<Walker>(Vec3(1, 0, 0));
    const auto blob     = a.save();
    // Classes renumbered: Sitter is now the first class and Walker replaces nothing
    {
        WorldSetup s = baseSetup();
        s.classes    = [](ClassList& c)
        {
            c.add<SpawnsOnLoad>();  // takes Walker's ID 64
        };
        s.migrations = [](MigrationList& m)
        {
            m.addRemoved("Walker", RemovedPolicy::DropEntity);
            m.addRemoved("Sitter", RemovedPolicy::DropComponent);
            m.addRemoved("Fragile", RemovedPolicy::DropComponent);
        };
        World w(s);
        ASSERT_TRUE(w.tryLoad(blob).ok());
        EXPECT_FALSE(w.alive(walker));
        EXPECT_TRUE(w.alive(sitter));
        EXPECT_FALSE(w.has<Behaviour>(sitter));
    }
    // Replace a class: Sitter objects become Loungers (another ID)
    {
        WorldSetup s = baseSetup();
        s.classes    = [](ClassList& c)
        {
            c.add<Walker>();
            c.add<Lounger>();
        };
        s.migrations = [](MigrationList& m) { m.addRemoved("Sitter", RemovedPolicy::Replace, "Lounger"); };
        World w(s);
        ASSERT_TRUE(w.tryLoad(blob).ok());
        EXPECT_EQ(w.read<Behaviour>(sitter).classId, Lounger::id);
        EXPECT_TRUE(dynamic_cast<const Lounger*>(w.read<Behaviour>(sitter).object.get()));
        EXPECT_TRUE(dynamic_cast<const Walker*>(w.read<Behaviour>(walker).object.get()));
    }
    // A component that is now Saved::No is dropped silently
    {
        WorldSetup s;
        s.components = [](Registry& r)
        {
            r.registerGlobal<Weather>(Replicate::Host);
            r.registerComponent<Link>(Replicate::Never, Saved::No);
            r.registerComponent<WalkerState>();
            r.registerComponent<WalkerRender>(Replicate::Never, Saved::No);
        };
        s.classes = baseClasses;
        World src(baseSetup());
        const Entity e = src.create();
        src.add<Link>(e);
        World w(s);
        ASSERT_TRUE(w.tryLoad(src.save()).ok());
        EXPECT_FALSE(w.has<Link>(e));
    }
}

namespace
{
struct Ping
{
    static constexpr const char* name = "Ping";
    static constexpr EventId id       = EventId(64);
    static constexpr uint32_t version = 1;
    int32_t n                         = 0;
    template <class V>
    void visit(V& v)
    {
        v("n", n);
    }
};
}  // namespace

TEST(SaveMigration, PendingEventsOfRemovedTypesAreDropped)
{
    WorldSetup s = baseSetup();
    s.events     = [](EventRegistry& r) { r.add<Ping>(); };
    World a(s);
    a.events.emit(Ping{5});
    a.events.emit(DamageEvent{});
    World b(baseSetup());
    ASSERT_TRUE(b.tryLoad(a.save()).ok());
    b.step();
    EXPECT_EQ(b.events.read<DamageEvent>().size(), 1u);
    World c(s);
    ASSERT_TRUE(c.tryLoad(a.save()).ok());
    c.step();
    ASSERT_EQ(c.events.read<Ping>().size(), 1u);
    EXPECT_EQ(c.events.read<Ping>()[0].n, 5);
}

// ---------------------------------------------------------------------------------------------------------------------
// Rejected saves leave the world unchanged

namespace
{
// Removed types replaced by engine components let a test build saves that break the engine's rules
struct FakeParent
{
    static constexpr const char* name = "FakeParent";
    static constexpr ComponentId id   = ComponentId(kFirstGameComponent + 4);
    static constexpr uint32_t version = 1;
    Entity entity;
    template <class V>
    void visit(V& v)
    {
        v("entity", entity);
    }
};
struct FakePersistent
{
    static constexpr const char* name = "FakePersistent";
    static constexpr ComponentId id   = ComponentId(kFirstGameComponent + 5);
    static constexpr uint32_t version = 1;
    PersistentId persistentId;
    template <class V>
    void visit(V& v)
    {
        v("persistentId", persistentId);
    }
};
WorldSetup fakeSetup()
{
    WorldSetup s = baseSetup();
    s.components = [](Registry& r)
    {
        baseComponents(r);
        r.registerComponent<FakeParent>();
        r.registerComponent<FakePersistent>();
    };
    return s;
}
LoadResult loadReplacingFakes(const std::vector<uint8_t>& blob)
{
    WorldSetup s = baseSetup();
    s.migrations = [](MigrationList& m)
    {
        m.addRemoved("FakeParent", RemovedPolicy::Replace, "Parent");
        m.addRemoved("FakePersistent", RemovedPolicy::Replace, "Persistent");
    };
    World w(s);
    fill(w);
    const auto before = w.save();
    LoadResult r      = w.tryLoad(blob);
    if (!r.committed)
    {
        EXPECT_EQ(w.save(), before) << "a rejected load changed the world";
    }
    return r;
}
}  // namespace

TEST(SaveRobustness, BrokenRulesAreRejected)
{
    // A Parent cycle
    {
        World a(fakeSetup());
        const Entity x = a.create(), y = a.create();
        a.add<FakeParent>(x, FakeParent{y});
        a.add<FakeParent>(y, FakeParent{x});
        const LoadResult r = loadReplacingFakes(a.save());
        EXPECT_FALSE(r.committed);
        EXPECT_NE(r.error.find("cycle"), std::string::npos) << r.error;
    }
    // A Parent naming a dead entity
    {
        World a(fakeSetup());
        const Entity x = a.create(), y = a.create();
        a.destroy(y);
        a.add<FakeParent>(x, FakeParent{y});
        const LoadResult r = loadReplacingFakes(a.save());
        EXPECT_FALSE(r.committed);
        EXPECT_NE(r.error.find("not alive"), std::string::npos) << r.error;
    }
    // A duplicate PersistentId, and one above the counter
    {
        World a(fakeSetup());
        const Entity x = a.create(), y = a.create();
        a.makePersistent(a.create());
        a.makePersistent(a.create());  // the counter is 2
        a.add<FakePersistent>(x, FakePersistent{PersistentId{1}});
        a.add<FakePersistent>(y, FakePersistent{PersistentId{1}});
        LoadResult r = loadReplacingFakes(a.save());
        EXPECT_FALSE(r.committed);
        EXPECT_NE(r.error.find("used twice"), std::string::npos) << r.error;
        World c(fakeSetup());
        c.add<FakePersistent>(c.create(), FakePersistent{PersistentId{5}});
        r = loadReplacingFakes(c.save());
        EXPECT_FALSE(r.committed);
        EXPECT_NE(r.error.find("above the saved counter"), std::string::npos) << r.error;
    }
    // A corrupt slot table (an invalid state) and a free list naming an alive slot
    {
        World a(baseSetup());
        fill(a);
        auto blob        = a.save();
        const size_t ent = sectionOffset(blob, kTagEntities);
        ASSERT_GT(ent, 0u);
        auto bad     = blob;
        bad[ent + 8] = 7;  // slot 1: u32 count, then generation u32, state u8
        World w(baseSetup());
        EXPECT_FALSE(w.tryLoad(bad).committed);
    }
}

TEST(SaveRobustness, TruncatedAndCorruptedSavesNeverChangeTheWorld)
{
    World a(baseSetup());
    fill(a);
    const auto blob = a.save();
    ASSERT_LT(blob.size(), 16384u);
    World w(baseSetup());
    w.spawn<Walker>(Vec3(1, 2, 3));
    const auto before = w.save();
    for (size_t n = 0; n < blob.size(); ++n)
    {
        std::vector<uint8_t> cut(blob.begin(), blob.begin() + std::ptrdiff_t(n));
        const LoadResult r = w.tryLoad(cut);
        ASSERT_FALSE(r.committed) << "truncated at " << n;
        ASSERT_EQ(w.save(), before) << "truncated at " << n;
    }
    // Flipped bytes: rejected (the world unchanged) or accepted as another valid world, never a crash
    uint64_t seed = 12345;
    for (int k = 0; k < 500; ++k)
    {
        seed           = seed * 6364136223846793005ull + 1442695040888963407ull;
        auto bad       = blob;
        const size_t i = size_t((seed >> 33) % bad.size());
        bad[i] ^= uint8_t(1u << ((seed >> 20) % 8));
        World x(baseSetup());
        x.spawn<Walker>(Vec3(1, 2, 3));
        const LoadResult r = x.tryLoad(bad);
        if (!r.committed)
        {
            ASSERT_EQ(x.save(), before) << "byte " << i;
        }
    }
    // An unknown section after the last one
    auto extra                = blob;
    const uint8_t section[12] = {'X', 'T', 'R', 'A', 0, 0, 0, 0, 0, 0, 0, 0};
    extra.insert(extra.end(), std::begin(section), std::end(section));
    EXPECT_FALSE(w.tryLoad(extra).committed);
    EXPECT_FALSE(w.tryLoad({}).committed);
}

// ---------------------------------------------------------------------------------------------------------------------
// State entities, time, listeners

TEST(Save, StateEntitiesFollowTheRegistrations)
{
    // A save without globals into a build with globals: the world-state entity is created with the defaults
    WorldSetup none;
    World a(none);
    a.create();
    World b(baseSetup());
    ASSERT_TRUE(b.tryLoad(a.save()).ok());
    ASSERT_TRUE(b.worldStateEntity().valid());
    EXPECT_EQ(b.readGlobal<Weather>().rain, 0);
    // The reverse: the world-state entity is freed
    World c(baseSetup());
    c.global<Weather>().rain = 1;
    WorldSetup dropWeather;
    // Types without saved values need no entry
    dropWeather.migrations = [](MigrationList& m) { m.addRemoved("Weather", RemovedPolicy::DropComponent); };
    World e(dropWeather);
    ASSERT_TRUE(e.tryLoad(c.save()).ok());
    EXPECT_FALSE(e.worldStateEntity().valid());
    EXPECT_EQ(e.registry().aliveCount(), 0u);
}

TEST(Save, AnotherStepRateKeepsTheTime)
{
    World a(baseSetup());
    a.step(90);  // 1.5 s at 60 Hz
    WorldSetup s            = baseSetup();
    s.config.stepsPerSecond = 30;
    World b(s);
    ASSERT_TRUE(b.tryLoad(a.save()).ok());
    EXPECT_DOUBLE_EQ(b.time(), 1.5);
    b.step(3);
    EXPECT_DOUBLE_EQ(b.time(), 1.6);
}

TEST(Save, LoadListenersGetTheLogRangeOfTheLoad)
{
    World a(baseSetup());
    fill(a);
    const auto blob = a.save();
    World w(baseSetup());
    StructuralLog& log  = w.structuralLog();
    const int consumer  = log.addConsumer();
    const Entity before = w.create();  // a change before the load: must not be mistaken for the load's
    LoadInfo got;
    int calls = 0;
    w.addLoadListener(
        [&](const LoadInfo& i)
        {
            got = i;
            calls++;
        });
    ASSERT_TRUE(w.tryLoad(blob).ok());
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(got.loadCount, 1u);
    EXPECT_EQ(got.toStep, a.stepIndex());
    EXPECT_EQ(log.cursor(consumer) + 1, got.logBegin);  // exactly the create before it
    EXPECT_EQ(got.logEnd, log.appendedTotal());
    std::vector<StructuralLog::Record> records;
    log.consume(consumer, [&](const StructuralLog::Record& r) { records.push_back(r); });
    ASSERT_FALSE(records.empty());
    EXPECT_EQ(records[0].entity, before);
    EXPECT_TRUE(records[0].added);
}

namespace
{
std::vector<uint8_t> g_loaderBlob;
class Loader final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(kFirstGameClass);
    static constexpr const char* name = "Loader";
    void onSpawn() override { world().load(g_loaderBlob); }
};
}  // namespace

#ifdef SAIGA_ASSERTS
TEST(SaveDeathTest, LoadInsideAGameHook)
{
    WorldSetup s;
    s.classes = [](ClassList& c) { c.add<Loader>(); };
    World a(s);
    g_loaderBlob = a.save();
    EXPECT_DEATH(a.spawn<Loader>(Vec3(0, 0, 0)), "inside a game hook");
}
#endif

// ---------------------------------------------------------------------------------------------------------------------
// Comparison and dumps

TEST(SaveDiff, NamesEveryKindOfDifference)
{
    World a(baseSetup());
    std::vector<Entity> e = fill(a);
    const auto blobA      = a.save();
    World b(baseSetup());
    ASSERT_TRUE(b.tryLoad(blobA).ok());
    b.get<Health>(e[0]).hp = 49;
    b.remove<Player>(e[1]);
    b.destroy(e[4]);
    b.random().nextU32();
    const SaveDiff d = diffSaves(blobA, b.save());
    ASSERT_TRUE(d.error.empty()) << d.error;
    auto has = [&](const std::string& kind, const std::string& component = "")
    {
        for (const SaveDifference& x : d.differences)
            if (x.kind == kind && (component.empty() || x.component == component)) return true;
        return false;
    };
    EXPECT_TRUE(has("field", "Health"));
    EXPECT_TRUE(has("component_only_in_a", "Player"));
    EXPECT_TRUE(has("only_in_a"));
    EXPECT_TRUE(has("rng"));
    EXPECT_TRUE(has("free_list"));
    for (const SaveDifference& x : d.differences)
        if (x.kind == "field")
        {
            EXPECT_EQ(x.describe(), "entity " + std::to_string(e[0].index) + ":0 Health.hp: 50 != 49");
        }
    EXPECT_EQ(d.summary["components"]["Health"].number_value(), 1);
    // Filters and paging
    DiffFilter f;
    f.components = {"Health"};
    EXPECT_EQ(diffSaves(blobA, b.save(), f).total, 1u);
    DiffFilter page;
    page.limit       = 1;
    page.after       = 1;
    const SaveDiff p = diffSaves(blobA, b.save(), page);
    EXPECT_EQ(p.differences.size(), 1u);
    EXPECT_EQ(p.total, d.total);
    // World::diff: one line per difference, then "... and N more"
    const std::string text = World::diff(a, b, 2);
    EXPECT_NE(text.find("... and"), std::string::npos) << text;
    EXPECT_EQ(World::diff(a, a), "");
}

TEST(SaveDiff, DumpJson)
{
    World a(baseSetup());
    std::vector<Entity> e = fill(a);
    std::string error;
    const Json one = Json::parse(a.dumpJson(e[0]), error);
    ASSERT_TRUE(error.empty());
    ASSERT_EQ(one["entities"].array_items().size(), 1u);
    EXPECT_EQ(one["entities"][0]["components"]["Health"]["hp"].number_value(), 50);
    const Json all = Json::parse(a.dumpJson(), error);
    EXPECT_EQ(all["entities"].array_items().size(), size_t(a.registry().aliveCount()));
    bool walker = false;
    for (auto& x : all["entities"].array_items()) walker = walker || x["class"].string_value() == "Walker";
    EXPECT_TRUE(walker);
}

// ---------------------------------------------------------------------------------------------------------------------
// Format stability across builds (committed files; regenerate with SAIGA_UPDATE_GOLDEN=1 after a deliberate change)

namespace
{
const std::string kGoldenDir = std::string(SAIGA_PROJECT_SOURCE_DIR) + "/data/tests/engine/";
bool updateGolden()
{
    const char* v = std::getenv("SAIGA_UPDATE_GOLDEN");
    return v && std::string(v) == "1";
}
std::string readFile(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
void writeFile(const std::string& path, const std::string& data)
{
    std::ofstream f(path, std::ios::binary);
    f << data;
}
// The engine's types only (a world without game registrations), so the golden files change only with the engine
Json engineSchemaJson()
{
    World w{WorldSetup{}};
    const SchemaTable t = SchemaTable::fromWorld(w);
    Json::array types;
    for (const SchemaType& s : t.types)
    {
        Json::array fields;
        for (const FieldInfo& f : s.fields)
        {
            Json::array values;
            for (auto& v : f.enumValues) values.push_back(v);
            fields.push_back(Json::object{{"name", f.name}, {"type", f.type}, {"values", values}});
        }
        types.push_back(
            Json::object{{"kind", int(s.kind)}, {"name", s.name}, {"version", int(s.version)}, {"fields", fields}});
    }
    return types;
}
void goldenWorld(World& w)
{
    const Entity a = w.create();
    w.get<Transform>(a).setPosition(Vec3(1.25, -2, 3e6));
    w.add<Health>(a, 12.5, 40.0);
    const Entity p = w.create();
    w.add<Player>(p);
    w.add<Parent>(p, Parent{a});
    w.add<LocalTransform>(p, LocalTransform{Vec3(0, 2, 0), Quat::Identity(), Vec3::Ones()});
    w.makePersistent(a);
    w.events.emit(DamageEvent{a, p, 3});
    w.step(2);
}
}  // namespace

// R-SAVE-1 in CI: an engine type whose fields change without a version increment fails here
TEST(SaveFormat, EngineSchemaIsCommitted)
{
    const std::string path = kGoldenDir + "schema.json";
    const Json current     = engineSchemaJson();
    if (updateGolden())
    {
        writeFile(path, current.dump());
        GTEST_SKIP() << "wrote " << path;
    }
    std::string error;
    const Json committed = Json::parse(readFile(path), error);
    ASSERT_TRUE(error.empty()) << path << ": " << error;
    for (const Json& c : current.array_items())
        for (const Json& k : committed.array_items())
            if (c["name"] == k["name"] && c["kind"] == k["kind"])
            {
                if (c["version"] == k["version"])
                    EXPECT_EQ(c["fields"], k["fields"])
                        << c["name"].string_value()
                        << " changed its fields without a version increment (R-SAVE-1: increment the version, add a "
                           "migration, then regenerate data/tests/engine/schema.json with SAIGA_UPDATE_GOLDEN=1)";
                else
                    ADD_FAILURE() << c["name"].string_value()
                                  << " has a new version: regenerate data/tests/engine/schema.json with "
                                     "SAIGA_UPDATE_GOLDEN=1 (and keep save_v1.bin loading)";
            }
}

// A save written by an earlier build still loads (and means the same)
TEST(SaveFormat, CommittedSaveLoads)
{
    const std::string blobPath = kGoldenDir + "save_v1.bin", dumpPath = kGoldenDir + "save_v1.json";
    if (updateGolden())
    {
        World w{WorldSetup{}};
        goldenWorld(w);
        const auto b = w.save();
        writeFile(blobPath, std::string(b.begin(), b.end()));
        writeFile(dumpPath, w.dumpJson());
        GTEST_SKIP() << "wrote " << blobPath;
    }
    const std::string data = readFile(blobPath);
    ASSERT_FALSE(data.empty()) << blobPath;
    World w{WorldSetup{}};
    const LoadResult r = w.tryLoad(std::vector<uint8_t>(data.begin(), data.end()));
    ASSERT_TRUE(r.ok()) << r.error;
    std::string error;
    const Json expected = Json::parse(readFile(dumpPath), error);
    ASSERT_TRUE(error.empty()) << error;
    EXPECT_EQ(Json::parse(w.dumpJson(), error), expected);
}

// ---------------------------------------------------------------------------------------------------------------------
// Review regressions (S11 round 1)

namespace
{
// Replaces the content of a section of a save (to build malformed saves)
std::vector<uint8_t> replaceSection(const std::vector<uint8_t>& data, uint32_t tag, const std::vector<uint8_t>& content)
{
    const size_t at = sectionOffset(data, tag);
    EXPECT_GT(at, 0u);
    uint64_t len;
    std::memcpy(&len, data.data() + at - 8, 8);
    std::vector<uint8_t> out(data.begin(), data.begin() + std::ptrdiff_t(at - 8));
    const uint64_t n = content.size();
    for (int i = 0; i < 8; ++i) out.push_back(uint8_t(n >> (8 * i)));
    out.insert(out.end(), content.begin(), content.end());
    out.insert(out.end(), data.begin() + std::ptrdiff_t(at + len), data.end());
    return out;
}
SchemaTable readSchema(const std::vector<uint8_t>& data)
{
    const size_t at = sectionOffset(data, kTagSchema);
    uint64_t len;
    std::memcpy(&len, data.data() + at - 8, 8);
    BinaryReader r(data.data() + at, size_t(len));
    SchemaTable t;
    std::string error;
    EXPECT_TRUE(t.read(r, error)) << error;
    return t;
}
std::vector<uint8_t> withSchema(const std::vector<uint8_t>& data, const SchemaTable& t)
{
    std::vector<uint8_t> content;
    BinaryWriter w(content);
    t.write(w);
    return replaceSection(data, kTagSchema, content);
}
// A save whose Weather type got an extra field of the given type (same version: the slow path decodes it)
std::vector<uint8_t> weatherWithField(const std::string& type)
{
    World a(baseSetup());
    a.global<Weather>().rain = 1;
    const auto blob          = a.save();
    SchemaTable t            = readSchema(blob);
    for (SchemaType& s : t.types)
        if (s.name == "Weather") s.fields.push_back(FieldInfo{"x", type, {}});
    return withSchema(blob, t);
}
}  // namespace

// A deeply nested container type in a crafted schema overflowed the stack; a nest of zero-byte arrays ran for minutes
TEST(SaveRobustness, MaliciousFieldTypesAreRejected)
{
    World w(baseSetup());
    std::string deep;
    for (int i = 0; i < 20; ++i) deep += "array<";
    deep += "int8,0>";
    for (int i = 0; i < 19; ++i) deep += ",1>";
    LoadResult r = w.tryLoad(weatherWithField(deep));
    EXPECT_FALSE(r.committed);
    EXPECT_NE(r.error.find("malformed field type"), std::string::npos) << r.error;
    EXPECT_FALSE(w.tryLoad(weatherWithField(std::string(300, 'x'))).committed);
    // Unbalanced brackets
    EXPECT_FALSE(w.tryLoad(weatherWithField("array<int8,0>,1>")).committed);
    // Zero-byte elements in nested arrays: bounded work, rejected
    const auto start = std::chrono::steady_clock::now();
    r                = w.tryLoad(weatherWithField("array<array<array<array<int8,0>,1000>,1000>,1000>"));
    EXPECT_FALSE(r.committed);
    EXPECT_NE(r.error.find("without bytes"), std::string::npos) << r.error;
    const auto blob = weatherWithField("array<array<array<array<int8,0>,1000>,1000>,1000>");
    World ok(baseSetup());
    EXPECT_FALSE(diffSaves(ok.save(), blob).error.empty());
    EXPECT_LT(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), 10.0);
}

// A schema with many types was checked for duplicates quadratically
TEST(SaveRobustness, LargeSchemasAreReadInLinearTime)
{
    World a(baseSetup());
    const auto blob = a.save();
    SchemaTable t   = readSchema(blob);
    for (int i = 0; i < 50000; ++i)
        t.types.push_back(SchemaType{SchemaKind::Struct, "Extra" + std::to_string(i), 0, 1, {}});
    const auto big   = withSchema(blob, t);
    const auto start = std::chrono::steady_clock::now();
    World w(baseSetup());
    EXPECT_TRUE(w.tryLoad(big).ok());
    EXPECT_LT(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), 2.0);
}

// An even RNG increment passed every check and asserted in the middle of the commit
TEST(SaveRobustness, AnEvenRngIncrementIsRejected)
{
    World a(baseSetup());
    fill(a);
    auto blob       = a.save();
    const size_t at = sectionOffset(blob, kTagState);
    blob[at + 8] &= uint8_t(~1u);  // the increment follows the state (u64)
    World w(baseSetup());
    const auto before  = w.save();
    const LoadResult r = w.tryLoad(blob);
    EXPECT_FALSE(r.committed);
    EXPECT_NE(r.error.find("increment"), std::string::npos) << r.error;
    EXPECT_EQ(w.save(), before);
}

// A freed state entity left its children with a dead parent; a game object on the world-state entity could destroy it
TEST(SaveRobustness, StateEntitiesWithChildrenOrGameObjects)
{
    {
        World a(baseSetup());
        const Entity child = a.create();
        a.add<Parent>(child, Parent{a.worldStateEntity()});
        WorldSetup s;
        s.migrations = [](MigrationList& m) { m.addRemoved("Weather", RemovedPolicy::DropComponent); };
        World w(s);
        const LoadResult r = w.tryLoad(a.save());
        EXPECT_FALSE(r.committed);
        EXPECT_NE(r.error.find("would be freed"), std::string::npos) << r.error;
    }
    {
        World a(baseSetup());
        Behaviour b;
        b.classId = Walker::id;
        a.registry().add<Behaviour>(a.worldStateEntity(), std::move(b));
        World w(baseSetup());
        const LoadResult r = w.tryLoad(a.save());
        EXPECT_FALSE(r.committed);
        EXPECT_NE(r.error.find("world-state entity has a game object"), std::string::npos) << r.error;
    }
}

// time() after a load drifted from steps * dt (the saved time as the base rounds differently)
TEST(Save, TheTimeContinuesBitForBit)
{
    World a(baseSetup());
    a.step(7);
    World b(baseSetup());
    ASSERT_TRUE(b.tryLoad(a.save()).ok());
    for (int i = 0; i < 100; ++i)
    {
        a.step();
        b.step();
        ASSERT_EQ(a.time(), b.time()) << "step " << i;
        ASSERT_EQ(a.save(), b.save()) << "step " << i;
    }
}

namespace v1b
{
// Item at the same version with a field of another type (R-SAVE-1 violated in a nested struct)
struct Item
{
    static constexpr const char* name = "Item";
    static constexpr uint32_t version = 1;
    std::string id;
    double count = 0;
    template <class V>
    void visit(V& v)
    {
        v("id", id);
        v("count", count);
    }
};
struct Stats
{
    static constexpr const char* name = "Stats";
    static constexpr ComponentId id   = ComponentId(kFirstGameComponent + 4);
    static constexpr uint32_t version = 1;
    std::vector<Item> bag;
    template <class V>
    void visit(V& v)
    {
        v("bag", bag);
    }
};
}  // namespace v1b

TEST(SaveMigration, NestedTypeChangesWithoutAVersionAreRejected)
{
    World a(setupV1());
    v1::Stats s;
    s.bag = {v1::Item{"apple", 3}};
    a.add<v1::Stats>(a.create(), s);
    const LoadResult r = loadInto<v1b::Stats>(a.save());
    EXPECT_FALSE(r.committed);
    EXPECT_NE(r.error.find("Item.count changed its type"), std::string::npos) << r.error;
}
