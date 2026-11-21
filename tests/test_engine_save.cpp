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

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
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

// onLoad may only add Saved::No components of its own entity (a load followed by save() gives the saved bytes): a hook
// error in every build (an assert stopped the program before the load could report it)
TEST(Save, StructuralChangesOfSavedComponentsInOnLoad)
{
    WorldSetup s;
    s.classes = [](ClassList& c) { c.add<SpawnsOnLoad>(); };
    World a(s);
    const Entity e = a.spawn<SpawnsOnLoad>(Vec3(0, 0, 0));
    World b(s);
    const LoadResult r = b.tryLoad(a.save());
    ASSERT_TRUE(r.committed);
    ASSERT_EQ(r.hookErrors.size(), 1u);
    EXPECT_NE(r.hookErrors[0].message.find("during a load"), std::string::npos) << r.hookErrors[0].message;
    EXPECT_FALSE(b.alive(e));
}

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

// S21: the golden save (no SPCE) loads into the golden setup with two appended templates (the prefix rule; no
// space-state components in the golden setup, so no new entities)
TEST(SaveFormat, CommittedSaveLoadsWithAppendedSpaces)
{
    const std::string blobPath = kGoldenDir + "save_v1.bin", dumpPath = kGoldenDir + "save_v1.json";
    const std::string data = readFile(blobPath);
    ASSERT_FALSE(data.empty()) << blobPath;
    WorldSetup s;
    s.config.spaceTemplates = {SpaceTemplate(), SpaceTemplate::enclosed("cellar"), SpaceTemplate::enclosed("cave")};
    World w(s);
    const LoadResult r = w.tryLoad(std::vector<uint8_t>(data.begin(), data.end()));
    ASSERT_TRUE(r.ok()) << r.error;
    std::string error;
    const Json expected = Json::parse(readFile(dumpPath), error);
    ASSERT_TRUE(error.empty()) << error;
    EXPECT_EQ(Json::parse(w.dumpJson(), error), expected);
}

// ---------------------------------------------------------------------------------------------------------------------
// Optional sections (S21, D7)

namespace
{
std::vector<uint8_t> replaceSection(const std::vector<uint8_t>& data, uint32_t tag,
                                    const std::vector<uint8_t>& content);  // below
WorldSetup spacesSaveSetup(std::vector<SpaceTemplate> templates)
{
    WorldSetup s            = baseSetup();
    s.config.spaceTemplates = std::move(templates);
    return s;
}
std::vector<SpaceTemplate> cellarSpaces()
{
    return {SpaceTemplate(), SpaceTemplate::enclosed("cellar")};
}
// A SPCE content: the version, then the rows
std::vector<uint8_t> spaceSection(uint32_t version, const std::vector<std::pair<std::string, uint8_t>>& rows)
{
    std::vector<uint8_t> out;
    BinaryWriter w(out);
    w.write(version);
    w.write(uint32_t(rows.size()));
    for (const auto& r : rows)
    {
        w.write(uint32_t(r.first.size()));
        for (char c : r.first) w.write(uint8_t(c));
        w.write(r.second);
    }
    return out;
}
// Inserts a section (tag, content) before the section `before`
std::vector<uint8_t> insertSection(const std::vector<uint8_t>& data, uint32_t before, uint32_t tag,
                                   const std::vector<uint8_t>& content)
{
    const size_t at = sectionOffset(data, before);
    EXPECT_GT(at, 0u);
    std::vector<uint8_t> out(data.begin(), data.begin() + std::ptrdiff_t(at - 12));
    BinaryWriter w(out);
    w.write(tag);
    w.write(uint64_t(content.size()));
    out.insert(out.end(), content.begin(), content.end());
    out.insert(out.end(), data.begin() + std::ptrdiff_t(at - 12), data.end());
    return out;
}
// Removes a section
std::vector<uint8_t> removeSection(const std::vector<uint8_t>& data, uint32_t tag)
{
    const size_t at = sectionOffset(data, tag);
    EXPECT_GT(at, 0u);
    uint64_t len;
    std::memcpy(&len, data.data() + at - 8, 8);
    std::vector<uint8_t> out(data.begin(), data.begin() + std::ptrdiff_t(at - 12));
    out.insert(out.end(), data.begin() + std::ptrdiff_t(at + len), data.end());
    return out;
}
}  // namespace

TEST(SaveSections, OptionalAbsentAndPresent)
{
    World one(baseSetup());
    ParsedSave p;
    std::string error;
    ASSERT_TRUE(parseSave(one.save(), p, error)) << error;
    EXPECT_FALSE(p.spaces.has_value());
    World two(spacesSaveSetup(cellarSpaces()));
    const auto blob = two.save();
    EXPECT_GT(sectionOffset(blob, kTagSpaces), 0u);
    ASSERT_TRUE(parseSave(blob, p, error)) << error;
    ASSERT_TRUE(p.spaces.has_value());
    ASSERT_EQ(p.spaces->rows.size(), 2u);
    EXPECT_EQ(p.spaces->rows[0].name, "overworld");
    EXPECT_EQ(p.spaces->rows[0].kind, SpaceKind::OpenWorld);
    EXPECT_EQ(p.spaces->rows[1].name, "cellar");
    EXPECT_EQ(p.spaces->rows[1].kind, SpaceKind::Enclosed);
    // A hand-inserted SPCE in a single-space save parses too
    const auto withTable = insertSection(one.save(), kTagEntities, kTagSpaces, spaceSection(1, {{"overworld", 0}}));
    ASSERT_TRUE(parseSave(withTable, p, error)) << error;
    ASSERT_TRUE(p.spaces.has_value());
    EXPECT_EQ(p.spaces->rows.size(), 1u);
    World load(baseSetup());
    EXPECT_TRUE(load.tryLoad(withTable).ok());
}

TEST(SaveSections, OrderErrors)
{
    World two(spacesSaveSetup(cellarSpaces()));
    const auto blob    = two.save();
    const auto content = spaceSection(1, {{"overworld", 0}, {"cellar", 1}});
    ParsedSave p;
    std::string error;
    // After ENTS
    const auto late = insertSection(removeSection(blob, kTagSpaces), kTagComponents, kTagSpaces, content);
    EXPECT_FALSE(parseSave(late, p, error));
    EXPECT_EQ(error, "unexpected section SPCE (expected COMP)");
    // Twice
    const auto twice = insertSection(blob, kTagEntities, kTagSpaces, content);
    EXPECT_FALSE(parseSave(twice, p, error));
    EXPECT_EQ(error, "unexpected section SPCE (expected GENR or TERR or CREC or ENTS)");
    // An unknown tag where an optional section may stand (S22's TERR is known)
    const auto unknown = insertSection(blob, kTagSpaces, saveTag("ZZZZ"), {1, 0, 0, 0});
    EXPECT_FALSE(parseSave(unknown, p, error));
    EXPECT_EQ(error, "unexpected section ZZZZ (expected SPCE or GENR or TERR or CREC or ENTS)");
    // A missing required section
    EXPECT_FALSE(parseSave(removeSection(blob, kTagEntities), p, error));
    EXPECT_EQ(error, "unexpected section COMP (expected GENR or TERR or CREC or ENTS)");
    // Every case leaves the world unchanged
    World w(spacesSaveSetup(cellarSpaces()));
    const auto before = w.save();
    for (const auto& b : {late, twice, unknown}) EXPECT_FALSE(w.tryLoad(b).committed);
    EXPECT_EQ(w.save(), before);
}

TEST(SaveSections, UnknownSectionVersion)
{
    World two(spacesSaveSetup(cellarSpaces()));
    const auto blob = replaceSection(two.save(), kTagSpaces, spaceSection(2, {{"overworld", 0}, {"cellar", 1}}));
    ParsedSave p;
    std::string error;
    EXPECT_FALSE(parseSave(blob, p, error));
    EXPECT_EQ(error, "SPCE: section format version 2 (this build reads 1)");
    // Malformed rows: an unknown kind, a long name, bytes left over
    EXPECT_FALSE(
        parseSave(replaceSection(two.save(), kTagSpaces, spaceSection(1, {{"overworld", 0}, {"x", 7}})), p, error));
    EXPECT_NE(error.find("unknown kind 7"), std::string::npos) << error;
    EXPECT_FALSE(parseSave(
        replaceSection(two.save(), kTagSpaces, spaceSection(1, {{"overworld", 0}, {std::string(257, 'a'), 1}})), p,
        error));
    EXPECT_NE(error.find("257 bytes"), std::string::npos) << error;
    auto extra = spaceSection(1, {{"overworld", 0}, {"cellar", 1}});
    extra.push_back(0);
    EXPECT_FALSE(parseSave(replaceSection(two.save(), kTagSpaces, extra), p, error));
    EXPECT_EQ(error, "SPCE: bytes left over");
}

TEST(SaveSections, DiffReportsSpaces)
{
    World a(spacesSaveSetup(cellarSpaces()));
    World b(spacesSaveSetup({SpaceTemplate(), SpaceTemplate::enclosed("cave")}));
    SaveDiff d = diffSaves(a.save(), b.save());
    ASSERT_TRUE(d.error.empty()) << d.error;
    ASSERT_EQ(d.differences.size(), 1u);
    EXPECT_EQ(d.differences[0].describe(), "space 1: 'cellar' (enclosed) vs 'cave' (enclosed)");
    World c(baseSetup());
    d = diffSaves(a.save(), c.save());
    ASSERT_TRUE(d.error.empty()) << d.error;
    bool section = false;
    for (const SaveDifference& x : d.differences) section = section || x.describe() == "section SPCE only in a";
    EXPECT_TRUE(section);
}

// Before S23a a setup cannot produce repeated names (unique templates, SpaceId = template index); the reader accepts
// them (several space instances of one template)
TEST(SaveSections, SpaceTableNamesMayRepeat)
{
    World two(spacesSaveSetup(cellarSpaces()));
    const auto blob = two.save();
    const auto rep  = replaceSection(blob, kTagSpaces, spaceSection(1, {{"overworld", 0}, {"overworld", 0}}));
    ParsedSave p;
    std::string error;
    ASSERT_TRUE(parseSave(rep, p, error)) << error;
    ASSERT_EQ(p.spaces->rows.size(), 2u);
    EXPECT_EQ(p.spaces->rows[0].name, "overworld");
    EXPECT_EQ(p.spaces->rows[1].name, "overworld");
    // diffSaves shows both rows (row 1 differs from the real save's)
    const SaveDiff d = diffSaves(blob, rep);
    ASSERT_TRUE(d.error.empty()) << d.error;
    ASSERT_EQ(d.differences.size(), 1u);
    EXPECT_EQ(d.differences[0].describe(), "space 1: 'cellar' (enclosed) vs 'overworld' (open world)");
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

// ---------------------------------------------------------------------------------------------------------------------
// Review regressions (S11 round 2)

// Nested structs of zero-byte fields multiplied the decoding work (10^depth field visits from a few KB)
TEST(SaveRobustness, NestedEmptyStructsAreBounded)
{
    World a(baseSetup());
    const auto blob = a.save();
    SchemaTable t   = readSchema(blob);
    const int depth = 12;
    for (int k = 0; k < depth; ++k)
    {
        SchemaType s{SchemaKind::Struct, "F" + std::to_string(k), 0, 1, {}};
        if (k + 1 < depth)
            for (int f = 0; f < 10; ++f)
                s.fields.push_back(FieldInfo{"f" + std::to_string(f), "F" + std::to_string(k + 1), {}});
        t.types.push_back(s);
    }
    for (SchemaType& s : t.types)
        if (s.name == "Weather") s.fields.push_back(FieldInfo{"x", "F0", {}});
    const auto bad   = withSchema(blob, t);
    const auto start = std::chrono::steady_clock::now();
    World w(baseSetup());
    const LoadResult r = w.tryLoad(bad);
    EXPECT_FALSE(r.committed);
    EXPECT_NE(r.error.find("without bytes"), std::string::npos) << r.error;
    EXPECT_FALSE(diffSaves(a.save(), bad).error.empty());
    EXPECT_LT(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), 5.0);
}

// A reused slot was one "generation" difference that an entity filter on the new handle could not find
TEST(SaveDiff, AReusedSlotIsTwoEntities)
{
    World a(baseSetup());
    const Entity old = a.create();
    a.add<Health>(old);
    const auto before = a.save();
    a.destroy(old);
    const Entity fresh = a.spawn<Sitter>(Vec3(4, 5, 6));
    ASSERT_EQ(fresh.index, old.index);
    const SaveDiff d = diffSaves(before, a.save());
    bool inA = false, inB = false;
    for (const SaveDifference& x : d.differences)
    {
        if (x.kind == "only_in_a" && x.entity == old) inA = true;
        if (x.kind == "only_in_b" && x.entity == fresh)
        {
            inB = true;
            EXPECT_EQ(x.className, "Sitter");
            EXPECT_EQ(x.position[0].number_value(), 4);
        }
    }
    EXPECT_TRUE(inA && inB);
    DiffFilter f;
    f.hasEntity        = true;
    f.entityIndex      = fresh.index;
    f.hasGeneration    = true;
    f.entityGeneration = fresh.generation;
    EXPECT_EQ(diffSaves(before, a.save(), f).total, 1u);
}

// A failed parent removed children whose hooks had run, unreported
TEST(Save, DescendantsOfAFailedObjectAreReportedAndNotHooked)
{
    World a(baseSetup());
    const Entity fragile = a.spawn<Fragile>(Vec3(0, 0, 0));
    const Entity child   = a.spawn<Walker>(Vec3(1, 0, 0));
    a.add<Parent>(child, Parent{fragile});
    g_throwOnLoad = true;
    g_onLoads     = 0;
    World b(baseSetup());
    const LoadResult r = b.tryLoad(a.save());
    g_throwOnLoad      = false;
    ASSERT_EQ(r.hookErrors.size(), 1u);
    ASSERT_EQ(r.removed.size(), 1u);
    EXPECT_EQ(r.removed[0], child);
    EXPECT_EQ(g_onLoads, 0);  // the child's onLoad did not run
}

// Non-finite header values, throwing migrations and queued actions of dropped players
TEST(SaveRobustness, HeaderMigrationsAndActions)
{
    World a(baseSetup());
    auto blob             = a.save();
    const size_t head     = sectionOffset(blob, kTagHeader);
    const double nan      = std::numeric_limits<double>::quiet_NaN();
    uint64_t versionBytes = 0;
    std::memcpy(&versionBytes, blob.data() + head, 8);  // the engine version string's length
    const size_t timeAt = head + 8 + size_t(versionBytes) + 8 + 8;
    auto bad            = blob;
    std::memcpy(bad.data() + timeAt, &nan, 8);
    World w(baseSetup());
    LoadResult r = w.tryLoad(bad);
    EXPECT_FALSE(r.committed);
    EXPECT_NE(r.error.find("finite"), std::string::npos) << r.error;

    // A migration that throws is a load error (the world unchanged)
    World v1w(setupV1());
    v1w.add<v1::Stats>(v1w.create());
    WorldSetup s = setupV2();
    s.migrations = [](MigrationList& m)
    {
        m.add<v2::Stats>(1, [](FieldMap&) { throw std::runtime_error("bad field"); });
        m.add<v2::Item>(1, [](FieldMap&) {});
        m.addRemoved("Old", RemovedPolicy::DropComponent);
    };
    World v2w(s);
    r = v2w.tryLoad(v1w.save());
    EXPECT_FALSE(r.committed);
    EXPECT_NE(r.error.find("threw: bad field"), std::string::npos) << r.error;

    // A queued action of a player dropped by the load stays (as in the live world: delivered with the stale handle)
    World p(setupV1());
    const Entity player = p.create();
    p.add<Player>(player);
    p.add<v1::Old>(player);
    p.queueAction(PlayerAction{player, 5, Vec3::Zero()});
    WorldSetup drop = baseSetup();
    drop.migrations = [](MigrationList& m) { m.addRemoved("Old", RemovedPolicy::DropEntity); };
    World q(drop);
    ASSERT_TRUE(q.tryLoad(p.save()).ok());
    q.step();
    ASSERT_EQ(q.events.read<ActionEvent>().size(), 1u);
    EXPECT_FALSE(q.alive(q.events.read<ActionEvent>()[0].player));
}

// ---------------------------------------------------------------------------------------------------------------------
// Review regressions (S11 round 3)

// The action size of the save reader was wrong (49 instead of 45 bytes): saves with 3 or more queued actions were
// rejected
TEST(Save, ManyQueuedActions)
{
    World a(baseSetup());
    std::vector<Entity> players;
    for (int i = 0; i < 5; ++i)
    {
        players.push_back(a.create());
        a.add<Player>(players.back());
        a.queueAction(PlayerAction{players.back(), uint32_t(10 + i), Vec3(i, 0, 0)});
    }
    World b(baseSetup());
    const LoadResult r = b.tryLoad(a.save());
    ASSERT_TRUE(r.ok()) << r.error;
    EXPECT_EQ(b.save(), a.save());
    b.step();
    ASSERT_EQ(b.events.read<ActionEvent>().size(), 5u);
    EXPECT_EQ(b.events.read<ActionEvent>()[4].action, 14u);
}

namespace
{
LoadResult g_innerLoad;
class LoadsInUpdate final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(kFirstGameClass);
    static constexpr const char* name = "LoadsInUpdate";
    void update(double) override { g_innerLoad = world().tryLoad(g_loaderBlob); }
};
}  // namespace

// A load inside a game hook (a "quick load" in update) destroyed the running object: refused in every build
TEST(Save, ALoadInsideAHookIsRefused)
{
    WorldSetup s;
    s.classes = [](ClassList& c) { c.add<LoadsInUpdate>(); };
    World a(s);
    g_loaderBlob                            = a.save();
    const Entity e                          = a.spawn<LoadsInUpdate>(Vec3(0, 0, 0));
    a.get<Behaviour>(e).alwaysTickEveryStep = true;
    a.step();
    EXPECT_FALSE(g_innerLoad.committed);
    EXPECT_NE(g_innerLoad.error.find("only between steps"), std::string::npos) << g_innerLoad.error;
    EXPECT_TRUE(a.alive(e));
}

// ---------------------------------------------------------------------------------------------------------------------
// CodeRabbit (PR #54)

namespace
{
enum class Light
{
    Off,
    On,
    Count
};
const char* const kLightNames[] = {"Off", "On"};
struct Lamp
{
    static constexpr const char* name = "Lamp";
    static constexpr ComponentId id   = ComponentId(kFirstGameComponent + 4);
    static constexpr uint32_t version = 1;
    bool lit                          = false;
    Light light                       = Light::Off;
    template <class V>
    void visit(V& v)
    {
        v("lit", lit);
        v("light", light, EnumNames{kLightNames});
    }
};
WorldSetup lampSetup()
{
    WorldSetup s = baseSetup();
    s.components = [](Registry& r)
    {
        baseComponents(r);
        r.registerComponent<Lamp>();
    };
    return s;
}
}  // namespace

// A new slot after a load into a world with more slots kept an old generation (the next handle differed)
TEST(Save, NewSlotsAfterALoadStartAtGenerationZero)
{
    World a(baseSetup());
    a.create();
    World b(baseSetup());
    std::vector<Entity> many;
    for (int i = 0; i < 5; ++i) many.push_back(b.create());
    for (Entity e : many) b.destroy(e);
    for (int i = 0; i < 5; ++i) b.create();
    ASSERT_TRUE(b.tryLoad(a.save()).ok());
    for (int i = 0; i < 6; ++i) EXPECT_EQ(b.create(), a.create()) << i;
}

// A dropped component's values were not checked: a malformed one let the load replace the world
TEST(SaveRobustness, DroppedComponentsAreChecked)
{
    World a(lampSetup());
    a.add<Lamp>(a.create());
    auto blob = a.save();
    ParsedSave p;
    std::string error;
    ASSERT_TRUE(parseSave(blob, p, error)) << error;
    size_t lampAt = 0;
    for (const auto& b : p.components)
        if (p.schema.types[b.schemaIndex].name == "Lamp") lampAt = b.begin + 4;  // after the entity index
    ASSERT_GT(lampAt, 0u);
    blob[lampAt] = 7;  // the bool
    WorldSetup s;
    s.components = baseComponents;
    s.classes    = baseClasses;
    s.migrations = [](MigrationList& m) { m.addRemoved("Lamp", RemovedPolicy::DropComponent); };
    World w(s);
    const auto before  = w.save();
    const LoadResult r = w.tryLoad(blob);
    EXPECT_FALSE(r.committed);
    EXPECT_EQ(w.save(), before);
}

// Regression (CodeRabbit, S11): a dropped component block could name the same entity twice (the staged path rejects
// that)
TEST(SaveRobustness, DroppedComponentsRejectDuplicateEntities)
{
    World a(lampSetup());
    a.add<Lamp>(a.create());
    a.add<Lamp>(a.create());
    auto blob = a.save();
    ParsedSave p;
    std::string error;
    ASSERT_TRUE(parseSave(blob, p, error)) << error;
    const ParsedSave::Block* lamps = nullptr;
    for (const auto& b : p.components)
        if (p.schema.types[b.schemaIndex].name == "Lamp") lamps = &b;
    ASSERT_NE(lamps, nullptr);
    ASSERT_EQ(lamps->count, 2u);
    // Both elements have the same size: the second index starts in the middle
    const size_t second = lamps->begin + (lamps->end - lamps->begin) / 2;
    std::copy(blob.begin() + std::ptrdiff_t(lamps->begin), blob.begin() + std::ptrdiff_t(lamps->begin) + 4,
              blob.begin() + std::ptrdiff_t(second));
    WorldSetup s;
    s.components = baseComponents;
    s.classes    = baseClasses;
    s.migrations = [](MigrationList& m) { m.addRemoved("Lamp", RemovedPolicy::DropComponent); };
    World w(s);
    w.create();
    const auto before  = w.save();
    const LoadResult r = w.tryLoad(blob);
    EXPECT_FALSE(r.committed);
    EXPECT_NE(r.error.find("appears twice"), std::string::npos) << r.error;
    EXPECT_EQ(w.save(), before);
}

// save() returned a block the loader rejects (an enum value without a name; more empty events than the reader allows)
TEST(Save, UnreadableSavesAreErrors)
{
    World a(lampSetup());
    const Entity e = a.create();
    a.add<Lamp>(e);
    a.get<Lamp>(e).light = Light::Count;
    EXPECT_THROW(a.save(), std::runtime_error);
}

namespace
{
struct Empty
{
    static constexpr const char* name = "Empty";
    static constexpr EventId id       = EventId(64);
    static constexpr uint32_t version = 1;
    template <class V>
    void visit(V&)
    {
    }
};
}  // namespace

TEST(Save, TooManyEmptyEventsAreAnError)
{
    WorldSetup s = baseSetup();
    s.events     = [](EventRegistry& r) { r.add<Empty>(); };
    World a(s);
    for (size_t i = 0; i < kMaxEmptyElements; ++i) a.events.emit(Empty{});
    World b(s);
    EXPECT_TRUE(b.tryLoad(a.save()).ok());  // exactly the limit
    a.events.emit(Empty{});
    EXPECT_THROW(a.save(), std::runtime_error);
}

// A load from a load listener (nested) reached the observers first; a listener removed by an earlier one was called
TEST(Save, NestedLoadsAndRemovedListeners)
{
    World a(baseSetup());
    const auto blob = a.save();
    World w(baseSetup());
    LoadResult inner;
    int second   = 0;
    int secondId = -1;
    w.addLoadListener(
        [&](const LoadInfo&)
        {
            inner = w.tryLoad(blob);
            w.removeLoadListener(secondId);
        });
    secondId = w.addLoadListener([&](const LoadInfo&) { second++; });
    ASSERT_TRUE(w.tryLoad(blob).ok());
    EXPECT_FALSE(inner.committed);
    EXPECT_NE(inner.error.find("another load"), std::string::npos) << inner.error;
    EXPECT_EQ(second, 0);
    // The guard is released: the next load works
    EXPECT_TRUE(w.tryLoad(blob).committed);
}

// The diff: free-list generations, classes by name, nested schemas before equal bytes
TEST(SaveDiff, GenerationsClassesAndSchemas)
{
    World a(baseSetup());
    const Entity e = a.create();
    a.destroy(e);
    const auto first = a.save();
    a.destroy(a.create());  // the same slot, one generation later
    const SaveDiff d = diffSaves(first, a.save());
    bool freeList    = false;
    for (const auto& x : d.differences) freeList = freeList || x.kind == "free_list";
    EXPECT_TRUE(freeList);

    // The same class under another ID: no difference
    World b(baseSetup());
    const Entity s1  = b.spawn<Sitter>(Vec3(0, 0, 0));
    WorldSetup other = baseSetup();
    other.classes    = [](ClassList& c)
    {
        c.add<SpawnsOnLoad>();  // ID 64
        c.add<Sitter>();        // ID 65 as before
        c.add<Fragile>();
    };
    World c(other);
    ASSERT_TRUE(c.tryLoad(b.save()).ok());
    EXPECT_EQ(diffSaves(b.save(), c.save()).total, 0u);
    (void)s1;
}

// Regression (CodeRabbit, S11): the schema reader limits array lengths; registered std::arrays (compile-time lengths,
// at most 2^31 - 1 by a static_assert of the field rules) round-trip around the old limit of 2^16
namespace
{
template <size_t N>
struct BigArray
{
    static constexpr const char* name = "BigArray";
    static constexpr ComponentId id   = ComponentId(kFirstGameComponent + 4);
    static constexpr uint32_t version = 1;
    std::array<uint8_t, N> bytes{};
    std::array<std::array<uint8_t, N>, 0> none{};  // no elements, the inner length is only in the schema
    template <class V>
    void visit(V& v)
    {
        v("bytes", bytes);
        v("none", none);
    }
};
template <size_t N>
void roundTripBigArray()
{
    WorldSetup s = baseSetup();
    s.components = [](Registry& r)
    {
        baseComponents(r);
        r.registerComponent<BigArray<N>>();
    };
    World a(s);
    const Entity e = a.create();
    a.add<BigArray<N>>(e);
    a.get<BigArray<N>>(e).bytes[N - 1] = 7;
    const auto blob                    = a.save();
    World b(s);
    const LoadResult r = b.tryLoad(blob);
    ASSERT_TRUE(r.committed) << r.error;
    EXPECT_EQ(b.read<BigArray<N>>(e).bytes[N - 1], 7);
    EXPECT_EQ(b.save(), blob);
}
}  // namespace

TEST(SaveRobustness, ArrayLengthsAroundTwoToTheSixteen)
{
    roundTripBigArray<65536>();
    roundTripBigArray<65537>();
}

// Regression (CodeRabbit, S11): the default of a state component missing from the save was constructed once as a probe
// before the commit and again after the world was cleared; a constructor that threw the second time left half a world.
// The value built before the commit is moved in now.
namespace
{
int flakyConstructionsLeft = 1000;
struct Flaky
{
    static constexpr const char* name = "Flaky";
    static constexpr ComponentId id   = ComponentId(kFirstGameComponent + 4);
    static constexpr uint32_t version = 1;
    int value                         = 3;
    Flaky()
    {
        if (flakyConstructionsLeft-- <= 0) throw std::runtime_error("no more Flaky");
    }
    template <class V>
    void visit(V& v)
    {
        v("value", value);
    }
};
}  // namespace

TEST(SaveRobustness, StateDefaultsAreBuiltBeforeTheCommit)
{
    World a(baseSetup());
    a.create();
    const auto blob = a.save();  // without Flaky
    WorldSetup s    = baseSetup();
    s.components    = [](Registry& r)
    {
        baseComponents(r);
        r.registerGlobal<Flaky>();
    };
    flakyConstructionsLeft = 1000;
    World b(s);
    // The schema of the world builds a throwaway instance of each type (before the commit, harmless)
    int left = flakyConstructionsLeft;
    (void)SchemaTable::fromWorld(b);
    const int schema = left - flakyConstructionsLeft;
    // The load builds the default once: the schema's instances and one staged default, none after the commit
    left         = flakyConstructionsLeft;
    LoadResult r = b.tryLoad(blob);
    ASSERT_TRUE(r.committed) << r.error;
    EXPECT_EQ(left - flakyConstructionsLeft, schema + 1);
    EXPECT_EQ(b.readGlobal<Flaky>().value, 3);
    // The staged default throws: the load is rejected before the world changes
    const auto before      = b.save();
    flakyConstructionsLeft = schema;
    r                      = b.tryLoad(blob);
    EXPECT_FALSE(r.committed);
    EXPECT_NE(r.error.find("no more Flaky"), std::string::npos) << r.error;
    flakyConstructionsLeft = 1000;
    EXPECT_EQ(b.save(), before);
    flakyConstructionsLeft = 1000;

    // S21: a space-state component with two spaces missing their entity: one staged default per space, the second
    // throws: rejected before the world changes
    WorldSetup t = baseSetup();
    t.components = [](Registry& r)
    {
        baseComponents(r);
        r.registerSpaceState<Flaky>();
    };
    t.config.spaceTemplates = {SpaceTemplate(), SpaceTemplate::enclosed("cellar"), SpaceTemplate::enclosed("cave")};
    flakyConstructionsLeft  = 1000;
    World c(t);
    left = flakyConstructionsLeft;
    r    = c.tryLoad(blob);
    ASSERT_TRUE(r.committed) << r.error;
    EXPECT_EQ(left - flakyConstructionsLeft, schema + 3);  // three spaces without their entity
    const auto beforeC     = c.save();
    flakyConstructionsLeft = schema + 1;
    r                      = c.tryLoad(blob);
    EXPECT_FALSE(r.committed);
    EXPECT_NE(r.error.find("no more Flaky"), std::string::npos) << r.error;
    flakyConstructionsLeft = 1000;
    EXPECT_EQ(c.save(), beforeC);
}
