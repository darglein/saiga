/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Data files (docs/engine_design.md §14, R-DATA-1; stage S11)

#include "saiga/engine/core/Log.h"
#include "saiga/engine/data/DataFile.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/gameplay/GameObject.h"
#include "saiga/engine/world/World.h"
#include "saiga/saiga_buildconfig.h"

#include "gtest/gtest.h"

#include <chrono>
#include <filesystem>
#include <fstream>

using namespace Saiga;
using namespace Saiga::Engine;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);

const std::string kDataDir = std::string(SAIGA_PROJECT_SOURCE_DIR) + "/data/tests/engine";

struct CreatureStats
{
    static constexpr const char* name = "CreatureStats";
    static constexpr uint32_t version = 1;
    double speed                      = 1;
    double health                     = 10;
    std::string sound                 = "none";
    template <class V>
    void visit(V& v)
    {
        v("speed", speed);
        v("health", health);
        v("sound", sound);
    }
};

// A component that refers to its data entry (systems read the tunables through it)
struct Creature
{
    static constexpr const char* name = "Creature";
    static constexpr ComponentId id   = ComponentId(kFirstGameComponent);
    static constexpr uint32_t version = 1;
    DataRef<CreatureStats> model;
    double hp = 0;
    template <class V>
    void visit(V& v)
    {
        v("model", model);
        v("hp", hp);
    }
};

// Reads its maximum health from the data in configure (constant configuration, R-STATE-1)
class Beast final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(kFirstGameClass);
    static constexpr const char* name = "Beast";
    void configure() override { maxHp = world().data(read<Creature>().model).health; }
    void onSpawn() override { write<Creature>().hp = maxHp; }
    double maxHp = 0;
};

WorldSetup dataSetup(const std::string& dir = kDataDir)
{
    WorldSetup s;
    s.components    = [](Registry& r) { r.registerComponent<Creature>(); };
    s.classes       = [](ClassList& c) { c.add<Beast>(); };
    s.dataTypes     = [](DataTypeList& d) { d.add<CreatureStats>(); };
    s.dataDirectory = dir;
    return s;
}

// A temporary data directory with one file
struct TempData
{
    std::filesystem::path dir;
    explicit TempData(const std::string& name)
    {
        // Unique per process: two test runs on one machine do not delete each other's files
        dir = std::filesystem::temp_directory_path() /
              ("saiga_engine_data_" + name + "_" +
               std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(dir);
    }
    ~TempData()
    {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
    void write(const std::string& content, const std::string& file = "creature_stats.json") const
    {
        std::ofstream f(dir / file, std::ios::binary);
        f << content;
    }
    std::string path() const { return dir.string(); }
};
}  // namespace

TEST(Data, EntriesLoadWithDefaults)
{
    World w(dataSetup());
    EXPECT_EQ(w.data<CreatureStats>("wolf").speed, 6.5);
    EXPECT_EQ(w.data<CreatureStats>("wolf").sound, "howl");
    // Missing keys keep the struct defaults
    EXPECT_EQ(w.data<CreatureStats>("bear").sound, "none");
    EXPECT_TRUE(w.hasData<CreatureStats>("bear"));
    EXPECT_FALSE(w.hasData<CreatureStats>("cat"));
    EXPECT_EQ(w.dataNames<CreatureStats>(), (std::vector<std::string>{"bear", "wolf"}));
    // An unknown entry is a DataError (fatal unless caught)
    EXPECT_THROW(w.data<CreatureStats>("cat"), DataError);
    EXPECT_EQ(snakeCase("CreatureStats"), "creature_stats");
}

TEST(Data, ConfigureReadsDataThroughADataRef)
{
    World w(dataSetup());
    Creature c;
    c.model.entry  = "bear";
    const Entity e = w.spawn<Beast>(Vec3(0, 0, 0), c);
    EXPECT_EQ(w.read<Creature>(e).hp, 120);
    // Resolved once: the pointer is cached
    EXPECT_EQ(&w.data(w.read<Creature>(e).model), &w.data<CreatureStats>("bear"));
    // Saved by name; JSON {"entry": name}
    EXPECT_EQ(toJson(w.read<Creature>(e))["model"]["entry"].string_value(), "bear");
    World b(dataSetup());
    ASSERT_TRUE(b.tryLoad(w.save()).ok());
    EXPECT_EQ(b.data(b.read<Creature>(e).model).health, 120);
    EXPECT_EQ(static_cast<const Beast*>(b.read<Behaviour>(e).object.get())->maxHp, 120);
}

TEST(Data, ReloadUpdatesEntriesInPlace)
{
    TempData t("reload");
    t.write(R"({"wolf": {"speed": 1}, "bear": {"speed": 2}})");
    World w(dataSetup(t.path()));
    const CreatureStats* wolf = &w.data<CreatureStats>("wolf");
    std::string error;
    // Changed values reach existing references; new entries are added
    t.write(R"({"wolf": {"speed": 5}, "bear": {"speed": 2}, "cat": {}})");
    ASSERT_TRUE(w.reloadData(error)) << error;
    EXPECT_EQ(&w.data<CreatureStats>("wolf"), wolf);
    EXPECT_EQ(wolf->speed, 5);
    EXPECT_TRUE(w.hasData<CreatureStats>("cat"));
    // An invalid file: nothing changes
    t.write(R"({"wolf": {"speed": "fast"}, "bear": {}, "cat": {}})");
    EXPECT_FALSE(w.reloadData(error));
    EXPECT_NE(error.find("wolf.speed"), std::string::npos) << error;
    EXPECT_EQ(wolf->speed, 5);
    // A removed entry: refused (components refer to entries)
    t.write(R"({"wolf": {}, "bear": {}})");
    EXPECT_FALSE(w.reloadData(error));
    EXPECT_NE(error.find("cat"), std::string::npos) << error;
    EXPECT_TRUE(w.hasData<CreatureStats>("cat"));
}

TEST(Data, MissingEntriesDuringALoadAreHookErrors)
{
    TempData t("missing");
    t.write(R"({"wolf": {}, "bear": {}})");
    World a(dataSetup(t.path()));
    Creature c;
    c.model.entry  = "wolf";
    const Entity e = a.spawn<Beast>(Vec3(0, 0, 0), c);
    // The other build's data no longer has the entry: the object fails to configure, the load reports it
    TempData u("missing_other");
    u.write(R"({"bear": {}})");
    World b(dataSetup(u.path()));
    const LoadResult r = b.tryLoad(a.save());
    EXPECT_TRUE(r.committed);
    ASSERT_EQ(r.hookErrors.size(), 1u);
    EXPECT_EQ(r.hookErrors[0].entity, e);
    EXPECT_NE(r.hookErrors[0].message.find("wolf"), std::string::npos) << r.hookErrors[0].message;
    EXPECT_FALSE(b.alive(e));
}

TEST(Data, ReadersReportFileAndKey)
{
    TempData t("readers");
    t.write(R"({"wolf": {"speedd": 1}})");
    std::map<std::string, CreatureStats> out;
    std::string error;
    EXPECT_FALSE(readDataFile((t.dir / "creature_stats.json").string(), out, error));
    EXPECT_NE(error.find("creature_stats.json: wolf.speedd: unknown field"), std::string::npos) << error;
    EXPECT_FALSE(readDataFile((t.dir / "missing.json").string(), out, error));
    EXPECT_NE(error.find("cannot be read"), std::string::npos) << error;
    // A shared store: worlds of one setup do not read the files again
    World a(dataSetup());
    WorldSetup s = dataSetup("/does/not/exist");
    s.sharedData = std::make_shared<DataStore>(DataTypeList(), "");
    {
        DataTypeList types;
        types.add<CreatureStats>();
        s.sharedData = std::make_shared<DataStore>(std::move(types), kDataDir);
        ASSERT_TRUE(s.sharedData->load(error)) << error;
    }
    World b(s);
    EXPECT_EQ(b.data<CreatureStats>("wolf").speed, 6.5);
}

namespace
{
// A data type with its own value check (validate)
struct BodyStats
{
    static constexpr const char* name = "BodyStats";
    static constexpr uint32_t version = 1;
    double radius                     = 0.5;
    template <class V>
    void visit(V& v)
    {
        v("radius", radius);
    }
    bool validate(std::string& error) const
    {
        if (radius > 0) return true;
        error = "radius must be positive";
        return false;
    }
};
}  // namespace

// Regression (CodeRabbit, S13b): reload_data accepted a collision radius of 0 (finite, so a valid double) and the
// platformer's controller divided by it; a type's validate() now rejects the entry, and a reload keeps the old values
TEST(Data, ValidateRejectsEntriesAndKeepsTheOldValues)
{
    TempData t("validate");
    t.write(R"({"default": {"radius": 0.3}})", "body_stats.json");
    WorldSetup s;
    s.dataTypes     = [](DataTypeList& d) { d.add<BodyStats>(); };
    s.dataDirectory = t.path();
    World w(s);
    EXPECT_EQ(w.data<BodyStats>("default").radius, 0.3);
    t.write(R"({"default": {"radius": 0}})", "body_stats.json");
    std::string error;
    EXPECT_FALSE(w.reloadData(error));
    EXPECT_NE(error.find("body_stats.json: default: radius must be positive"), std::string::npos) << error;
    EXPECT_EQ(w.data<BodyStats>("default").radius, 0.3);
}

// R-DATA-1: in every build (not only with SAIGA_ASSERTS) the World constructor stops on a bad data file, naming the
// file and the key
TEST(DataDeathTest, BadFilesStopTheWorldConstructor)
{
    TempData t("death");
    t.write(R"({"wolf": {"speedd": 1}})");
    EXPECT_DEATH(World w(dataSetup(t.path())), "creature_stats.json: wolf.speedd: unknown field \\(fields: speed");
    t.write(R"({"wolf": {"speed": "fast"}})");
    EXPECT_DEATH(World w(dataSetup(t.path())), "creature_stats.json: wolf.speed: expected a number");
    t.write(R"([1, 2])");
    EXPECT_DEATH(World w(dataSetup(t.path())), "expected an object of named entries");
    t.write(R"({"wolf": )");
    EXPECT_DEATH(World w(dataSetup(t.path())), "creature_stats.json");
    EXPECT_DEATH(World w(dataSetup(t.path() + "/nothing")), "creature_stats.json: cannot be read");
}

// Review regressions (S11 round 1)

// Changing a DataRef's entry returned the old entry's data (the cache ignored the name)
TEST(Data, ChangingADataRefResolvesAgain)
{
    World w(dataSetup());
    DataRef<CreatureStats> ref("wolf");
    EXPECT_EQ(w.data(ref).speed, 6.5);
    ref.entry = "bear";
    EXPECT_EQ(w.data(ref).speed, 4);
    // Saving or hashing a component does not drop the cache (visit no longer resets it)
    Creature c;
    c.model                     = ref;
    const CreatureStats* cached = &w.data(c.model);
    (void)toJson(c);
    EXPECT_EQ(&w.data(c.model), cached);
    // Another world's store resolves again
    World other(dataSetup());
    EXPECT_EQ(&other.data(c.model), &other.data<CreatureStats>("bear"));
}

// An entry that is not an object produced "wolf.expected an object"
TEST(Data, AnEntryThatIsNotAnObject)
{
    TempData t("not_object");
    t.write(R"({"wolf": 5})");
    std::map<std::string, CreatureStats> out;
    std::string error;
    EXPECT_FALSE(readDataFile((t.dir / "creature_stats.json").string(), out, error));
    EXPECT_NE(error.find("creature_stats.json: wolf: expected an object"), std::string::npos) << error;
}
