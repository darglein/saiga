/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The content generation (docs/engine_design.md §12.1, §15.3, R-CONTENT-1; S23a): the fixed-point toolkit and the
// integer lint (this unit is compiled with trapping UBSan on the Clang CI job, SAIGA_CONTENT_UBSAN).

#include "saiga/engine/content/ContentCache.h"
#include "saiga/engine/content/ContentConvert.h"
#include "saiga/engine/content/GenerationPipeline.h"
#include "saiga/engine/content/GeneratorList.h"
#include "saiga/engine/content/Prefab.h"
#include "saiga/engine/content/PrefabStore.h"
#include "saiga/engine/content/gen/AuthoredSpaceGenerator.h"
#include "saiga/engine/content/gen/FixedMath.h"
#include "saiga/engine/content/gen/FlatGenerator.h"
#include "saiga/engine/content/gen/IntNoise.h"
#include "saiga/engine/content/gen/TrigTable.h"
#include "saiga/engine/core/Hash.h"
#include "saiga/engine/core/JobPool.h"
#include "saiga/engine/core/StreamingPool.h"
#include "saiga/engine/terrain/HookGenerator.h"
#include "saiga/engine/terrain/TerrainGeneration.h"
#include "saiga/engine/world/SpaceTemplate.h"
#include "saiga/engine/world/World.h"

#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <clocale>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <set>
#include <thread>

#include "content_lint.h"
#include <condition_variable>

using namespace Saiga;
using namespace Saiga::Engine;

// ---- FixedMath -----------------------------------------------------------------------------------------------------

TEST(FixedMath, FloorDivModShift)
{
    for (int64_t a = -1000; a <= 1000; ++a)
        for (int64_t b = 1; b <= 1000; b += (b < 20 ? 1 : 37))
        {
            const int64_t q = int64_t(std::floor(double(a) / double(b)));
            ASSERT_EQ(floorDiv(a, b), q) << a << " / " << b;
            ASSERT_EQ(floorMod(a, b), a - q * b) << a << " % " << b;
            ASSERT_EQ(ceilDiv(a, b), int64_t(std::ceil(double(a) / double(b)))) << a << " / " << b;
        }
    for (int64_t a = -1000; a <= 1000; ++a)
        for (int s = 0; s < 12; ++s) ASSERT_EQ(floorShift(a, s), floorDiv(a, int64_t(1) << s)) << a << " >> " << s;
    const int64_t mn = std::numeric_limits<int64_t>::min(), mx = std::numeric_limits<int64_t>::max();
    EXPECT_EQ(floorDiv(mn, 1), mn);
    EXPECT_EQ(floorDiv(mx, 2), mx / 2);
    EXPECT_EQ(floorDiv(mn + 1, 2), mn / 2);
    EXPECT_EQ(floorDiv(-mx, mx), -1);
    EXPECT_EQ(floorMod(mn, 3), (mn % 3 + 3) % 3);
    EXPECT_EQ(floorShift(mn, 62), -2);
    EXPECT_EQ(floorShift(-1, 40), -1);
}

TEST(FixedMath, MulShiftAndLerp)
{
    EXPECT_EQ(mulShift(-3, 1, 1), -2);
    EXPECT_EQ(mulShift(3, 1, 1), 1);
    EXPECT_EQ(mulShift(-5, 3, 2), -4);  // -15 / 4 = -3.75 -> -4
    // The trig series' largest x^2 (about 2.65e9) by a Q30 coefficient fits
    EXPECT_EQ(mulShift(2650000000ll, int64_t(1) << 30, 30), 2650000000ll);
    EXPECT_TRUE(productFits(2650000000ll, int64_t(1) << 31));
    EXPECT_FALSE(productFits(int64_t(1) << 32, int64_t(1) << 31));
    EXPECT_FALSE(productFits(std::numeric_limits<int64_t>::min(), 1));
    EXPECT_EQ(lerpQ16(-100, 300, 0), -100);
    EXPECT_EQ(lerpQ16(-100, 300, 65536), 300);
    EXPECT_EQ(lerpQ16(0, 3, 32768), 1);
    EXPECT_EQ(lerpQ16(0, -3, 32768), -2);
}

#ifdef SAIGA_ASSERTS
TEST(FixedMathDeathTest, MulShiftOverflowStops)
{
    EXPECT_DEATH(mulShift(int64_t(1) << 32, int64_t(1) << 31, 0), "overflows");
}
#endif

TEST(FixedMath, SmoothQ16)
{
    EXPECT_EQ(smoothQ16(0), 0);
    EXPECT_EQ(smoothQ16(65536), 65536);
    // Monotone with Q32 intermediates (the S23a plan's Q16 Horner scheme had 1601 decreases, five of 2 LSB, and errors
    // up to 3.6 LSB; see its review log)
    int decreases   = 0;
    double maxError = 0;
    for (int64_t t = 0; t <= 65536; ++t)
    {
        const int64_t s = smoothQ16(t);
        ASSERT_GE(s, 0) << t;
        ASSERT_LE(s, 65536) << t;
        if (t > 0)
        {
            const int64_t p = smoothQ16(t - 1);
            ASSERT_GE(s, p - 1) << t;
            if (s < p) decreases++;
        }
        const double x     = double(t) / 65536.0;
        const double exact = (6 * x * x * x * x * x - 15 * x * x * x * x + 10 * x * x * x) * 65536.0;
        maxError           = std::max(maxError, std::abs(double(s) - exact));
    }
    EXPECT_LE(maxError, 2.0);
    EXPECT_EQ(decreases, 0);
    std::cout << "smoothQ16: " << decreases << " decreases, max error vs. the polynomial " << maxError << " LSB"
              << std::endl;
}

TEST(FixedMath, Isqrt64)
{
    auto check = [](uint64_t v)
    {
        const uint64_t r = isqrt64(v);
        ASSERT_LE(r, 0xFFFFFFFFull) << v;
        ASSERT_LE(r * r, v) << v;
        // (r + 1)^2 > v, also where (r + 1)^2 overflows (r = 2^32 - 1)
        if (r < 0xFFFFFFFFull)
        {
            ASSERT_GT((r + 1) * (r + 1), v) << v;
        }
    };
    for (uint64_t v = 0; v < (uint64_t(1) << 20); ++v) check(v);
    for (uint64_t k = 0; k < (uint64_t(1) << 16); ++k)
    {
        check(k * k);
        check(k * k + 1);
        if (k > 0) check(k * k - 1);
    }
    for (uint64_t i = 0; i < 100000; ++i)
    {
        const uint64_t k = contentHash64(7, i, 0, 0, 0) & 0xFFFFFFFFull;
        check(k * k);
        check(k * k + 1);
        if (k > 0) check(k * k - 1);
    }
    check(std::numeric_limits<uint64_t>::max());
    EXPECT_EQ(isqrt64(std::numeric_limits<uint64_t>::max()), 0xFFFFFFFFull);
}

// ---- TrigTable -----------------------------------------------------------------------------------------------------

TEST(TrigTable, WithinBoundOfLibm)
{
    int64_t maxError = 0;
    for (uint32_t a = 0; a < kTrigSteps; ++a)
    {
        const double angle = double(a) * 2.0 * 3.14159265358979323846 / double(kTrigSteps);
        const int64_t s    = sinQ30(a);
        const int64_t ref  = std::llround(std::sin(angle) * double(kTrigOne));
        maxError           = std::max(maxError, std::abs(s - ref));
    }
    EXPECT_LE(maxError, 8);
    std::cout << "sinQ30: max error vs. libm " << maxError << " LSB (Q30; derived bound 4)" << std::endl;
    EXPECT_EQ(sinQ30(0), 0);
    EXPECT_EQ(sinQ30(kTrigQuarter), sinQ30(kTrigQuarter));
    EXPECT_NEAR(double(sinQ30(kTrigQuarter)), double(kTrigOne), 8);
}

TEST(TrigTable, ExactSymmetries)
{
    for (uint32_t a = 0; a < kTrigSteps; ++a)
    {
        ASSERT_EQ(sinQ30(a + kTrigQuarter), cosQ30(a)) << a;
        ASSERT_EQ(sinQ30(kTrigSteps - a), -sinQ30(a)) << a;
        ASSERT_EQ(sinQ30(a + kTrigSteps), sinQ30(a)) << a;
    }
}

// ---- IntNoise ------------------------------------------------------------------------------------------------------

TEST(IntNoise, Ranges)
{
    int64_t lo = 0, hi = 0;
    // The centres of 2^16 cells (cell size 2^8): all four gradients within 22.5 degrees of the diagonal towards the
    // centre in about one cell of 850, so the extremes of both signs reach at least cos(22.5 deg) * 65536
    for (int64_t cz = -128; cz < 128; ++cz)
        for (int64_t cx = -128; cx < 128; ++cx)
        {
            const int64_t n = gradientNoise(99, cx * 256 + 128, cz * 256 + 128, 8);
            lo              = std::min(lo, n);
            hi              = std::max(hi, n);
        }
    std::cout << "gradientNoise at 2^16 cell centres: [" << lo << ", " << hi << "]" << std::endl;
    EXPECT_GE(hi, 60000);
    EXPECT_LE(hi, 65536);
    EXPECT_LE(lo, -60000);
    EXPECT_GE(lo, -65536);
    for (int64_t i = 0; i < 20000; ++i)
    {
        const int64_t x = int64_t(contentHash64(3, uint64_t(i), 0, 0, 0) % 2000000) - 1000000;
        const int64_t z = int64_t(contentHash64(3, uint64_t(i), 1, 0, 0) % 2000000) - 1000000;
        for (int s : {1, 4, 9, 16, 17, 30})
        {
            ASSERT_LE(std::abs(gradientNoise(5, x, z, s)), 65536);
            ASSERT_LE(std::abs(valueNoise(5, x, z, s)), 65536);
        }
        const int32_t f = fbm(5, x, z, 12, 6);
        ASSERT_LE(std::abs(f), 65536);
        const int32_t r = ridged(5, x, z, 12, 5, 2);
        ASSERT_GE(r, 0);
        ASSERT_LE(r, 65536);
    }
}

// Neighbouring samples differ by a bounded step (no jumps at cell borders), also across 0 for negative coordinates
// (floor, not truncation)
TEST(IntNoise, Continuity)
{
    for (int s : {4, 6, 8, 12})
    {
        const int64_t cell = int64_t(1) << s;
        // The slope bound of the blended gradients: about 2 * 1.41 * 65536 * 1.875 per cell
        const int64_t maxStep = 4 * 65536 * 2 / cell + 64;
        for (int64_t x = -3 * cell; x < 3 * cell; ++x)
            for (int64_t z : {int64_t(-5), int64_t(0), int64_t(7) * cell / 3})
            {
                ASSERT_LE(std::abs(gradientNoise(11, x + 1, z, s) - gradientNoise(11, x, z, s)), maxStep)
                    << "gradient, cell " << s << " x " << x;
                ASSERT_LE(std::abs(gradientNoise(11, z, x + 1, s) - gradientNoise(11, z, x, s)), maxStep)
                    << "gradient, cell " << s << " z " << x;
                ASSERT_LE(std::abs(valueNoise(11, x + 1, z, s) - valueNoise(11, x, z, s)), 4 * 131072 / cell + 64)
                    << "value, cell " << s << " x " << x;
            }
    }
    // At the lattice points gradient noise is 0 (every dot product is 0)
    for (int64_t x : {int64_t(-256), int64_t(0), int64_t(256), int64_t(-1024)})
        EXPECT_EQ(gradientNoise(1, x, -512, 8), 0) << x;
}

TEST(IntNoise, IndependentOctaveSeeds)
{
    // fbm's octaves use different seeds: an fbm of 2 octaves differs from the same noise added at two scales
    int differ = 0;
    for (int64_t i = 0; i < 1000; ++i)
    {
        const int64_t x = i * 37, z = i * 91 - 5000;
        const int64_t same =
            floorDiv(int64_t(gradientNoise(9, x, z, 10)) * 65536 + int64_t(gradientNoise(9, x, z, 9)) * 32768, 98304);
        if (fbm(9, x, z, 10, 2) != same) differ++;
    }
    EXPECT_GT(differ, 900);
    EXPECT_NE(contentHash64(1, 0, 0, 0, 0), contentHash64(1, 1, 0, 0, 0));
    EXPECT_NE(contentHash64(1, 0, 0, 0, 0), contentHash64(2, 0, 0, 0, 0));
    EXPECT_NE(contentHash64(1, 1, 2, 0, 0), contentHash64(1, 2, 1, 0, 0));
}

TEST(IntNoiseDeathTest, FbmOctavesBeyondCellShiftStop)
{
    EXPECT_DEATH(fbm(1, 0, 0, 3, 4), "octaves <= cellShift");
    EXPECT_DEATH(ridged(1, 0, 0, 20, 17), "octaves <= 16");
    EXPECT_DEATH(fbm(1, 0, 0, 20, 10, 2), "gainShift <= 16");
}

// ---- ContentLint ---------------------------------------------------------------------------------------------------

namespace
{
std::string describe(const std::vector<ContentLint::Finding>& f)
{
    std::string s;
    for (const auto& x : f) s += "  line " + std::to_string(x.line) + ": " + x.token + "\n";
    return s;
}
}  // namespace

// Every generator file (Scope 10 of the S23a plan): the files under engine/content/gen/, the interface headers,
// SpaceKind.h, the test generator and the games' content/ directories (each from the step that adds it)
TEST(ContentLint, Sources)
{
    namespace fs           = std::filesystem;
    const std::string root = SAIGA_PROJECT_SOURCE_DIR;
    ASSERT_FALSE(root.empty()) << "ContentLint needs SAIGA_WRITE_PROJECT_DIR_TO_HEADER=ON";
    const fs::path base(root);
    ASSERT_TRUE(fs::exists(base / "src/saiga/engine")) << "ContentLint: no source tree at " << root;

    struct Entry
    {
        fs::path path;
        ContentLint::IncludeRules rules;
    };
    std::vector<Entry> files;
    auto addDir = [&](const fs::path& dir, ContentLint::IncludeRules rules)
    {
        if (!fs::exists(dir)) return;
        for (const auto& e : fs::recursive_directory_iterator(dir))
        {
            if (!e.is_regular_file()) continue;
            const std::string ext = e.path().extension().string();
            if (ext == ".h" || ext == ".hpp" || ext == ".cpp" || ext == ".inl") files.push_back({e.path(), rules});
        }
    };
    auto addFile = [&](const fs::path& f, ContentLint::IncludeRules rules)
    {
        if (fs::exists(f)) files.push_back({f, rules});
    };
    addDir(base / "src/saiga/engine/content/gen", {});
    for (const char* h : {"Generator.h", "WorldPlan.h", "ChunkContent.h"})
        addFile(base / "src/saiga/engine/content" / h, {});
    addFile(base / "src/saiga/engine/world/SpaceKind.h", {});
    ContentLint::IncludeRules testRules;
    testRules.testGenerator = true;
    addFile(base / "tests/content_test_generator.h", testRules);
    ContentLint::IncludeRules gameRules;
    gameRules.gameContent = true;
    if (fs::exists(base / "samples/games"))
        for (const auto& g : fs::directory_iterator(base / "samples/games"))
            if (g.is_directory()) addDir(g.path() / "content", gameRules);
    std::sort(files.begin(), files.end(), [](const Entry& a, const Entry& b) { return a.path < b.path; });

    int findings = 0;
    for (const Entry& f : files)
    {
        const auto found = ContentLint::scan(ContentLint::readFile(f.path), f.rules);
        for (const auto& x : found)
            ADD_FAILURE() << fs::relative(f.path, base).generic_string() << ":" << x.line
                          << ": refused in content code: " << x.token;
        findings += int(found.size());
    }
    std::cout << "ContentLint: " << files.size() << " files scanned, " << findings << " findings" << std::endl;
    EXPECT_GE(files.size(), 5u);
}

// The scanner's self-test: tests/content_lint_fixture/bad_generator.h has every refused token once in code (reported),
// the same tokens in comments and literals and allowed look-alikes (not reported), CRLF line ends in part and a refused
// token split by a backslash-newline (reported at its first line)
TEST(ContentLint, Fixture)
{
    const std::string root = SAIGA_PROJECT_SOURCE_DIR;
    ASSERT_FALSE(root.empty()) << "ContentLint needs SAIGA_WRITE_PROJECT_DIR_TO_HEADER=ON";
    const std::string text = ContentLint::readFile(root + "/tests/content_lint_fixture/bad_generator.h");
    ASSERT_FALSE(text.empty());
    ASSERT_NE(text.find("\r\n"), std::string::npos) << "the fixture must keep its CRLF lines";
    const std::vector<ContentLint::Finding> expected = {
        {9, "cmath"},
        {10, "saiga/engine/world/SpaceTemplate.h"},
        {12, "float"},
        {13, "double"},
        {14, "long"},
        {15, "size_t"},
        {16, "__int128"},
        {17, "unordered_map"},
        {18, "uniform_int_distribution"},
        {19, "rand("},
        {20, "Rng"},
        {21, "sin("},
        {22, "floor("},
        {23, "std::hash"},
        {24, "kHeightUnit"},
        {25, "terrainSurfaceDouble"},
        {26, "chunkOrigin"},
        {26, "chunkSize"},
        {27, "Vec3"},
        {27, "quat"},
        {28, "SpaceTemplate"},
        {29, "0.7"},
        {30, "1e3"},
        {31, ".5f"},
        {32, "2."},
        {33, "0x1p3"},
        {40, "pow("},
        {41, "double"},
        {43, "cbrt("},
    };
    std::vector<ContentLint::Finding> sorted = expected;
    std::sort(sorted.begin(), sorted.end());
    const auto found = ContentLint::scan(text, {});
    EXPECT_EQ(found, sorted) << "found:\n" << describe(found) << "expected:\n" << describe(sorted);
}

// ---- Generator registration and the engine generators -------------------------------------------------------------

namespace
{
// A generator whose slot table and counts are set by the test (registration checks)
class SlotTestGenerator final : public Generator
{
   public:
    std::string generatorName = "slot_test";
    std::vector<GeneratorSlot> table;
    uint32_t perChunk = 16, keptAt0 = 16;
    bool increasingKept = false, legacy = false;
    const char* name() const override { return generatorName.c_str(); }
    uint16_t version() const override { return 1; }
    std::vector<GeneratorSlot> slots() const override { return table; }
    bool legacyTerrain() const override { return legacy; }
    void plan(uint16_t, WorldPlan&) const override {}
    int32_t height(uint16_t, const WorldPlan&, int64_t, int64_t) const override { return 0; }
    bool candidate(uint16_t, const WorldPlan&, ChunkId, uint32_t, uint32_t, Candidate&) const override { return false; }
    uint32_t candidatesPerChunk(uint32_t) const override { return perChunk; }
    uint32_t keptCount(uint32_t, int level) const override
    {
        if (level == 0) return keptAt0;
        return increasingKept ? perChunk + uint32_t(level) : perChunk >> level;
    }
    void grassMask(uint16_t, const WorldPlan& plan, ChunkId, GrassMask& out) const override
    {
        zeroGrassMask(plan, out);
    }
};

std::shared_ptr<SlotTestGenerator> slotGenerator(int slots)
{
    auto g = std::make_shared<SlotTestGenerator>();
    for (int i = 0; i < slots; ++i) g->table.push_back({"slot" + std::to_string(i), ReSnapOnEdit, {}});
    return g;
}

void registerOne(std::shared_ptr<const Generator> g)
{
    GeneratorList list;
    list.add(0, std::make_shared<FlatGenerator>());
    list.add(1, std::move(g));
    SpaceTemplate t;
    t.generator = 1;
    list.validate({t});
}
}  // namespace

TEST(Generator, RegistrationPasses)
{
    registerOne(slotGenerator(200));
    SpaceTemplate t;
    const GeneratorList list = makeGeneratorList(nullptr, {t});
    ASSERT_NE(list.find(0), nullptr);
    EXPECT_STREQ(list.find(0)->generator->name(), "flat");
    EXPECT_EQ(list.entries().size(), 1u);
}

TEST(GeneratorDeathTest, RegistrationChecks)
{
    EXPECT_DEATH(registerOne(slotGenerator(201)), "at most 200");
    {
        auto g           = slotGenerator(2);
        g->table[1].name = "slot0";
        EXPECT_DEATH(registerOne(g), "slot name is used twice");
    }
    {
        auto g            = slotGenerator(1);
        g->table[0].flags = ReSnapOnEdit | TombstoneOnEdit;
        EXPECT_DEATH(registerOne(g), "exactly one of ReSnapOnEdit and TombstoneOnEdit");
        g->table[0].flags = Cosmetic;
        EXPECT_DEATH(registerOne(g), "exactly one of ReSnapOnEdit and TombstoneOnEdit");
    }
    {
        auto g     = slotGenerator(1);
        g->keptAt0 = 15;
        EXPECT_DEATH(registerOne(g), "keptCount\\(slot, 0\\)");
    }
    {
        auto g            = slotGenerator(1);
        g->increasingKept = true;
        EXPECT_DEATH(registerOne(g), "keptCount increases");
    }
    {
        auto g      = slotGenerator(1);
        g->perChunk = g->keptAt0 = (uint32_t(1) << 24) + 1;
        EXPECT_DEATH(registerOne(g), "above 2\\^24");
    }
    {
        auto g    = slotGenerator(1);
        g->legacy = true;
        EXPECT_DEATH(registerOne(g), "legacyTerrain generator has no slots");
    }
    {
        // An unregistered template generator
        SpaceTemplate t;
        t.name      = "lost";
        t.generator = 7;
        EXPECT_DEATH(makeGeneratorList(nullptr, {t}), "'lost': the generator 7 is not registered");
    }
    {
        // A duplicate name and a duplicate ID (0 is FlatGenerator's)
        GeneratorList list;
        list.add(1, slotGenerator(0));
        EXPECT_DEATH(list.add(2, slotGenerator(0)), "registered twice");
        EXPECT_DEATH(list.add(1, std::make_shared<FlatGenerator>()), "is used by");
    }
}

TEST(Generator, FlatAndAuthored)
{
    WorldPlan plan;
    plan.space.sizeChunks  = 16;
    plan.space.baseSpacing = 2;
    plan.space.kind        = SpaceKind::Enclosed;
    FlatGenerator flat;
    EXPECT_FALSE(flat.hasTerrain());
    EXPECT_TRUE(flat.legacyTerrain());
    GrassMask mask;
    flat.grassMask(1, plan, ChunkId{}, mask);
    EXPECT_EQ(mask.size, 128);
    EXPECT_EQ(mask.density, std::vector<uint8_t>(128 * 128, 0));

    AuthoredSpaceGenerator dungeon("test_dungeon", "test_dungeon_rooms", 1);
    EXPECT_FALSE(dungeon.hasTerrain());
    EXPECT_FALSE(dungeon.legacyTerrain());
    dungeon.plan(1, plan);
    ASSERT_EQ(plan.sites.size(), 1u);
    const Site& s = plan.sites[0];
    // E4: the centre of the middle chunk, 2176 m for 16 chunks
    EXPECT_EQ(s.x, 2176);
    EXPECT_EQ(s.z, 2176);
    EXPECT_EQ(s.x, SpaceTemplate::enclosedOrigin(16));
    EXPECT_EQ(s.prefab, "test_dungeon_rooms");
    EXPECT_EQ(s.footprint[0], -2176);
    EXPECT_EQ(s.footprint[2], 4096 - 2176);
}

TEST(Generator, HookGeneratorClampsAndKeepsTheGrassRule)
{
    WorldPlan plan;
    plan.seed              = 5;
    plan.space.baseSpacing = 2;
    plan.space.sizeChunks  = 16;
    plan.space.seaLevel    = 0;
    HookGenerator big("big", [](uint64_t, int64_t, int64_t) { return std::numeric_limits<int32_t>::max(); });
    EXPECT_EQ(big.height(1, plan, 3, 4), kTerrainHeightLimit);
    EXPECT_TRUE(big.hasTerrain());
    EXPECT_TRUE(big.legacyTerrain());
    EXPECT_TRUE(big.integerOnly());

    auto hook = [](uint64_t seed, int64_t x, int64_t z) { return int32_t((x * 37 + z * 11 + int64_t(seed)) % 4000); };
    HookGenerator withRule("hook", hook, HookGrassRule{0.82, 0.4}, false);
    EXPECT_FALSE(withRule.integerOnly());
    EXPECT_EQ(withRule.height(1, plan, 100, 200), hook(5, 100, 200));
    ChunkId c;
    c.x = 3;
    c.z = 2;
    GrassMask mask;
    withRule.grassMask(1, plan, c, mask);
    // S20's rule on the hook's level-0 tile
    Vulkan::TerrainLayout layout;
    layout.baseSpacing = 2;
    layout.sizeChunks  = 16;
    RegionId r;
    r.x                             = 3;
    r.z                             = 2;
    const Vulkan::TerrainTileData t = generateTile(TerrainHeightSource::fromHook(hook, 5), layout, r);
    const Vulkan::GrassMaskData ref = generateGrassMask(t, layout, 0, 0.82, 0.4);
    EXPECT_EQ(mask.size, ref.size);
    EXPECT_EQ(mask.density, *ref.density);
    HookGenerator noRule("plain", hook);
    noRule.grassMask(1, plan, c, mask);
    EXPECT_EQ(mask.density, std::vector<uint8_t>(128 * 128, 0));
}

// ---- Prefabs -------------------------------------------------------------------------------------------------------

namespace
{
const std::string kTestData = std::string(SAIGA_PROJECT_SOURCE_DIR) + "/data/tests/engine";

// Parses a prefab from text; the error (empty on success)
std::string prefabError(const std::string& text, Prefab* out = nullptr)
{
    std::string error;
    const json11::Json j = json11::Json::parse(text, error);
    EXPECT_TRUE(error.empty()) << error;
    Prefab p;
    p.name = "t";
    if (parsePrefab(j, "t.json", p, error))
    {
        if (out) *out = p;
        return "";
    }
    return error;
}

const char* kMinimal = R"({"format": 1, "footprint": [-4, -4, 4, 4],
    "terrain_patch": {"origin": [-1, -1], "size": [2, 2], "heights": [1, 2, 3, 4], "weights": [256, 128, 0, 64]},
    "exclusion": [[-2, -2, 2, 2]],
    "entities": [{"class": "A", "position": [0.5, 0.25, -1.75], "yaw_deg": 90, "components": {"Health": {"hp": 3}}}],
    "static_geometry": [{"mesh": "m", "position": [1, 0, 1], "yaw_deg": -90}]})";

std::string replaced(std::string s, const std::string& from, const std::string& to)
{
    const size_t at = s.find(from);
    EXPECT_NE(at, std::string::npos) << from;
    if (at != std::string::npos) s.replace(at, from.size(), to);
    return s;
}
}  // namespace

TEST(Prefab, VersionsSelectFiles)
{
    std::string error;
    Prefab v1, v2;
    ASSERT_TRUE(readPrefab(kTestData, "test_ruin", 1, v1, error)) << error;
    ASSERT_TRUE(readPrefab(kTestData, "test_ruin", 2, v2, error)) << error;
    EXPECT_EQ(v1.file, kTestData + "/prefabs/test_ruin.json");
    EXPECT_EQ(v2.file, kTestData + "/prefabs/test_ruin.v2.json");
    EXPECT_EQ(v1.statics.size(), 2u);
    EXPECT_EQ(v2.statics.size(), 3u);
    EXPECT_NE(v1.contentHash, v2.contentHash);
    // The values, exact
    ASSERT_EQ(v1.entities.size(), 3u);
    EXPECT_EQ(v1.entities[0].className, "TestChest");
    EXPECT_EQ(v1.entities[0].x, 640);
    EXPECT_EQ(v1.entities[0].z, -320);
    EXPECT_EQ(v1.entities[0].yaw, 16384);
    EXPECT_EQ(v1.entities[1].y, 512);
    ASSERT_TRUE(v1.entities[2].spaceLink);
    EXPECT_FALSE(v1.entities[2].spaceLink->toParent);
    EXPECT_EQ(v1.entities[2].spaceLink->x, 2176 * 256 + 128);
    EXPECT_EQ(v1.spaces, std::vector<std::string>{"test_dungeon"});
    ASSERT_TRUE(v1.patch);
    EXPECT_EQ(v1.patch->sizeX, 25);

    // A missing version names the file
    EXPECT_FALSE(readPrefab(kTestData, "test_ruin", 3, v1, error));
    EXPECT_NE(error.find("test_ruin.v3.json"), std::string::npos) << error;
    PrefabStore store;
    EXPECT_EQ(store.get(kTestData, "test_shrine", 1, error), store.get(kTestData, "test_shrine", 1, error));
    EXPECT_EQ(store.size(), 1u);
    EXPECT_EQ(store.get(kTestData, "nope", 1, error), nullptr);
    EXPECT_NE(error.find("prefabs/nope.json"), std::string::npos) << error;
}

TEST(Prefab, ErrorsNameFileAndKey)
{
    Prefab p;
    ASSERT_EQ(prefabError(kMinimal, &p), "");
    EXPECT_EQ(p.entities[0].x, 128);
    EXPECT_EQ(p.entities[0].y, 256);
    EXPECT_EQ(p.entities[0].z, -448);
    EXPECT_EQ(p.statics[0].yaw, 65536 - 16384);
    EXPECT_EQ(prefabYaw(1), 182);     // floorDiv(65536 + 180, 360)
    EXPECT_EQ(prefabYaw(-1), 65354);  // floorDiv(-65536 + 180, 360) = -182, wrapped

    auto expectError = [](const std::string& text, const std::string& part)
    {
        const std::string e = prefabError(text);
        EXPECT_NE(e.find("t.json: "), std::string::npos) << e;
        EXPECT_NE(e.find(part), std::string::npos) << "expected '" << part << "' in: " << e;
    };
    expectError(replaced(kMinimal, "\"format\": 1,", "\"format\": 1, \"colour\": 3,"), "colour: unknown key");
    expectError(replaced(kMinimal, "\"class\": \"A\",", "\"class\": \"A\", \"scale\": 2,"),
                "entities[0].scale: unknown key");
    expectError(replaced(kMinimal, "[0.5, 0.25, -1.75]", "[0.1, 0, 0]"), "entities[0].position[0]: not an exact");
    expectError(replaced(kMinimal, "[0.5, 0.25, -1.75]", "[0, 0.0001, 0]"), "entities[0].position[1]: not an exact");
    expectError(replaced(kMinimal, "[256, 128, 0, 64]", "[256, 128, 0, 257]"), "terrain_patch.weights[3]");
    expectError(replaced(kMinimal, "[1, 2, 3, 4]", "[1, 2, 3]"), "terrain_patch.heights: expected 4 values");
    expectError(replaced(kMinimal, "[0.5, 0.25, -1.75]", "[4, 0, 0]"), "entities[0].position: outside the footprint");
    expectError(replaced(kMinimal, "[[-2, -2, 2, 2]]", "[[-2, -2, 5, 2]]"), "exclusion[0]: outside the footprint");
    expectError(replaced(kMinimal, "\"origin\": [-1, -1]", "\"origin\": [3, 3]"),
                "terrain_patch: the patch lies outside the footprint");
    expectError(replaced(kMinimal, "\"format\": 1,", "\"format\": 1, \"pieces\": [],"),
                "pieces: placed pieces in prefabs come with S30");
    expectError(replaced(kMinimal, "\"format\": 1,", "\"format\": 2,"), "format: out of range");
    expectError(replaced(kMinimal, "\"yaw_deg\": 90", "\"yaw_deg\": 22.5"), "entities[0].yaw_deg: expected an integer");
    expectError(
        R"({"format": 1, "footprint": [0, 0, 4, 4], "entities": [{"class": "A", "space_link": {"to": "link"}}]})",
        "entities[0].space_link.link: the prefab lists 0 spaces");
}

// The content hash covers the integer content, not whitespace or component values
TEST(Prefab, ContentHash)
{
    Prefab a, b, c, d;
    ASSERT_EQ(prefabError(kMinimal, &a), "");
    std::string spaced = kMinimal;
    for (size_t i = 0; i < spaced.size(); ++i)
        if (spaced[i] == ',') spaced.insert(++i, "\n   ");
    ASSERT_EQ(prefabError(spaced, &b), "");
    EXPECT_EQ(a.contentHash, b.contentHash);
    ASSERT_EQ(prefabError(replaced(kMinimal, "{\"hp\": 3}", "{\"hp\": 4}"), &c), "");
    EXPECT_EQ(a.contentHash, c.contentHash);
    ASSERT_EQ(prefabError(replaced(kMinimal, "[1, 2, 3, 4]", "[1, 2, 3, 5]"), &d), "");
    EXPECT_NE(a.contentHash, d.contentHash);
}

// Regression (S23a): a decimal-comma C locale must not change the parsed values (json11 used strtod)
TEST(Prefab, IndependentOfCLocale)
{
    std::string error;
    Prefab before, after;
    ASSERT_TRUE(readPrefab(kTestData, "test_ruin", 1, before, error)) << error;
    const char* locales[]      = {"de_DE.UTF-8", "de_DE.utf8", "de_DE", "German_Germany.1252"};
    const std::string previous = std::setlocale(LC_NUMERIC, nullptr);
    bool set                   = false;
    for (auto l : locales)
        if (!set && std::setlocale(LC_NUMERIC, l)) set = true;
    if (!set) GTEST_SKIP() << "No locale with a decimal comma installed";
    const bool ok = readPrefab(kTestData, "test_ruin", 1, after, error);
    std::setlocale(LC_NUMERIC, previous.c_str());
    ASSERT_TRUE(ok) << error;
    EXPECT_EQ(after.contentHash, before.contentHash);
    EXPECT_EQ(after.entities[0].x, before.entities[0].x);
    EXPECT_EQ(after.entities[0].z, -320);
}

// ---- The pipeline (S23a step 6): a site-test generator with configurable sites and temporary prefabs --------------

namespace
{
// Sites, listed instances and slots set by the test; natural height 2x - z (metres to height units)
class SiteTestGenerator final : public Generator
{
   public:
    std::vector<Site> sites;          // the overworld's (template 0)
    std::vector<Site> instanceSites;  // every other instance's
    std::vector<SpaceInstance> listed;
    bool changeSeed = false, terrain = true, baseFromHeight = true;
    std::atomic<int> planCalls{0};
    // A candidate of this chunk blocks until released (the held-job test)
    mutable std::atomic<bool> holdArmed{false}, holding{false}, release{false};
    ChunkId holdChunk;

    const char* name() const override { return "site_test"; }
    uint16_t version() const override { return 2; }
    std::vector<GeneratorSlot> slots() const override { return {{"rock", ReSnapOnEdit, {"m"}}}; }
    bool hasTerrain() const override { return terrain; }
    void plan(uint16_t version, WorldPlan& plan) const override
    {
        const_cast<SiteTestGenerator*>(this)->planCalls++;
        if (changeSeed) plan.seed++;
        plan.sites = plan.space.templateIndex == 0 ? sites : instanceSites;
        if (baseFromHeight)
            for (Site& s : plan.sites) s.baseHeight = height(version, plan, s.x, s.z);
        if (plan.space.templateIndex == 0) plan.spaces = listed;
    }
    int32_t height(uint16_t, const WorldPlan& plan, int64_t x, int64_t z) const override
    {
        if (!terrain || plan.space.kind == SpaceKind::Enclosed) return 0;
        return int32_t(2 * x - z);
    }
    bool candidate(uint16_t version, const WorldPlan&, ChunkId chunk, uint32_t, uint32_t counter,
                   Candidate& out) const override
    {
        if (holdArmed && chunk.x == holdChunk.x && chunk.z == holdChunk.z && chunk.space == holdChunk.space &&
            counter == 0)
        {
            holding = true;
            while (!release) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        out.x        = int32_t((counter % 8) * 8192 + 100 + version);
        out.z        = int32_t((counter / 8) * 8192 + 100);
        out.yaw      = uint16_t(counter * 1000);
        out.meshSlot = 0;
        return counter % 5 != 3;
    }
    uint32_t candidatesPerChunk(uint32_t) const override { return 64; }
    uint32_t keptCount(uint32_t, int level) const override { return 64u >> level; }
    void grassMask(uint16_t, const WorldPlan& plan, ChunkId, GrassMask& out) const override
    {
        out.size = 256 / plan.space.baseSpacing;
        out.density.assign(uint32_t(out.size * out.size), 200);
    }
};

std::shared_ptr<SiteTestGenerator> gSite;
void registerSiteTest(GeneratorList& l)
{
    l.add(1, gSite);
}

// Prefabs in a temporary data directory
struct TempPrefabs
{
    std::filesystem::path dir;
    TempPrefabs()
    {
        dir = std::filesystem::temp_directory_path() /
              ("saiga_content_test_" +
               std::to_string(uint64_t(std::chrono::steady_clock::now().time_since_epoch().count())));
        std::filesystem::create_directories(dir / "prefabs");
    }
    ~TempPrefabs()
    {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
    void write(const std::string& name, const std::string& json) const
    {
        std::ofstream(dir / "prefabs" / (name + ".json")) << json;
    }
};

// 9 x 9 patch at local [-4, 4]^2: height 100 relative, the marker 5000 at local (3, 1); weight 256, the first row
// (z = -4) 128; an exclusion [-2, 2)^2; an entity and a static part at the marker
std::string markerPrefab()
{
    std::string h, w;
    for (int z = -4; z <= 4; ++z)
        for (int x = -4; x <= 4; ++x)
        {
            h += (h.empty() ? "" : ", ") + std::string(x == 3 && z == 1 ? "5000" : "100");
            w += (w.empty() ? "" : ", ") + std::string(z == -4 ? "128" : "256");
        }
    return R"({"format": 1, "footprint": [-8, -8, 8, 8],
        "terrain_patch": {"origin": [-4, -4], "size": [9, 9], "heights": [)" +
           h + R"(], "weights": [)" + w + R"(]},
        "exclusion": [[-2, -2, 2, 2]],
        "entities": [{"class": "TestMarker", "position": [3, 0.5, 1], "yaw_deg": 0}],
        "static_geometry": [{"mesh": "m", "position": [3, 0, 1], "yaw_deg": 90}]})";
}

Site site(uint32_t key, const std::string& prefab, int64_t x, int64_t z, uint8_t q, std::array<int32_t, 4> f)
{
    Site s;
    s.key          = key;
    s.prefab       = prefab;
    s.x            = x;
    s.z            = z;
    s.quarterTurns = q;
    for (int i = 0; i < 4; ++i) s.footprint[i] = f[size_t(i)];
    return s;
}

SpaceTemplate openTemplate(const std::string& name, GeneratorId g, int32_t chunks = 16, int spacing = 2)
{
    SpaceTemplate t;
    t.name        = name;
    t.generator   = g;
    t.sizeChunks  = chunks;
    t.baseSpacing = spacing;
    t.seaLevel    = std::nullopt;
    return t;
}

WorldSetup siteSetup(const TempPrefabs& prefabs, std::vector<SpaceTemplate> templates, uint64_t seed = 77)
{
    WorldSetup s;
    s.seed                  = seed;
    s.generators            = registerSiteTest;
    s.dataDirectory         = prefabs.dir.string();
    s.config.spaceTemplates = std::move(templates);
    return s;
}
}  // namespace

// Site composition: the patch at full weight, the partial weight floor-rounded, natural outside; all four quarter
// turns (the marker at local (3, 1) lands at (1, -3), (-3, -1), (-1, 3)); the patch's first and last sample rows and
// columns are patched and the next sample outside is natural for every q
TEST(Pipeline, SiteCompositionAndQuarterTurns)
{
    TempPrefabs prefabs;
    prefabs.write("marker", markerPrefab());
    gSite = std::make_shared<SiteTestGenerator>();
    // Four sites far apart, one per quarter turn
    for (uint8_t q = 0; q < 4; ++q) gSite->sites.push_back(site(q, "marker", 500 + 600 * q, 700, q, {-8, -8, 8, 8}));
    ContentCache cache;
    const GenerationPlans plans =
        GenerationPipeline::build(siteSetup(prefabs, {openTemplate("overworld", 1)}), {}, cache);
    ASSERT_TRUE(plans.ok()) << plans.error;
    GenerationPipeline pipe(plans, std::make_shared<ContentCache>());
    const WorldPlan& plan      = *pipe.space(0).plan;
    auto natural               = [](int64_t x, int64_t z) { return int32_t(2 * x - z); };
    const int64_t marker[4][2] = {{3, 1}, {1, -3}, {-3, -1}, {-1, 3}};
    for (uint8_t q = 0; q < 4; ++q)
    {
        const Site& s = plan.sites[q];
        // baseHeight = the natural height at the site origin (plan() may call height())
        EXPECT_EQ(s.baseHeight, natural(s.x, s.z));
        const int64_t mx = s.x + marker[q][0], mz = s.z + marker[q][1];
        EXPECT_EQ(pipe.height(0, mx, mz), s.baseHeight + 5000) << "q " << int(q);
        // Every sample of the patch: patched (full or the first row's partial weight), the ring outside natural
        for (int lz = -5; lz <= 5; ++lz)
            for (int lx = -5; lx <= 5; ++lx)
            {
                int64_t rx = 0, rz = 0;
                rotateQuarter(lx, lz, q, rx, rz);
                const int64_t x = s.x + rx, z = s.z + rz;
                const int32_t n  = natural(x, z);
                int32_t expected = n;
                if (lx >= -4 && lx <= 4 && lz >= -4 && lz <= 4)
                {
                    const int64_t target = int64_t(s.baseHeight) + (lx == 3 && lz == 1 ? 5000 : 100);
                    const int64_t w      = lz == -4 ? 128 : 256;
                    expected             = int32_t(n + floorDiv((target - n) * w, 256));
                }
                ASSERT_EQ(pipe.height(0, x, z), expected) << "q " << int(q) << " local (" << lx << ", " << lz << ")";
            }
        // The stamped entity and the static part: rotated, the composed yaw
        ChunkId c;
        c.x                  = int32_t(floorDiv(mx, 256));
        c.z                  = int32_t(floorDiv(mz, 256));
        const auto& stamps   = pipe.stampList(c);
        const StampRecord* r = nullptr;
        for (const auto& x : stamps)
            if (x.siteKey == q) r = &x;
        ASSERT_NE(r, nullptr);
        EXPECT_EQ(r->x, mx * 256);
        EXPECT_EQ(r->z, mz * 256);
        EXPECT_EQ(r->y, s.baseHeight + 512);
        EXPECT_EQ(r->yaw, uint16_t(q * 16384));
        const auto& statics = pipe.staticList(c);
        ASSERT_FALSE(statics.empty());
        EXPECT_EQ(statics[0].yaw, uint16_t(16384 + q * 16384));
    }
    // Nothing was generated for the stamp lists
    EXPECT_EQ(pipe.cache().generations(), 0u);
}

// Candidates inside an exclusion are rejected and grass cells inside are 0; every candidate's y equals the surface rule
// recomputed from height(); instances sorted, x and z in range
TEST(Pipeline, ExclusionsAndSurface)
{
    TempPrefabs prefabs;
    prefabs.write("marker", markerPrefab());
    gSite = std::make_shared<SiteTestGenerator>();
    // A site whose exclusion [-2, 2)^2 contains the candidate at (32 m + 100 / 256 m, ...): the origin at (33, 33)
    gSite->sites.push_back(site(1, "marker", 33, 33, 1, {-8, -8, 8, 8}));
    ContentCache cache;
    // Pinned v1, so placement v1 and v2 are both valid
    GenerationPins pins;
    pins.pinned["site_test"] = 1;
    const GenerationPlans plans =
        GenerationPipeline::build(siteSetup(prefabs, {openTemplate("overworld", 1)}), pins, cache);
    ASSERT_TRUE(plans.ok()) << plans.error;
    GenerationPipeline pipe(plans, std::make_shared<ContentCache>());
    ChunkId c;
    auto p = pipe.chunkPlacement(c, 2);
    // Candidate 1 (x 8192 + 102, z 100) is at (32.4, 0.4): outside the exclusion; candidate 9 at (32.4, 32.4): inside
    bool has9 = false, has1 = false;
    InstanceKey last{0, 0};
    for (size_t i = 0; i < p->instances.size(); ++i)
    {
        const TierOneInstance& t = p->instances[i];
        if (i > 0)
        {
            EXPECT_LT(last, t.key);
        }
        last = t.key;
        has9 = has9 || t.key.counter == 9;
        has1 = has1 || t.key.counter == 1;
        EXPECT_NE(t.key.counter % 5, 3u);
        ASSERT_GE(t.x, 0);
        ASSERT_LT(t.x, 65536);
        // The surface rule from height() (2 m cells, fracBits 9)
        const int64_t i0 = t.x / 512, j0 = t.z / 512;
        auto h = [&](int64_t a, int64_t b) { return pipe.height(0, a * 2, b * 2); };
        EXPECT_EQ(t.y, terrainSurfaceUnits(h(i0, j0), h(i0 + 1, j0), h(i0, j0 + 1), h(i0 + 1, j0 + 1), t.x - i0 * 512,
                                           t.z - j0 * 512, 9));
    }
    EXPECT_TRUE(has1);
    EXPECT_FALSE(has9);
    // Grass: the exclusion [-2, 2)^2 rotated by one quarter turn covers x in [31, 35) and z in (31, 35] (the
    // point-in-local-rectangle rule); cell centres at 2 i + 1
    EXPECT_EQ(p->grass.density[size_t(16 * 128 + 16)], 0);    // centre (33, 33)
    EXPECT_EQ(p->grass.density[size_t(17 * 128 + 15)], 0);    // centre (31, 35)
    EXPECT_EQ(p->grass.density[size_t(15 * 128 + 15)], 200);  // centre (31, 31): z = 31 is outside
    EXPECT_EQ(p->grass.density[size_t(17 * 128 + 17)], 200);  // centre (35, 35): x = 35 is outside
    EXPECT_EQ(p->grass.density[size_t(14 * 128 + 14)], 200);  // centre (29, 29)
    // The cache: the same pointer for equal keys; the heights generated once for two placement versions
    EXPECT_EQ(pipe.chunkPlacement(c, 2), p);
    auto p1 = pipe.chunkPlacement(c, 1);
    EXPECT_NE(p1->hashes.tier1, p->hashes.tier1);
    EXPECT_EQ(p1->hashes.heights, p->hashes.heights);
    EXPECT_EQ(p1->hashes.stamps, p->hashes.stamps);
    EXPECT_EQ(pipe.cache().stats().baseGenerations, 1u);
    EXPECT_EQ(pipe.cache().stats().placementGenerations, 2u);
}

// Plan errors: a rotated footprint outside the space, overlapping and touching footprints; a site ending on a chunk
// border is listed in both chunks and the second chunk's stamp list is empty
TEST(Pipeline, FootprintRules)
{
    TempPrefabs prefabs;
    prefabs.write("marker", markerPrefab());
    prefabs.write("edge", R"({"format": 1, "footprint": [0, 0, 16, 16],
        "entities": [{"class": "TestMarker", "position": [1, 0, 1]}]})");
    auto planError = [&](std::vector<Site> sites)
    {
        gSite        = std::make_shared<SiteTestGenerator>();
        gSite->sites = std::move(sites);
        ContentCache cache;
        return GenerationPipeline::build(siteSetup(prefabs, {openTemplate("overworld", 1)}), {}, cache).error;
    };
    EXPECT_EQ(planError({site(1, "marker", 8, 8, 0, {-8, -8, 8, 8})}), "");
    EXPECT_NE(planError({site(1, "marker", 7, 8, 0, {-8, -8, 8, 8})}).find("leaves the space"), std::string::npos);
    // q = 1 maps x to -z: [-8, 8) in z becomes [-8, 8] in x around the origin: 4088 + 8 = 4096 is still inside
    EXPECT_EQ(planError({site(1, "marker", 4088, 100, 1, {-8, -8, 8, 8})}), "");
    EXPECT_NE(planError({site(1, "marker", 4089, 100, 1, {-8, -8, 8, 8})}).find("leaves the space"), std::string::npos);
    EXPECT_NE(
        planError({site(1, "marker", 100, 100, 0, {-8, -8, 8, 8}), site(2, "marker", 110, 100, 0, {-8, -8, 8, 8})})
            .find("overlaps"),
        std::string::npos);
    // Touching footprints overlap (closed bounds)
    EXPECT_NE(
        planError({site(1, "marker", 100, 100, 0, {-8, -8, 8, 8}), site(2, "marker", 116, 100, 0, {-8, -8, 8, 8})})
            .find("overlaps"),
        std::string::npos);
    EXPECT_EQ(
        planError({site(1, "marker", 100, 100, 0, {-8, -8, 8, 8}), site(2, "marker", 117, 100, 0, {-8, -8, 8, 8})}),
        "");
    EXPECT_NE(
        planError({site(1, "marker", 100, 100, 0, {-8, -8, 8, 8}), site(1, "marker", 300, 100, 0, {-8, -8, 8, 8})})
            .find("key is used twice"),
        std::string::npos);
    EXPECT_NE(planError({site(1, "marker", 100, 100, 0, {-4, -4, 4, 4})}).find("does not fit"), std::string::npos);
    EXPECT_NE(planError({site(1, "nope", 100, 100, 0, {-8, -8, 8, 8})}).find("prefabs/nope.json"), std::string::npos);

    // A site ending exactly on the chunk border x = 256: listed in chunks 0 and 1, the stamp in chunk 0 only
    gSite        = std::make_shared<SiteTestGenerator>();
    gSite->sites = {site(1, "edge", 240, 100, 0, {0, 0, 16, 16})};
    ContentCache cache;
    const GenerationPlans plans =
        GenerationPipeline::build(siteSetup(prefabs, {openTemplate("overworld", 1)}), {}, cache);
    ASSERT_TRUE(plans.ok()) << plans.error;
    GenerationPipeline pipe(plans, std::make_shared<ContentCache>());
    ChunkId c0, c1;
    c1.x = 1;
    EXPECT_EQ(pipe.space(0).plan->sitesByChunk.count(c0), 1u);
    EXPECT_EQ(pipe.space(0).plan->sitesByChunk.count(c1), 1u);
    EXPECT_EQ(pipe.stampList(c0).size(), 1u);
    EXPECT_TRUE(pipe.stampList(c1).empty());
}

// The space list: the overworld, listed instances, site instances (the "site" domain seed, link > 0), unreferenced
// templates in template order; FlatGenerator gives S21's list and seeds
TEST(Pipeline, SpaceList)
{
    TempPrefabs prefabs;
    prefabs.write("creator", R"({"format": 1, "footprint": [-4, -4, 4, 4],
        "spaces": [{"template": "cave"}, {"template": "cave"}]})");
    prefabs.write("nested", R"({"format": 1, "footprint": [-4, -4, 4, 4], "spaces": [{"template": "cave"}]})");
    gSite = std::make_shared<SiteTestGenerator>();
    gSite->sites.push_back(site(9, "creator", 100, 100, 0, {-4, -4, 4, 4}));
    SpaceInstance listed;
    listed.templateIndex = 1;
    listed.seed          = 4242;
    gSite->listed.push_back(listed);
    SpaceTemplate cave                         = SpaceTemplate::enclosed("cave", 4);
    cave.generator                             = 0;
    const std::vector<SpaceTemplate> templates = {openTemplate("overworld", 1), openTemplate("valley", 1), cave,
                                                  openTemplate("unused", 0)};
    ContentCache cache;
    const GenerationPlans plans = GenerationPipeline::build(siteSetup(prefabs, templates, 77), {}, cache);
    ASSERT_TRUE(plans.ok()) << plans.error;
    ASSERT_EQ(plans.spaces.size(), 5u);
    const uint64_t site0 = hashCombine(hashCombine(77, fnv1a64("site")), 9);
    EXPECT_EQ(plans.spaces[0]->instance.seed, 77u);
    EXPECT_EQ(plans.spaces[1]->instance.templateIndex, 1);
    EXPECT_EQ(plans.spaces[1]->instance.seed, 4242u);
    EXPECT_EQ(plans.spaces[1]->instance.origin, SpaceInstance::Listed);
    EXPECT_EQ(plans.spaces[2]->instance.templateIndex, 2);
    EXPECT_EQ(plans.spaces[2]->instance.seed, site0);
    EXPECT_EQ(plans.spaces[2]->instance.siteKey, 9u);
    EXPECT_EQ(plans.spaces[3]->instance.link, 1);
    EXPECT_EQ(plans.spaces[3]->instance.seed, hashCombine(site0, 1));
    EXPECT_EQ(plans.spaces[4]->instance.templateIndex, 3);
    EXPECT_EQ(plans.spaces[4]->instance.seed, hashCombine(77, 3));
    EXPECT_EQ(plans.spaces[4]->instance.origin, SpaceInstance::Unreferenced);
    EXPECT_FALSE(plans.spaces[2]->hasTerrain);
    EXPECT_TRUE(plans.spaces[0]->hasTerrain);
    EXPECT_FALSE(plans.spaces[4]->hasTerrain);  // FlatGenerator
    EXPECT_TRUE(plans.nonLegacy);
    EXPECT_TRUE(plans.anySites);

    // An instance-space site whose prefab creates spaces: a plan error (no nested instances)
    gSite->instanceSites.push_back(site(1, "nested", 100, 100, 0, {-4, -4, 4, 4}));
    ContentCache cache2;
    EXPECT_NE(
        GenerationPipeline::build(siteSetup(prefabs, templates, 77), {}, cache2).error.find("no nested instances"),
        std::string::npos);

    // FlatGenerator only: S21's list (one per template, SpaceId = template index, S21's seeds); no generation state
    WorldSetup flat;
    flat.seed                  = 5;
    flat.config.spaceTemplates = {openTemplate("a", 0), SpaceTemplate::enclosed("b", 4), openTemplate("c", 0)};
    ContentCache cache3;
    const GenerationPlans f = GenerationPipeline::build(flat, {}, cache3);
    ASSERT_TRUE(f.ok()) << f.error;
    ASSERT_EQ(f.spaces.size(), 3u);
    for (size_t i = 0; i < 3; ++i)
    {
        EXPECT_EQ(f.spaces[i]->instance.templateIndex, i);
        EXPECT_EQ(f.spaces[i]->instance.seed, i == 0 ? 5u : hashCombine(5, i));
    }
    EXPECT_FALSE(f.nonLegacy);
    EXPECT_FALSE(f.anySites);

    // A flat overworld with an AuthoredSpaceGenerator space writes generation state
    WorldSetup authored = flat;
    authored.generators = [](GeneratorList& l)
    { l.add(3, std::make_shared<AuthoredSpaceGenerator>("dungeon", "nope")); };
    authored.config.spaceTemplates[1].generator = 3;
    ContentCache cache4;
    const GenerationPlans a = GenerationPipeline::build(authored, {}, cache4);
    EXPECT_NE(a.error.find("prefabs/nope.json"), std::string::npos) << a.error;
}

TEST(PipelineDeathTest, GeneratorChangingTheSeedStops)
{
    TempPrefabs prefabs;
    gSite             = std::make_shared<SiteTestGenerator>();
    gSite->changeSeed = true;
    ContentCache cache;
    EXPECT_DEATH(GenerationPipeline::build(siteSetup(prefabs, {openTemplate("overworld", 1)}), {}, cache),
                 "changed plan.seed");
}

#ifdef SAIGA_ASSERTS
TEST(PipelineDeathTest, ChunkOutsideTheSpaceStops)
{
    TempPrefabs prefabs;
    gSite = std::make_shared<SiteTestGenerator>();
    ContentCache cache;
    const GenerationPlans plans =
        GenerationPipeline::build(siteSetup(prefabs, {openTemplate("overworld", 1)}), {}, cache);
    ASSERT_TRUE(plans.ok());
    GenerationPipeline pipe(plans, std::make_shared<ContentCache>());
    ChunkId c;
    c.x = 16;
    EXPECT_DEATH(pipe.chunkBase(c), "outside the space");
    c.x = -1;
    EXPECT_DEATH(pipe.chunkBase(c), "outside the space");
}
#endif

// height() from several threads equals the serial result; chunks generated serially and from a JobPool in reverse
// order give equal bytes and hashes
TEST(Pipeline, ThreadIndependence)
{
    TempPrefabs prefabs;
    prefabs.write("marker", markerPrefab());
    gSite = std::make_shared<SiteTestGenerator>();
    for (uint32_t k = 0; k < 6; ++k)
        gSite->sites.push_back(site(k, "marker", 100 + 300 * k, 120 + 200 * k, uint8_t(k % 4), {-8, -8, 8, 8}));
    ContentCache cache;
    const GenerationPlans plans =
        GenerationPipeline::build(siteSetup(prefabs, {openTemplate("overworld", 1)}), {}, cache);
    ASSERT_TRUE(plans.ok()) << plans.error;
    GenerationPipeline serial(plans, std::make_shared<ContentCache>());
    GenerationPipeline parallel(plans, std::make_shared<ContentCache>());
    std::vector<ChunkId> chunks;
    for (int z = 0; z < 4; ++z)
        for (int x = 0; x < 4; ++x)
        {
            ChunkId c;
            c.x = x;
            c.z = z;
            chunks.push_back(c);
        }
    std::vector<uint64_t> a(chunks.size()), b(chunks.size());
    for (size_t i = 0; i < chunks.size(); ++i) a[i] = serial.chunkPlacement(chunks[i], 2)->hashes.total;
    JobPool pool(3);
    pool.parallelFor(0, int64_t(chunks.size()), 1,
                     [&](int64_t lo, int64_t hi)
                     {
                         for (int64_t i = hi - 1; i >= lo; --i)
                         {
                             const size_t k = chunks.size() - 1 - size_t(i);
                             b[k]           = parallel.chunkPlacement(chunks[k], 2)->hashes.total;
                         }
                     });
    EXPECT_EQ(a, b);
    for (size_t i = 0; i < chunks.size(); ++i)
        EXPECT_EQ(*serial.chunkBase(chunks[i])->heights, *parallel.chunkBase(chunks[i])->heights);
    std::vector<int32_t> h1(4096), h2(4096);
    for (int64_t i = 0; i < 4096; ++i) h1[size_t(i)] = serial.height(0, i % 1024, i / 4);
    pool.parallelFor(0, 4096, 64,
                     [&](int64_t lo, int64_t hi)
                     {
                         for (int64_t i = lo; i < hi; ++i) h2[size_t(i)] = serial.height(0, i % 1024, i / 4);
                     });
    EXPECT_EQ(h1, h2);
}

// The exact conversions: a position 30 km from the origin round-trips to the integer units; a yaw quaternion has norm
// 1 within 1e-15
TEST(ContentConvert, Exact)
{
    const int64_t x = 30000 * 256 + 37, z = 29999 * 256 + 255;
    const Vec3 p = contentPosition(x, -12345, z);
    EXPECT_EQ(int64_t(p.x() * 256), x);
    EXPECT_EQ(int64_t(p.z() * 256), z);
    EXPECT_EQ(p.y() * 1024, -12345.0);
    ChunkId c;
    c.x = 117;
    c.z = 3;
    EXPECT_EQ(contentPosition(c, 100, 0, 65535), Vec3(117 * 256 + 100.0 / 256, 0, 3 * 256 + 65535.0 / 256));
    for (uint32_t yaw = 0; yaw < 65536; yaw += 7)
    {
        const Quat q = contentRotation(uint16_t(yaw));
        EXPECT_NEAR(q.norm(), 1.0, 1e-15);
    }
    const Quat q = contentRotation(16384);  // 90 degrees: local +x maps to -z
    const Vec3 v = q * Vec3(1, 0, 0);
    EXPECT_NEAR(v.x(), 0, 1e-8);
    EXPECT_NEAR(v.z(), -1, 1e-8);
    EXPECT_EQ(contentScale(1536), 1.5);
}

// The cache: LRU eviction by the byte budget; a pool job holding key K while the main thread requests K: no wait,
// inline generation, the first insert wins, equal hashes; a cancelled queued job leaves no in-flight entry; a second
// identity under one name stops
TEST(ContentCache, LruAndInFlight)
{
    TempPrefabs prefabs;
    gSite      = std::make_shared<SiteTestGenerator>();
    auto cache = std::make_shared<ContentCache>(size_t(200) << 10);  // 200 KB: about two chunks
    const GenerationPlans plans =
        GenerationPipeline::build(siteSetup(prefabs, {openTemplate("overworld", 1)}), {}, *cache);
    ASSERT_TRUE(plans.ok()) << plans.error;
    GenerationPipeline pipe(plans, cache);
    ChunkId c0, c1, c2;
    c1.x = 1;
    c2.x = 2;
    pipe.chunkPlacement(c0, 2);
    pipe.chunkPlacement(c1, 2);
    pipe.chunkPlacement(c2, 2);
    EXPECT_GT(cache->stats().evictions, 0u);
    EXPECT_LE(cache->stats().bytes, cache->byteBudget());

    // The held pool job: chunk (3, 0)'s first candidate blocks
    ChunkId held;
    held.x           = 3;
    gSite->holdChunk = held;
    gSite->holdArmed = true;
    StreamingPool pool(1);
    std::shared_ptr<const ChunkPlacement> fromJob;
    const ContentSource source = pipe.contentSource(0);
    source.requestPlacement(pool, held, 2, 0, [&](std::shared_ptr<const ChunkPlacement> p) { fromJob = p; });
    while (!gSite->holding) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const ContentCache::ChunkKey key{"site_test", 2, pipe.space(0).planHash, held};
    EXPECT_TRUE(cache->inFlight(key));
    // The main thread does not wait: inline generation (the hold applies only to the job's first call)
    gSite->holdArmed                              = false;
    std::shared_ptr<const ChunkPlacement> inlineP = pipe.chunkPlacement(held, 2);
    ASSERT_NE(inlineP, nullptr);
    gSite->release = true;
    pool.waitIdle();
    pool.drain();
    ASSERT_NE(fromJob, nullptr);
    EXPECT_EQ(fromJob->hashes.total, inlineP->hashes.total);
    // The first insert won: the job's result is the inline one
    EXPECT_EQ(fromJob, inlineP);
    EXPECT_FALSE(cache->inFlight(key));
    EXPECT_GE(cache->stats().droppedDuplicates, 1u);

    // A queued job cancelled before it starts leaves nothing behind
    StreamingPool idle(1);
    gSite->holding = false;
    gSite->release = false;
    ChunkId blocker;
    blocker.x        = 4;
    gSite->holdChunk = blocker;
    gSite->holdArmed = true;
    source.requestPlacement(idle, blocker, 2, 0, nullptr);
    while (!gSite->holding) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ChunkId queued;
    queued.x          = 5;
    const uint64_t id = source.requestPlacement(idle, queued, 2, 0, nullptr);
    EXPECT_TRUE(idle.cancel(id));
    const ContentCache::ChunkKey queuedKey{"site_test", 2, pipe.space(0).planHash, queued};
    EXPECT_FALSE(cache->inFlight(queuedKey));
    gSite->release = true;
    idle.waitIdle();
    idle.drain();
    EXPECT_EQ(cache->findPlacement(queuedKey), nullptr);
    EXPECT_NE(pipe.chunkPlacement(queued, 2), nullptr);
}

TEST(ContentCacheDeathTest, SecondIdentityStops)
{
    ContentCache cache;
    cache.registerIdentity({"g", 1, 7});
    cache.registerIdentity({"g", 1, 7});
    EXPECT_DEATH(cache.registerIdentity({"g", 2, 7}), "another slot table or version");
}
