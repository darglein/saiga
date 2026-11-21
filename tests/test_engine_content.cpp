/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The content generation (docs/engine_design.md §12.1, §15.3, R-CONTENT-1; S23a): the fixed-point toolkit and the
// integer lint (this unit is compiled with trapping UBSan on the Clang CI job, SAIGA_CONTENT_UBSAN).

#include "saiga/engine/api/ComponentJson.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/SpaceLink.h"
#include "saiga/engine/content/ChunkRecords.h"
#include "saiga/engine/content/ContentCache.h"
#include "saiga/engine/content/ContentConvert.h"
#include "saiga/engine/content/ContentHashFile.h"
#include "saiga/engine/content/ContentMessages.h"
#include "saiga/engine/content/GenerationPipeline.h"
#include "saiga/engine/content/GeneratorList.h"
#include "saiga/engine/content/Prefab.h"
#include "saiga/engine/content/PrefabStore.h"
#include "saiga/engine/content/Systems.h"
#include "saiga/engine/content/Tombstones.h"
#include "saiga/engine/content/gen/AuthoredSpaceGenerator.h"
#include "saiga/engine/content/gen/FixedMath.h"
#include "saiga/engine/content/gen/FlatGenerator.h"
#include "saiga/engine/content/gen/IntNoise.h"
#include "saiga/engine/content/gen/TrigTable.h"
#include "saiga/engine/core/Hash.h"
#include "saiga/engine/core/JobPool.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/core/LogRing.h"
#include "saiga/engine/core/StreamingPool.h"
#include "saiga/engine/net/LoopbackTransport.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/save/SaveDiff.h"
#include "saiga/engine/save/SaveReader.h"
#include "saiga/engine/sync/RecordingRenderSink.h"
#include "saiga/engine/terrain/HookGenerator.h"
#include "saiga/engine/terrain/TerrainBrush.h"
#include "saiga/engine/terrain/TerrainEditStore.h"
#include "saiga/engine/terrain/TerrainGeneration.h"
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/engine/world/SpaceTemplate.h"
#include "saiga/engine/world/World.h"

#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <thread>

#include "content_lint.h"
#include "content_test_setup.h"
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
            if (ext != ".h" && ext != ".hpp" && ext != ".cpp" && ext != ".inl") continue;
            ContentLint::IncludeRules r = rules;
            if (r.gameContent) r.gameDirs = {e.path().parent_path(), dir.parent_path()};
            files.push_back({e.path(), r});
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
        std::string text;
        if (!ContentLint::readFile(f.path, text))
        {
            ADD_FAILURE() << fs::relative(f.path, base).generic_string() << ": cannot be read";
            continue;
        }
        const auto found = ContentLint::scan(text, f.rules);
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
    std::string text;
    ASSERT_TRUE(ContentLint::readFile(root + "/tests/content_lint_fixture/bad_generator.h", text));
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

// Regressions of the scanner (PR review): a missing file read as empty and passed; "#include /* why */ <cmath>" and
// "std :: hash" were not seen; a libm call with a newline before "(" passed; a token after a multi-line comment or raw
// string was reported at the comment's first line; a game's quoted "cmath" passed as a game header
TEST(ContentLint, ScannerEdgeCases)
{
    std::string text;
    EXPECT_FALSE(ContentLint::readFile("/nonexistent/saiga_content_lint.h", text));
    using F = ContentLint::Finding;
    const std::string code =
        "#include /* why */ <cmath>\n"      // 1
        "int a = std :: hash<int>{}(1);\n"  // 2
        "int b = std::sin\n"                // 3
        "    (1);\n"                        // 4
        "/* a\n"                            // 5
        "*/ double c;\n"                    // 6
        "auto d = R\"x(\n"                  // 7
        ")x\" double e;\n";                 // 8
    std::vector<F> expected = {{1, "cmath"}, {2, "std::hash"}, {3, "sin("}, {6, "double"}, {8, "double"}};
    EXPECT_EQ(ContentLint::scan(code, {}), expected) << describe(ContentLint::scan(code, {}));

    const std::string root = SAIGA_PROJECT_SOURCE_DIR;
    ContentLint::IncludeRules game;
    game.gameContent           = true;
    game.gameDirs              = {std::filesystem::path(root) / "tests"};
    expected                   = {{2, "cmath"}};
    const std::string includes = "#include \"content_lint.h\"\n#include \"cmath\"\n";
    EXPECT_EQ(ContentLint::scan(includes, game), expected) << describe(ContentLint::scan(includes, game));
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
    // Regression: "test_ruin.v2" at v1 named the file of test_ruin at v2 (one file for two pins)
    EXPECT_FALSE(readPrefab(kTestData, "test_ruin.v2", 1, v2, error));
    EXPECT_NE(error.find("not a valid prefab name"), std::string::npos) << error;
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
    // Regression: degrees * 65536 overflowed for large angles; a full turn is 65536 steps (wrapped away)
    EXPECT_EQ(prefabYaw(361), prefabYaw(1));
    EXPECT_EQ(prefabYaw(-359), prefabYaw(1));
    EXPECT_EQ(prefabYaw(INT64_MAX), prefabYaw(INT64_MAX % 360));
    EXPECT_EQ(prefabYaw(INT64_MIN), prefabYaw(INT64_MIN % 360 + 360));

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
        // Deterministic per test, run directory and construction: a death test's child (which re-runs the test up to
        // the death statement and never runs destructors) writes the same directories, which the parent removes
        static std::map<std::string, int> counters;
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        const std::string test =
            info ? std::string(info->test_suite_name()) + "_" + info->name() : std::string("no_test");
        const uint64_t run = fnv1a64(std::filesystem::current_path().string());
        dir = std::filesystem::temp_directory_path() / ("saiga_content_test_" + std::to_string(run % 1000000) + "_" +
                                                        test + "_" + std::to_string(counters[test]++));
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);  // a stale one of an aborted run
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

// Regression: a failed parse left the requested name and version in a previously read prefab (with its old payload)
TEST(Prefab, FailedReadLeavesTheOutputUnchanged)
{
    TempPrefabs prefabs;
    prefabs.write("broken", R"({"format": 1, "footprint": [4, 4, 0, 0]})");
    std::string error;
    Prefab p;
    ASSERT_TRUE(readPrefab(kTestData, "test_ruin", 1, p, error)) << error;
    const uint64_t hash = p.contentHash;
    EXPECT_FALSE(readPrefab(prefabs.dir.string(), "broken", 1, p, error));
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(p.name, "test_ruin");
    EXPECT_EQ(p.version, 1);
    EXPECT_EQ(p.contentHash, hash);
}

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
    // Regression: an origin near INT64_MAX overflowed the translated bounds (the wrapped right edge passed the check)
    {
        gSite                 = std::make_shared<SiteTestGenerator>();
        gSite->baseFromHeight = false;
        gSite->sites          = {site(1, "edge", INT64_MAX, 0, 0, {0, 0, 1, 1})};
        ContentCache cache;
        const std::string e =
            GenerationPipeline::build(siteSetup(prefabs, {openTemplate("overworld", 1)}), {}, cache).error;
        EXPECT_NE(e.find("the origin (9223372036854775807, 0) leaves the space"), std::string::npos) << e;
    }

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

// The exact conversions: a position 30 km from the origin round-trips to the integer units; a yaw quaternion (the Q30
// table, not normalised) has norm 1 within 2^-28
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
        EXPECT_NEAR(q.norm(), 1.0, std::ldexp(1.0, -28));
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

// ---- The committed content hashes and the content test setup (S23a step 8) ----------------------------------------

namespace
{
std::string contentHashFile()
{
    return std::string(SAIGA_PROJECT_SOURCE_DIR) + "/data/tests/engine/content_hashes.json";
}

// A copy of the test prefabs in a temporary directory (a test overwrites one of them)
struct PrefabCopy : TempPrefabs
{
    PrefabCopy()
    {
        for (const auto& e : std::filesystem::directory_iterator(kTestData + "/prefabs"))
            std::filesystem::copy_file(e.path(), dir / "prefabs" / e.path().filename(),
                                       std::filesystem::copy_options::overwrite_existing);
    }
};
}  // namespace

// The S23a acceptance test: the committed hashes of the engine test generator reproduce (GCC, Clang with UBSan, MSVC
// in CI). SAIGA_UPDATE_CONTENT_HASHES=1 rewrites the file in the source tree and fails the run
TEST(ContentHash, CommittedHashesReproduce)
{
    const std::string root = SAIGA_PROJECT_SOURCE_DIR;
    ASSERT_FALSE(root.empty()) << "the committed hashes need SAIGA_WRITE_PROJECT_DIR_TO_HEADER=ON";
    // Only the value 1 (PR review: an exported SAIGA_UPDATE_CONTENT_HASHES=0 rewrote the committed file)
    const char* env                = std::getenv("SAIGA_UPDATE_CONTENT_HASHES");
    const bool update              = env && std::string(env) == "1";
    const ContentHashReport report = checkContentHashes(ContentTest::contentTestSetup(2), contentHashFile(), update);
    std::cout << report.summary() << std::endl;
    EXPECT_TRUE(report.ok) << report.summary();
    EXPECT_EQ(report.entries, 2);
    // The toolkit section is optional for checkContentHashes, the engine's file has it (PR review)
    json11::Json file;
    std::string error;
    ASSERT_TRUE(readJsonFile(contentHashFile(), file, error)) << error;
    EXPECT_TRUE(file["toolkit"].is_object());
}

// Plan v1 with placement v1 and v2: heights, stamps and statics equal; tier1 and grass differ; both entries are
// committed
TEST(ContentHash, PlacementVersionsCoexist)
{
    GenerationPins pins;
    pins.pinned["test_terrain"] = 1;
    ContentCache cache;
    const GenerationPlans plans = GenerationPipeline::build(ContentTest::contentTestSetup(2), pins, cache);
    ASSERT_TRUE(plans.ok()) << plans.error;
    GenerationPipeline pipe(plans, std::make_shared<ContentCache>());
    int grassDiffers = 0;
    for (const auto& [x, z] : {std::pair<int, int>{4, 5}, {0, 0}, {6, 6}})
    {
        ChunkId c;
        c.x           = x;
        c.z           = z;
        const auto p1 = pipe.chunkPlacement(c, 1), p2 = pipe.chunkPlacement(c, 2);
        EXPECT_EQ(p1->hashes.heights, p2->hashes.heights);
        EXPECT_EQ(p1->hashes.stamps, p2->hashes.stamps);
        EXPECT_EQ(p1->hashes.statics, p2->hashes.statics);
        EXPECT_NE(p1->hashes.tier1, p2->hashes.tier1);
        EXPECT_NE(p1->hashes.total, p2->hashes.total);
        // The grass threshold differs (chunks without steep cells may have equal masks)
        grassDiffers += p1->hashes.grass != p2->hashes.grass ? 1 : 0;
    }
    EXPECT_GE(grassDiffers, 1);
    json11::Json file;
    std::string error;
    ASSERT_TRUE(readJsonFile(contentHashFile(), file, error)) << error;
    std::set<double> placements;
    for (const auto& e : file["setups"].array_items())
        placements.insert(e["generators"]["test_terrain"]["placement_version"].number_value());
    EXPECT_EQ(placements, (std::set<double>{1, 2}));
}

TEST(ContentHashFile, ParseU64)
{
    uint64_t v = 0;
    EXPECT_TRUE(parseU64("12345", false, v));
    EXPECT_EQ(v, 12345u);
    EXPECT_TRUE(parseU64("18446744073709551615", false, v));  // above 2^63
    EXPECT_EQ(v, std::numeric_limits<uint64_t>::max());
    EXPECT_TRUE(parseU64("0xFFfe0000000000a1", true, v));
    EXPECT_EQ(v, 0xFFFE0000000000A1ull);
    EXPECT_FALSE(parseU64("12345x", false, v));
    EXPECT_FALSE(parseU64("0x12g", true, v));
    EXPECT_FALSE(parseU64("123", true, v));  // no 0x prefix
    EXPECT_FALSE(parseU64("0x", true, v));
    EXPECT_FALSE(parseU64("", false, v));
    EXPECT_FALSE(parseU64("-1", false, v));
    EXPECT_FALSE(parseU64("18446744073709551616", false, v));
    EXPECT_EQ(hexU64(0xA1), "0x00000000000000a1");
}

// The check without a World (phase A only): a missing file is an error naming the path (also with update); a skeleton
// copy is filled by an update run (which fails) and then passes; a changed value is reported with entry, chunk and
// section
TEST(ContentHashFile, UpdateAndMismatch)
{
    const WorldSetup setup = ContentTest::contentTestSetup(2);
    TempPrefabs tmp;
    const std::string missing = (tmp.dir / "nope.json").string();
    for (bool update : {false, true})
    {
        const ContentHashReport r = checkContentHashes(setup, missing, update);
        EXPECT_FALSE(r.ok);
        ASSERT_FALSE(r.problems.empty());
        EXPECT_NE(r.problems[0].find(missing), std::string::npos);
        EXPECT_FALSE(std::filesystem::exists(missing));
    }
    // A skeleton of one entry with two chunks and a small digest block
    const std::string file = (tmp.dir / "hashes.json").string();
    std::ofstream(file) << R"({"format": 1, "setups": [{"name": "small", "seed": "12345",
        "generators": {"test_terrain": {"plan_version": 1, "placement_version": 2, "slots": ["test_tree", "test_rock"]},
                       "test_dungeon": {"plan_version": 1, "placement_version": 1, "slots": []}},
        "plans": {"0": {"generator": "test_terrain", "hash": "0x0"}, "1": {"generator": "test_terrain", "hash": "0x0"},
                  "2": {"generator": "test_dungeon", "hash": "0x0"}},
        "combined_plan_hash": "0x0", "digest_block": [0, 0, 1, 1],
        "chunks": {"0:4:5": {"total": "0x0", "heights": "0x0", "tier1": "0x0", "grass": "0x0", "stamps": "0x0",
                             "statics": "0x0"},
                   "2:8:8": {"total": "0x0", "heights": "0x0", "tier1": "0x0", "grass": "0x0", "stamps": "0x0",
                             "statics": "0x0"}},
        "digest": "0x0"}]})";
    const ContentHashReport first = checkContentHashes(setup, file, false);
    EXPECT_FALSE(first.ok);
    const ContentHashReport updated = checkContentHashes(setup, file, true);
    EXPECT_FALSE(updated.ok);
    EXPECT_TRUE(updated.updated);
    const ContentHashReport second = checkContentHashes(setup, file, false);
    EXPECT_TRUE(second.ok) << second.summary();
    std::cout << "small hash file: " << second.chunks << " chunks in " << second.seconds << " s" << std::endl;

    // A changed tier1 value of chunk 0:4:5
    json11::Json j;
    std::string error;
    ASSERT_TRUE(readJsonFile(file, j, error)) << error;
    std::string text        = j.dump();
    const std::string tier1 = j["setups"][0]["chunks"]["0:4:5"]["tier1"].string_value();
    text.replace(text.find(tier1), tier1.size(), "0x0000000000000001");
    std::ofstream(file, std::ios::trunc) << text;
    const ContentHashReport changed = checkContentHashes(setup, file, false);
    EXPECT_FALSE(changed.ok);
    ASSERT_EQ(changed.problems.size(), 1u) << changed.summary();
    EXPECT_NE(
        changed.problems[0].find("setups[0] (small).chunks.0:4:5 tier1: expected 0x0000000000000001, actual " + tier1),
        std::string::npos)
        << changed.problems[0];

    // Problems of the structure are reported per field, and an update run leaves the file unchanged (regressions: a
    // chunk outside its space was dropped from the rewritten file; plan_version 1.5 was read as 1 and a digest_block
    // value of 1e300 converted to int64_t)
    auto structural = [&](const std::string& from, const std::string& to, const std::string& expected)
    {
        std::string bad = j.dump();
        ASSERT_NE(bad.find(from), std::string::npos) << from;
        bad.replace(bad.find(from), from.size(), to);
        std::ofstream(file, std::ios::trunc) << bad;
        const ContentHashReport r = checkContentHashes(setup, file, true);
        EXPECT_FALSE(r.ok);
        EXPECT_FALSE(r.updated);
        EXPECT_NE(r.summary().find(expected), std::string::npos) << r.summary();
        std::ifstream in(file, std::ios::binary);
        EXPECT_EQ(std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()), bad);
    };
    structural("\"0:4:5\"", "\"0:128:5\"", "setups[0] (small).chunks.0:128:5: not a chunk of the setup");
    structural("\"plan_version\": 1,", "\"plan_version\": 1.5,",
               "setups[0] (small).generators.test_dungeon.plan_version: expected an integer in [1, 65535], got 1.5");
    structural("\"digest_block\": [0, 0, 1, 1]", "\"digest_block\": [0, 0, 1e300, 1]",
               "setups[0] (small).digest_block[2]: expected an integer");
    // Regressions (PR review): a block past the space was clipped silently; a non-canonical chunk key named the same
    // chunk as the canonical one; slots, chunks and toolkit values of another type read as empty
    structural("\"digest_block\": [0, 0, 1, 1]", "\"digest_block\": [-1, 0, 1, 1]",
               "setups[0] (small).digest_block[0]: expected an integer in [0, 128]");
    structural("\"0:4:5\"", "\"0:04:5\"", "setups[0] (small).chunks.0:04:5: not a chunk of the setup");
    structural("\"slots\": []", "\"slots\": 7", "setups[0] (small).generators.test_dungeon.slots: expected an array");
    structural("\"chunks\": {", "\"chunks\": [], \"unused\": {", "setups[0] (small).chunks: expected an object");
    structural("\"format\": 1,", "\"format\": 1, \"toolkit\": [],", "toolkit: expected an object");
}

// The test setup in a World: the space list (0 overworld, 1 test_valley listed, 2 test_dungeon of site 0), terrain per
// space, generation state, site 0's baseHeight is the natural height at its origin and its patch composes
TEST(Pipeline, ContentTestSetupWorld)
{
    World w(ContentTest::contentTestSetup(2));
    ASSERT_EQ(w.spaceCount(), 3u);
    EXPECT_EQ(w.spaceTemplate(0).name, "overworld");
    EXPECT_EQ(w.spaceTemplate(1).name, "test_valley");
    EXPECT_EQ(w.spaceTemplate(2).name, "test_dungeon");
    EXPECT_TRUE(w.spaceHasTerrain(0));
    EXPECT_TRUE(w.spaceHasTerrain(1));
    EXPECT_FALSE(w.spaceHasTerrain(2));
    EXPECT_TRUE(w.generation().nonLegacy());
    EXPECT_EQ(w.generationSeed(), 12345u);
    const SpaceGeneration& o = w.generation().space(0);
    EXPECT_EQ(w.generation().space(2).instance.siteKey, 0u);
    EXPECT_EQ(w.spaceSeed(2), hashCombine(hashCombine(12345, fnv1a64("site")), 0));
    const Site& ruin = o.plan->sites[0];
    EXPECT_EQ(ruin.prefab, "test_ruin");
    EXPECT_EQ(ruin.baseHeight, o.generator->height(o.pinned, *o.plan, ruin.x, ruin.z));
    // The patch centre (full weight, 2048 above the base): terrainHeight is exact
    EXPECT_EQ(w.terrainHeight(0, double(ruin.x), double(ruin.z)), double(ruin.baseHeight + 2048) / 1024.0);
    // Natural terrain outside the footprint
    const int64_t fx = ruin.x + 40, fz = ruin.z;
    EXPECT_EQ(w.terrain().sample(0, fx / 2, fz / 2), o.generator->height(o.pinned, *o.plan, fx, fz));
    std::cout << "test setup: " << o.plan->sites.size() << " overworld sites, "
              << w.generation().space(1).plan->sites.size() << " valley sites" << std::endl;
    EXPECT_GE(o.plan->sites.size(), 2u);
    EXPECT_LE(o.plan->sites.size(), 66u);
    // The dungeon's site: stamps from the plan (no chunk generation)
    ChunkId d;
    d.space = 2;
    d.x     = 8;
    d.z     = 8;
    EXPECT_EQ(w.generation().stampList(d).size(), 2u);
    EXPECT_EQ(w.generation().cache().generations(), 0u);
    EXPECT_NE(w.contentHash(d), 0u);
}

// Phase B: the prefab entities against the World's classes, components and meshes (fatal in every build, naming
// prefab, entity and field)
TEST(PrefabDeathTest, EntitiesCheckedAfterRenderAssets)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    // The copy outside the death statements: the parent removes it (a dying child runs no destructors)
    const PrefabCopy copy;
    auto withShrine = [&copy](const std::string& entity)
    {
        copy.write("test_shrine", R"({"format": 1, "footprint": [-8, -8, 8, 8], "entities": [)" + entity + "]}");
        WorldSetup s    = ContentTest::contentTestSetup(2);
        s.dataDirectory = copy.dir.string();
        World w(s);
    };
    // Valid: a registered mesh (renderAssets runs before phase B)
    withShrine(R"({"class": "TestMarker", "components": {"RenderInstance": {"mesh": "test/pillar"}}})");
    EXPECT_DEATH(withShrine(R"({"class": "Nope"})"), "prefab 'test_shrine' v1 .*entity 0 \\('Nope'\\): unknown class");
    EXPECT_DEATH(withShrine(R"({"class": "TestMarker", "components": {"Nope": {}}})"), "unknown component \"Nope\"");
    EXPECT_DEATH(withShrine(R"({"class": "TestMarker", "components": {"Health": {"hq": 1}}})"), "Health: .*hq");
    EXPECT_DEATH(withShrine(R"({"class": "TestMarker", "components": {"Transform": {}}})"), "Transform");
    EXPECT_DEATH(withShrine(R"({"class": "TestMarker", "components": {"Parent": {}}})"),
                 "Parent cannot be set by a prefab");
    EXPECT_DEATH(withShrine(R"({"class": "TestMarker", "components": {"NetOwner": {}}})"),
                 "NetOwner cannot be set by a prefab");
    EXPECT_DEATH(withShrine(R"({"class": "TestMarker", "components": {"SpaceLink": {}}})"),
                 "SpaceLink cannot be set by a prefab");
    EXPECT_DEATH(withShrine(R"({"class": "TestMarker", "components": {"RenderInstance": {"mesh": "test/none"}}})"),
                 "the mesh \"test/none\" is not registered");
}

TEST(PrefabDeathTest, PlanErrorsAreFatal)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const PrefabCopy copy;
    auto with = [&copy](const std::string& name, const std::string& json)
    {
        copy.write(name, json);
        WorldSetup s    = ContentTest::contentTestSetup(2);
        s.dataDirectory = copy.dir.string();
        World w(s);
    };
    // A terrain_patch in a prefab of an enclosed-space site
    EXPECT_DEATH(with("test_dungeon_rooms", R"({"format": 1, "footprint": [-8, -8, 8, 8],
        "terrain_patch": {"origin": [0, 0], "size": [1, 1], "heights": [1], "weights": [256]}})"),
                 "a terrain_patch in a space without terrain");
    // Nested spaces: an instance-space prefab with spaces
    EXPECT_DEATH(with("test_dungeon_rooms", R"({"format": 1, "footprint": [-8, -8, 8, 8],
        "spaces": [{"template": "test_dungeon"}]})"),
                 "no nested instances");
    EXPECT_DEATH(with("test_shrine", R"({"format": 1, "footprint": [-8, -8, 8, 8], "pieces": []})"),
                 "pieces: placed pieces in prefabs come with S30");
    EXPECT_DEATH(with("test_shrine", R"({"format": 1, "footprint": [-8, -8, 8, 8], "entities": [{"class": "TestMarker",
        "position": [0.1, 0, 0]}]})"),
                 "test_shrine.json: entities\\[0\\].position\\[0\\]: not an exact");
}

// ---- Stamping (S23a step 9) -----------------------------------------------------------------------------------------

namespace
{
// Destroys itself in onSpawn (a prefab component cannot be applied any more: logged and counted)
class TestVanish final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(ContentTest::ContentTestClass::Count);
    static constexpr const char* name = "TestVanish";
    void onSpawn() override { destroy(); }
};

// Throws in onSpawn while g_stampThrow is set
bool g_stampThrow = false;
class TestThrowOnSpawn final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(int(ContentTest::ContentTestClass::Count) + 1);
    static constexpr const char* name = "TestThrowOnSpawn";
    void onSpawn() override
    {
        if (g_stampThrow) throw std::runtime_error("TestThrowOnSpawn");
    }
};

void vanishClasses(ClassList& c)
{
    ContentTest::registerContentTestClasses(c);
    c.add<TestVanish>();
    c.add<TestThrowOnSpawn>();
}

Entity addPlayer(World& w, const Vec3& p, SpaceId space = 0)
{
    const Entity e = w.createIn(space);
    w.get<Transform>(e).setPosition(p);
    w.add<Player>(e);
    return e;
}

template <class T>
std::vector<Entity> entitiesOf(World& w)
{
    std::vector<Entity> out;
    w.viewAll<const Behaviour>().each(
        [&](Entity e, const Behaviour& b)
        {
            if (b.classId == T::id) out.push_back(e);
        });
    return out;
}
}  // namespace

// The stamp system is registered when some plan has sites or a generator is not legacy (and gets its profiler scope);
// legacy worlds have none
TEST(Stamp, RegisteredOnlyWithSites)
{
    World content(ContentTest::contentTestSetup(2));
    bool found = false;
    for (const auto& s : content.systemList().systems(Phase::Streaming)) found = found || s.name == "chunk_stamp";
    EXPECT_TRUE(found);
    ASSERT_FALSE(content.systemList().systems(Phase::Streaming).empty());
    EXPECT_EQ(content.systemList().systems(Phase::Streaming).front().name, "chunk_stamp");
    WorldSetup flat;
    World plain(flat);
    EXPECT_TRUE(plain.systemList().systems(Phase::Streaming).empty());
    EXPECT_TRUE(plain.contentRecords().empty());
    // Regression (PR review): a non-legacy world without sites had no stamp system, so a load whose saved pins bring
    // plans with sites never stamped them
    WorldSetup noSites;
    noSites.seed       = 1;
    noSites.generators = [](GeneratorList& l) { l.add(1, slotGenerator(1)); };
    SpaceTemplate t;
    t.generator                   = 1;
    noSites.config.spaceTemplates = {t};
    World generated(noSites);
    EXPECT_FALSE(generated.generation().plans().anySites);
    ASSERT_FALSE(generated.systemList().systems(Phase::Streaming).empty());
    EXPECT_EQ(generated.systemList().systems(Phase::Streaming).front().name, "chunk_stamp");
}

// During a step: a class that adds Health in onSpawn gets the prefab's value (no "already has" assert); a class that
// adds nothing gets the prefab's component at the phase-7 flush through addDefaultById; positions and yaws exact; the
// entrance's SpaceLink targets the site's instance; the record and the stamped entry; a second visit stamps nothing
TEST(Stamp, DuringAStep)
{
    World w(ContentTest::contentTestSetup(2));
    const SpaceGeneration& o = w.generation().space(0);
    const Site& ruin         = o.plan->sites[0];
    // 2 km away: nothing
    const Entity player = addPlayer(w, Vec3(3100, 0, 1300));
    w.step(2);
    EXPECT_TRUE(entitiesOf<ContentTest::TestChest>(w).empty());
    // Next to the site (scenario worlds lower fullRadius to 128 m; this World keeps 1000 m)
    w.get<Transform>(player).setPosition(Vec3(1100, 0, 1200));
    w.step(1);
    const auto chests = entitiesOf<ContentTest::TestChest>(w);
    ASSERT_EQ(chests.size(), 1u);
    const Transform& t = w.read<Transform>(chests[0]);
    // Local (2.5, 0, -1.25) rotated by one quarter turn: (-1.25, -2.5)
    EXPECT_EQ(t.position(), Vec3(double(ruin.x) - 1.25, double(ruin.baseHeight) / 1024.0, double(ruin.z) - 2.5));
    EXPECT_EQ(t.rotation().coeffs(), contentRotation(32768).coeffs());
    EXPECT_EQ(w.read<Health>(chests[0]).hp, 50);
    const auto markers = entitiesOf<ContentTest::TestMarker>(w);
    ASSERT_GE(markers.size(), 1u);
    // The ruin's marker (hp 7) and the shrines' (hp 3) within 1000 m
    bool ruinMarker = false;
    for (Entity m : markers)
    {
        ASSERT_TRUE(w.has<Health>(m));
        ruinMarker = ruinMarker || w.read<Health>(m).hp == 7;
    }
    EXPECT_TRUE(ruinMarker);
    const auto entrances = entitiesOf<ContentTest::TestEntrance>(w);
    ASSERT_EQ(entrances.size(), 1u);
    const SpaceLink& link = w.read<SpaceLink>(entrances[0]);
    EXPECT_EQ(link.target, 2);
    EXPECT_EQ(link.arrival, Vec3(2176.5, 0, 2180));
    ChunkId c;
    c.x = 4;
    c.z = 5;
    EXPECT_TRUE(w.contentRecords().isStamped(c));
    ASSERT_NE(w.contentRecords().find(c), nullptr);
    EXPECT_EQ(w.contentRecords().find(c)->placementVersion, 2);
    const uint64_t entities = w.contentRecords().stats().entities;
    w.get<Transform>(player).setPosition(Vec3(5000, 0, 5000));
    w.step(2);
    w.get<Transform>(player).setPosition(Vec3(1100, 0, 1200));
    w.step(2);
    EXPECT_EQ(entitiesOf<ContentTest::TestChest>(w).size(), 1u);
    EXPECT_EQ(w.contentRecords().stats().entities, entities);
    EXPECT_EQ(w.contentRecords().stats().failedComponents, 0u);
}

// Regression (PR review): the candidate range started at floor((x - r) / 256), so a site chunk whose square only
// touches the disc (distance exactly fullRadius, accepted by the distance test) was skipped on its lower side; a
// coordinate of 1e30 overflowed the conversion to int64_t
TEST(Stamp, ChunkTouchingTheRadius)
{
    World w(ContentTest::contentTestSetup(2));
    const SpaceGeneration& o = w.generation().space(0);
    const Site& ruin         = o.plan->sites[0];
    ChunkId c;
    c.x = int32_t(ruin.x / 256);
    c.z = int32_t(ruin.z / 256);
    ASSERT_FALSE(w.generation().stampList(c).empty());
    // The chunk's right edge is exactly fullRadius away
    const double x = double(c.x + 1) * 256 + o.settings.fullRadius;
    ASSERT_LT(x, double(o.settings.sizeChunks) * 256);
    const Entity player = addPlayer(w, Vec3(1e30, 0, -1e30));
    w.step(1);
    EXPECT_FALSE(w.contentRecords().isStamped(c));
    w.get<Transform>(player).setPosition(Vec3(x, 0, double(c.z) * 256 + 128));
    w.step(1);
    EXPECT_TRUE(w.contentRecords().isStamped(c));
}

// Between steps the spawn applies at once; a stamp whose onSpawn destroys the entity is logged and counted
TEST(Stamp, BetweenStepsAndDestroyedInOnSpawn)
{
    World w(ContentTest::contentTestSetup(2));
    const ClassList::ClassInfo* info = w.classList().find("TestMarker");
    ASSERT_NE(info, nullptr);
    bool after     = false;
    const Entity e = w.spawnByClassDeferred(
        *info, 0, Vec3(1, 2, 3), contentRotation(16384),
        [&](World& world, Entity x)
        {
            std::string error;
            EXPECT_TRUE(mergeComponentValues(world, x, Health::id, json11::Json::object{{"hp", 9}}, error)) << error;
            after = true;
        });
    EXPECT_TRUE(after);
    ASSERT_TRUE(w.alive(e));
    EXPECT_EQ(w.read<Health>(e).hp, 9);
    EXPECT_EQ(w.read<Transform>(e).position(), Vec3(1, 2, 3));

    PrefabCopy copy;
    copy.write("test_shrine", R"({"format": 1, "footprint": [-8, -8, 8, 8],
        "entities": [{"class": "TestVanish", "components": {"Health": {"hp": 4}}}]})");
    WorldSetup s    = ContentTest::contentTestSetup(2);
    s.dataDirectory = copy.dir.string();
    s.classes       = vanishClasses;
    World v(s);
    const Site& shrine = v.generation().space(0).plan->sites[1];
    addPlayer(v, Vec3(double(shrine.x), 0, double(shrine.z)));
    v.step(1);
    EXPECT_GE(v.contentRecords().stats().failedComponents, 1u);
    EXPECT_TRUE(entitiesOf<TestVanish>(v).empty());
}

// A spawn of a chunk's stamp that throws rolls back the chunk's earlier spawns and leaves the chunk unstamped; the next
// step stamps it whole (regression: the stamped entry applied first, so the chunk stayed stamped with only the entities
// spawned before the exception and was never completed)
TEST(Stamp, ThrowingSpawnLeavesTheChunkUnstamped)
{
    PrefabCopy copy;
    copy.write("test_shrine", R"({"format": 1, "footprint": [-8, -8, 8, 8],
        "entities": [{"class": "TestMarker", "components": {"Health": {"hp": 11}}}, {"class": "TestThrowOnSpawn"}]})");
    WorldSetup s    = ContentTest::contentTestSetup(2);
    s.dataDirectory = copy.dir.string();
    s.classes       = vanishClasses;
    World v(s);
    const Site& shrine = v.generation().space(0).plan->sites[1];
    addPlayer(v, Vec3(double(shrine.x), 0, double(shrine.z)));
    ChunkId c;
    c.x                = int32_t(shrine.x / 256);
    c.z                = int32_t(shrine.z / 256);
    auto shrineMarkers = [&]()
    {
        size_t n = 0;
        for (Entity m : entitiesOf<ContentTest::TestMarker>(v)) n += v.has<Health>(m) && v.read<Health>(m).hp == 11;
        return n;
    };
    g_stampThrow = true;
    EXPECT_THROW(v.step(1), std::runtime_error);
    g_stampThrow = false;
    EXPECT_FALSE(v.contentRecords().isStamped(c));
    EXPECT_EQ(shrineMarkers(), 0u);
    EXPECT_TRUE(entitiesOf<TestThrowOnSpawn>(v).empty());
    v.step(1);
    EXPECT_TRUE(v.contentRecords().isStamped(c));
    EXPECT_GE(entitiesOf<TestThrowOnSpawn>(v).size(), 1u);
    EXPECT_EQ(shrineMarkers(), entitiesOf<TestThrowOnSpawn>(v).size());
    EXPECT_EQ(v.contentRecords().stats().entities,
              entitiesOf<ContentTest::TestMarker>(v).size() + entitiesOf<TestThrowOnSpawn>(v).size() +
                  entitiesOf<ContentTest::TestChest>(v).size() + entitiesOf<ContentTest::TestEntrance>(v).size());
}

// ---- Records and saves (S23a step 10) -------------------------------------------------------------------------------

namespace
{
// A generator that delegates to TestTerrainGenerator under another name or with another slot table (load errors)
class ChangedGenerator final : public Generator
{
   public:
    ChangedGenerator(std::string name, bool renameSlot) : nameText(std::move(name)), renameSlot(renameSlot) {}
    const char* name() const override { return nameText.c_str(); }
    uint16_t version() const override { return base.version(); }
    std::vector<GeneratorSlot> slots() const override
    {
        auto s = base.slots();
        if (renameSlot) s[1].name = "test_stone";
        return s;
    }
    void plan(uint16_t v, WorldPlan& p) const override { base.plan(v, p); }
    int32_t height(uint16_t v, const WorldPlan& p, int64_t x, int64_t z) const override
    {
        return base.height(v, p, x, z);
    }
    bool candidate(uint16_t v, const WorldPlan& p, ChunkId c, uint32_t s, uint32_t n, Candidate& o) const override
    {
        return base.candidate(v, p, c, s, n, o);
    }
    uint32_t candidatesPerChunk(uint32_t s) const override { return base.candidatesPerChunk(s); }
    uint32_t keptCount(uint32_t s, int l) const override { return base.keptCount(s, l); }
    void grassMask(uint16_t v, const WorldPlan& p, ChunkId c, GrassMask& o) const override
    {
        base.grassMask(v, p, c, o);
    }

   private:
    ContentTest::TestTerrainGenerator base{2};
    std::string nameText;
    bool renameSlot = false;
};

// Stamps site 0 (chunk (4, 5)) of a content world between steps
void stampSiteZero(World& w)
{
    const Entity p = addPlayer(w, Vec3(1100, 0, 1200));
    w.step(1);
    w.destroy(p);
}

ChunkId chunkAt(int32_t x, int32_t z, SpaceId space = 0)
{
    ChunkId c;
    c.space = space;
    c.x     = x;
    c.z     = z;
    return c;
}

std::string committedSection(int entry, const std::string& chunk, const std::string& section)
{
    json11::Json j;
    std::string error;
    EXPECT_TRUE(readJsonFile(contentHashFile(), j, error)) << error;
    return j["setups"][size_t(entry)]["chunks"][chunk][section].string_value();
}

LoadResult loadInto(World& w, const std::vector<uint8_t>& blob)
{
    const auto before = w.save();
    LoadResult r      = w.tryLoad(blob);
    if (!r.committed)
    {
        EXPECT_EQ(w.save(), before) << "a rejected load changed the world";
    }
    return r;
}
}  // namespace

// GENR / CREC round trip; records only for stamped chunks (a visited chunk without sites and an edited chunk get none);
// a flat world writes neither; world.hash changes with a record and is equal after save / load; diffSaves names records
TEST(ContentSave, RoundTripAndRecords)
{
    World a(ContentTest::contentTestSetup(2));
    const uint64_t h0 = a.hash();
    // A visited chunk without sites and an edited chunk: no record
    const Entity visitor = addPlayer(a, Vec3(20000, 0, 20000));
    a.step(1);
    a.destroy(visitor);
    TerrainBrush b;
    b.center   = Vec3(300, 0, 300);
    b.radius   = 10;
    b.strength = 2;
    a.editTerrain(0, b);
    EXPECT_TRUE(a.contentRecords().records().empty());
    stampSiteZero(a);
    EXPECT_NE(a.hash(), h0);
    ASSERT_TRUE(a.contentRecords().isStamped(chunkAt(4, 5)));
    for (const auto& [c, r] : a.contentRecords().records())
    {
        EXPECT_TRUE(a.contentRecords().isStamped(c));
        EXPECT_EQ(r.placementVersion, 2);
    }
    const auto blob = a.save();
    ParsedSave parsed;
    std::string error;
    ASSERT_TRUE(parseSave(blob, parsed, error)) << error;
    ASSERT_TRUE(parsed.generation);
    ASSERT_TRUE(parsed.chunkRecords);
    EXPECT_EQ(parsed.generation->seed, 12345u);
    EXPECT_EQ(parsed.generation->spaceCount, 3u);

    World b2(ContentTest::contentTestSetup(2));
    const LoadResult r = b2.tryLoad(blob);
    ASSERT_TRUE(r.ok()) << r.error;
    EXPECT_EQ(b2.contentRecords().records().size(), a.contentRecords().records().size());
    EXPECT_EQ(b2.contentRecords().stamped(), a.contentRecords().stamped());
    EXPECT_EQ(b2.hash(), a.hash());
    EXPECT_EQ(World::diff(a, b2), "");
    // No second stamp after the load
    stampSiteZero(b2);
    EXPECT_EQ(entitiesOf<ContentTest::TestChest>(b2).size(), 1u);
    // A record difference is named
    World c(ContentTest::contentTestSetup(2));
    EXPECT_NE(World::diff(a, c).find("record space 0 chunk (4, 5)"), std::string::npos) << World::diff(a, c);

    // A flat world writes neither section
    World flat{WorldSetup{}};
    ParsedSave pf;
    ASSERT_TRUE(parseSave(flat.save(), pf, error)) << error;
    EXPECT_FALSE(pf.generation);
    EXPECT_FALSE(pf.chunkRecords);
    EXPECT_FALSE(flat.writesGenerationState());
}

// Load errors leave the world unchanged: another seed, a newer version, a changed slot table, another generator name,
// a changed released plan, another space list
TEST(ContentSave, LoadErrors)
{
    World a(ContentTest::contentTestSetup(2));
    stampSiteZero(a);
    const auto blob = a.save();
    {
        WorldSetup s = ContentTest::contentTestSetup(2);
        s.seed       = 7;
        World w(s);
        const LoadResult r = loadInto(w, blob);
        EXPECT_FALSE(r.committed);
        EXPECT_EQ(r.error, "save seed 12345, setup seed 7: the seed is part of the setup");
    }
    {
        // Pinned 3 > the build's 2
        WorldSetup s3 = ContentTest::contentTestSetup(1);
        s3.generators = [](GeneratorList& l)
        {
            l.add(ContentTest::kTestTerrainGenerator, std::make_shared<ContentTest::TestTerrainGenerator>(3));
            l.add(ContentTest::kTestDungeonGenerator,
                  std::make_shared<AuthoredSpaceGenerator>("test_dungeon", "test_dungeon_rooms", 1));
        };
        World v3(s3);
        World w(ContentTest::contentTestSetup(2));
        const LoadResult r = loadInto(w, v3.save());
        EXPECT_FALSE(r.committed);
        EXPECT_NE(r.error.find("save from a newer generator test_terrain v3"), std::string::npos) << r.error;
    }
    auto changed = [&](const std::string& name, bool renameSlot)
    {
        static std::string gName;
        static bool gRename = false;
        gName               = name;
        gRename             = renameSlot;
        WorldSetup s        = ContentTest::contentTestSetup(2);
        s.generators        = [](GeneratorList& l)
        {
            l.add(ContentTest::kTestTerrainGenerator, std::make_shared<ChangedGenerator>(gName, gRename));
            l.add(ContentTest::kTestDungeonGenerator,
                  std::make_shared<AuthoredSpaceGenerator>("test_dungeon", "test_dungeon_rooms", 1));
        };
        World w(s);
        return loadInto(w, blob);
    };
    {
        const LoadResult r = changed("test_terrain", true);
        EXPECT_FALSE(r.committed);
        EXPECT_NE(r.error.find("slot 1 is 'test_rock' in the save and 'test_stone'"), std::string::npos) << r.error;
    }
    {
        const LoadResult r = changed("renamed", false);
        EXPECT_FALSE(r.committed);
        EXPECT_EQ(r.error, "space template 'overworld' used generator 'test_terrain', now 'renamed'");
    }
    {
        // A released v1 that changed: same pins, another plan
        World v1(ContentTest::contentTestSetup(1));
        World w(ContentTest::contentTestSetup(1, nullptr, ContentTest::Variant::ReleasedChangeV1));
        const LoadResult r = loadInto(w, v1.save());
        EXPECT_FALSE(r.committed);
        EXPECT_NE(r.error.find("produce another plan than the saved one"), std::string::npos) << r.error;
    }
    {
        // TableChangeV2: a v1 save into the v2 binary, whose own list has another space
        World v1(ContentTest::contentTestSetup(1, nullptr, ContentTest::Variant::TableChangeV2));
        World w(ContentTest::contentTestSetup(2, nullptr, ContentTest::Variant::TableChangeV2));
        ASSERT_EQ(w.spaceCount(), 4u);
        const LoadResult r = loadInto(w, v1.save());
        EXPECT_FALSE(r.committed);
        EXPECT_NE(r.error.find("space"), std::string::npos) << r.error;
        std::cout << "TableChangeV2: " << r.error << std::endl;
    }
}

// A world of v1 (own cache) with site 0 stamped, loaded into the v2 binary (own cache): plan v1 kept, the stamped chunk
// at placement v1 (its tier1 hash is the committed v1 entry's), an unrecorded chunk at v2 (the committed v2 entry's)
TEST(ContentSave, OlderVersionKeepsRecordedPlacement)
{
    World v1(ContentTest::contentTestSetup(1));
    stampSiteZero(v1);
    World w(ContentTest::contentTestSetup(2));
    const LoadResult r = w.tryLoad(v1.save());
    ASSERT_TRUE(r.ok()) << r.error;
    EXPECT_EQ(w.generation().plans().pinned.at("test_terrain"), 1);
    EXPECT_EQ(w.generation().plans().current.at("test_terrain"), 2);
    ASSERT_NE(w.contentRecords().find(chunkAt(4, 5)), nullptr);
    EXPECT_EQ(w.contentRecords().find(chunkAt(4, 5))->placementVersion, 1);
    EXPECT_EQ(hexU64(w.generation().chunkPlacement(chunkAt(4, 5), 1)->hashes.tier1),
              committedSection(0, "0:4:5", "tier1"));
    EXPECT_EQ(hexU64(w.generation().chunkPlacement(chunkAt(0, 0), 2)->hashes.tier1),
              committedSection(1, "0:0:0", "tier1"));
    // A save with fewer templates loads into a setup with an appended template (the prefix rule)
    WorldSetup more = ContentTest::contentTestSetup(2);
    SpaceTemplate extra;
    extra.name = "extra";
    more.config.spaceTemplates.push_back(extra);
    World m(more);
    ASSERT_EQ(m.spaceCount(), 4u);
    EXPECT_EQ(m.spaceTemplate(3).name, "extra");
    const LoadResult rm = m.tryLoad(v1.save());
    EXPECT_TRUE(rm.ok()) << rm.error;
}

// PlanChangeV2: a v1 save loads into the v2 binary (same space table, other heights): the plans are replaced, the
// sink's terrain cleared and the pyramid sent again; terrainHeight at the moved site and an edited sample there equal
// v1's
TEST(ContentSave, RepinDropsDerivedTerrain)
{
    World a(ContentTest::contentTestSetup(1, nullptr, ContentTest::Variant::PlanChangeV2));
    TerrainBrush brush;
    brush.center   = Vec3(1536, 0, 1540);
    brush.radius   = 6;
    brush.strength = 3;
    a.editTerrain(0, brush);
    const double siteHeight = a.terrainHeight(0, 1536, 1536);
    const int32_t edited    = a.terrainSample(0, 768, 770);
    ASSERT_NE(edited, a.terrain().sample(0, 768, 770));
    World b(ContentTest::contentTestSetup(2, nullptr, ContentTest::Variant::PlanChangeV2));
    ASSERT_NE(b.terrainHeight(0, 1536, 1536), siteHeight);
    RecordingRenderSink sink;
    b.setRenderSink(&sink);
    const Vec3 camera(1536, 200, 1536);
    ASSERT_TRUE(b.terrain().completeView(0, camera));
    const uint64_t pyramids = sink.counters().setTerrainPyramid;
    const LoadResult r      = b.tryLoad(a.save());
    ASSERT_TRUE(r.ok()) << r.error;
    EXPECT_EQ(sink.counters().clearTerrain, 1u);
    EXPECT_EQ(b.terrainHeight(0, 1536, 1536), siteHeight);
    EXPECT_EQ(b.terrainSample(0, 768, 770), edited);
    ASSERT_TRUE(b.terrain().completeView(0, camera));
    EXPECT_GT(sink.counters().setTerrainPyramid, pyramids);
    b.setRenderSink(nullptr);
}

namespace
{
// test_terrain whose listed instance (test_valley) gets a seed per plan version
class ListedSeedGenerator final : public Generator
{
   public:
    explicit ListedSeedGenerator(uint16_t maxVersion) : base(maxVersion) {}
    const char* name() const override { return base.name(); }
    uint16_t version() const override { return base.version(); }
    std::vector<GeneratorSlot> slots() const override { return base.slots(); }
    void plan(uint16_t v, WorldPlan& p) const override
    {
        base.plan(v, p);
        for (SpaceInstance& i : p.spaces) i.seed = hashCombine(i.seed, v);
    }
    int32_t height(uint16_t v, const WorldPlan& p, int64_t x, int64_t z) const override
    {
        return base.height(v, p, x, z);
    }
    bool candidate(uint16_t v, const WorldPlan& p, ChunkId c, uint32_t s, uint32_t n, Candidate& o) const override
    {
        return base.candidate(v, p, c, s, n, o);
    }
    uint32_t candidatesPerChunk(uint32_t s) const override { return base.candidatesPerChunk(s); }
    uint32_t keptCount(uint32_t s, int l) const override { return base.keptCount(s, l); }
    void grassMask(uint16_t v, const WorldPlan& p, ChunkId c, GrassMask& o) const override
    {
        base.grassMask(v, p, c, o);
    }

   private:
    ContentTest::TestTerrainGenerator base;
};

template <uint16_t MaxVersion>
void listedSeedGenerators(GeneratorList& l)
{
    l.add(ContentTest::kTestTerrainGenerator, std::make_shared<ListedSeedGenerator>(MaxVersion));
    l.add(ContentTest::kTestDungeonGenerator,
          std::make_shared<AuthoredSpaceGenerator>("test_dungeon", "test_dungeon_rooms", 1));
}
}  // namespace

// Regression (PR review): a load whose saved pins give a listed instance another seed (same template indices) kept
// the construction's seed in World::spaceSeed
TEST(ContentSave, RepinUpdatesSpaceSeeds)
{
    WorldSetup s1 = ContentTest::contentTestSetup(1);
    s1.generators = listedSeedGenerators<1>;
    WorldSetup s2 = ContentTest::contentTestSetup(2);
    s2.generators = listedSeedGenerators<2>;
    World a(s1);
    World b(s2);
    ASSERT_EQ(a.generation().space(1).instance.origin, SpaceInstance::Listed);
    ASSERT_NE(a.spaceSeed(1), b.spaceSeed(1));
    EXPECT_EQ(b.spaceSeed(0), a.spaceSeed(0));
    const LoadResult r = b.tryLoad(a.save());
    ASSERT_TRUE(r.ok()) << r.error;
    EXPECT_EQ(b.spaceSeed(1), a.spaceSeed(1));
    EXPECT_EQ(b.spaceSeed(2), a.spaceSeed(2));
    // An unchanged seed stays valid: a v2 save into a v2 world
    World c(s2);
    ASSERT_TRUE(c.tryLoad(World(s2).save()).ok());
    EXPECT_EQ(c.spaceSeed(1), World(s2).spaceSeed(1));
}

// A pre-S23 save with TERR into a setup whose template moved to a non-legacy generator: a load error
TEST(ContentSave, PreS23SaveNeedsLegacyGenerators)
{
    WorldSetup hook;
    hook.seed = 12345;
    SpaceTemplate t;
    t.name                     = "overworld";
    t.generator                = 1;
    t.seaLevel                 = 0;
    hook.config.spaceTemplates = {t};
    hook.generators            = [](GeneratorList& l)
    { l.add(1, std::make_shared<HookGenerator>("old", [](uint64_t, int64_t x, int64_t) { return int32_t(x); })); };
    World old(hook);
    TerrainBrush b;
    b.center   = Vec3(300, 0, 300);
    b.radius   = 5;
    b.strength = 1;
    old.editTerrain(0, b);
    ParsedSave p;
    std::string error;
    const auto blob = old.save();
    ASSERT_TRUE(parseSave(blob, p, error));
    EXPECT_FALSE(p.generation);
    EXPECT_TRUE(p.terrain);
    // The same template 0 (and two appended ones) on the test generator
    World w(ContentTest::contentTestSetup(2));
    const LoadResult r = loadInto(w, blob);
    EXPECT_FALSE(r.committed);
    EXPECT_EQ(r.error, "save from before S23: template 'overworld' now uses generator 'test_terrain'");
}

// The committed save of S23a (GENR, CREC with stamped chunks, an edited chunk) keeps loading
TEST(ContentSave, CommittedSaveLoads)
{
    const std::string path = std::string(SAIGA_PROJECT_SOURCE_DIR) + "/data/tests/engine/save_s23a.bin";
    const char* update     = std::getenv("SAIGA_UPDATE_GOLDEN");
    if (update && std::string(update) == "1")
    {
        World w(ContentTest::contentTestSetup(2));
        stampSiteZero(w);
        TerrainBrush b;
        b.center   = Vec3(1110, 0, 1290);
        b.radius   = 8;
        b.strength = 2;
        w.editTerrain(0, b);
        w.step(3);
        const auto blob = w.save();
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(blob.data()), std::streamsize(blob.size()));
        out.close();
        ASSERT_TRUE(out) << "cannot write " << path;
        GTEST_SKIP() << "wrote " << path;
    }
    std::ifstream f(path, std::ios::binary);
    const std::vector<uint8_t> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    ASSERT_FALSE(blob.empty()) << path;
    World w(ContentTest::contentTestSetup(2));
    const LoadResult r = w.tryLoad(blob);
    ASSERT_TRUE(r.ok()) << r.error;
    EXPECT_TRUE(w.contentRecords().isStamped(chunkAt(4, 5)));
    EXPECT_EQ(entitiesOf<ContentTest::TestChest>(w).size(), 1u);
    EXPECT_GT(w.terrainEdits().chunkCount(), 0u);
    // It saves again with the same state
    World again(ContentTest::contentTestSetup(2));
    ASSERT_TRUE(again.tryLoad(w.save()).ok());
    EXPECT_EQ(again.hash(), w.hash());
}

// ---- GenerationState (S23a step 11) ---------------------------------------------------------------------------------

namespace
{
// Records whether the client had checked the host's generation state when it received this object
std::atomic<int> g_probeChecked{-1};
class TestProbe final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(ContentTest::ContentTestClass::Count);
    static constexpr const char* name = "TestProbe";
    void onReplicatedSpawn() override { g_probeChecked = world().generationChecked() ? 1 : 0; }
};
void probeClasses(ClassList& c)
{
    ContentTest::registerContentTestClasses(c);
    c.add<TestProbe>();
}

// A host and one client over loopback, each with its own setup (and cache); the client joins at construction
struct GenNet
{
    std::vector<std::unique_ptr<LoopbackTransport>> t;
    std::unique_ptr<World> host, client;
    GenNet(WorldSetup hostSetup, WorldSetup clientSetup, std::function<void(World&)> beforeJoin = nullptr)
    {
        t                   = LoopbackTransport::createNetwork(2, LoopbackOptions());
        hostSetup.transport = t[0].get();
        host                = std::make_unique<World>(hostSetup);
        if (beforeJoin) beforeJoin(*host);
        clientSetup.transport = t[1].get();
        clientSetup.role      = NetRole::Client;
        clientSetup.localPeer = 1;
        client                = std::make_unique<World>(clientSetup);
    }
    void step(int n = 3)
    {
        for (int k = 0; k < n; ++k)
        {
            host->step();
            client->step();
        }
    }
};

std::string logSince(uint64_t sequence)
{
    std::string out;
    for (const auto& l : engineLog().since(sequence)) out += "[" + l.category + "] " + l.message + "\n";
    return out;
}

// The client refuses: counted, nothing applied, generationMismatch set, the log names the difference
void expectRefused(GenNet& n, const std::string& difference)
{
    const uint64_t log = engineLog().lastSequence();
    n.step();
    EXPECT_TRUE(n.client->generationMismatch());
    EXPECT_FALSE(n.client->generationChecked());
    EXPECT_GE(n.client->netStats().rejected, 1u);
    EXPECT_NE(logSince(log).find(difference), std::string::npos) << logSince(log);
}
}  // namespace

// Equal setups pass; on a join the state precedes the spawn set (an object of the instance space arrives after the
// check), after a host load it follows the Resync
TEST(GenerationState, EqualSetupsPass)
{
    WorldSetup s = ContentTest::contentTestSetup(2);
    s.classes    = probeClasses;
    // The client's own setup (its own content cache: it reproduces the host's plans independently)
    WorldSetup cs  = ContentTest::contentTestSetup(2);
    cs.classes     = probeClasses;
    g_probeChecked = -1;
    GenNet n(s, cs, [](World& h) { h.spawnIn<TestProbe>(2, Vec3(2176, 0, 2176)); });
    n.step();
    EXPECT_TRUE(n.client->generationChecked());
    EXPECT_FALSE(n.client->generationMismatch());
    EXPECT_EQ(n.host->replication().stats().generationStatesSent, 1u);
    EXPECT_EQ(g_probeChecked, 1);
    EXPECT_EQ(n.client->netStats().rejected, 0u);
    // A host load: Resync, then the state again
    g_probeChecked  = -1;
    const auto blob = n.host->save();
    ASSERT_TRUE(n.host->tryLoad(blob).ok());
    n.step();
    EXPECT_EQ(n.host->replication().stats().generationStatesSent, 2u);
    EXPECT_TRUE(n.client->generationChecked());
    EXPECT_EQ(g_probeChecked, 1);
}

TEST(GenerationState, MismatchesAreRefused)
{
    {
        WorldSetup c = ContentTest::contentTestSetup(2);
        c.seed       = 7;
        GenNet n(ContentTest::contentTestSetup(2), c);
        expectRefused(n, "host seed 12345, this client's setup seed 7");
    }
    {
        // The host loaded a v1 save (pins v1); a fresh v2 client pins v2
        World v1(ContentTest::contentTestSetup(1));
        const auto blob = v1.save();
        GenNet n(ContentTest::contentTestSetup(2), ContentTest::contentTestSetup(2),
                 [&](World& h) { ASSERT_TRUE(h.tryLoad(blob).ok()); });
        expectRefused(n, "host uses generator test_terrain pinned v1, this client pins v2");
    }
    {
        // A client binary whose newest version is below the host's current one (the pins agree: v1)
        World v1(ContentTest::contentTestSetup(1));
        const auto blob = v1.save();
        GenNet n(ContentTest::contentTestSetup(2), ContentTest::contentTestSetup(1),
                 [&](World& h) { ASSERT_TRUE(h.tryLoad(blob).ok()); });
        expectRefused(n, "host places generator test_terrain at v2, this client's newest version is v1");
    }
    {
        WorldSetup c = ContentTest::contentTestSetup(2);
        c.generators = [](GeneratorList& l)
        {
            l.add(ContentTest::kTestTerrainGenerator, std::make_shared<ChangedGenerator>("test_terrain", true));
            l.add(ContentTest::kTestDungeonGenerator,
                  std::make_shared<AuthoredSpaceGenerator>("test_dungeon", "test_dungeon_rooms", 1));
        };
        GenNet n(ContentTest::contentTestSetup(2), c);
        expectRefused(n, "generator test_terrain: the host's slot table differs");
    }
    {
        GenNet n(ContentTest::contentTestSetup(2),
                 ContentTest::contentTestSetup(2, nullptr, ContentTest::Variant::PlanChangeV2));
        expectRefused(n, "the host's plans differ from this client's");
    }
}

// Regression (implementation review of S23a): after a refused GenerationState the client applied the host's terrain
// edits (deltas against a generation it does not have); now they are rejected like S23b's record messages
TEST(GenerationState, TerrainEditsAfterARefusalAreRejected)
{
    WorldSetup c = ContentTest::contentTestSetup(2);
    c.seed       = 7;
    TerrainBrush b;
    b.center   = Vec3(300, 0, 300);
    b.radius   = 10;
    b.strength = 2;
    // An edit before the join (the chunk snapshot) and one after the refusal (a TerrainModification)
    GenNet n(ContentTest::contentTestSetup(2), c, [&](World& h) { h.editTerrain(0, b); });
    expectRefused(n, "host seed 12345, this client's setup seed 7");
    const uint64_t rejected = n.client->netStats().rejected;
    b.center                = Vec3(600, 0, 600);
    n.host->editTerrain(0, b);
    n.step();
    EXPECT_GT(n.client->netStats().rejected, rejected);
    EXPECT_EQ(n.client->terrainEdits().chunkCount(), 0u);
}

// Regression: a client whose setup writes generation state accepted a flat host's join (no GenerationState arrives)
// and generated its own, other terrain; the spawn set without a preceding GenerationState is refused
TEST(GenerationState, MissingStateIsRefused)
{
    WorldSetup h = WorldSetup{};
    h.classes    = probeClasses;
    WorldSetup c = ContentTest::contentTestSetup(2);
    c.classes    = probeClasses;
    GenNet n(h, c, [](World& host) { host.spawnIn<TestProbe>(0, Vec3(100, 0, 100)); });
    expectRefused(n, "the host sent no generation state");
    EXPECT_EQ(n.host->replication().stats().generationStatesSent, 0u);
}

// Regression (PR review): a host that sends no GenerationState (a legacy setup) and no spawn set left a client whose
// setup writes generation state unchecked, and the host's terrain states were applied (a join's chunk snapshot, and
// after a Resync of a host load, which also resets the refusal, the chunk states arrive before the spawn set)
TEST(GenerationState, TerrainBeforeAStateIsRejected)
{
    TerrainBrush b;
    b.center   = Vec3(300, 0, 300);
    b.radius   = 10;
    b.strength = 2;
    GenNet n(WorldSetup{}, ContentTest::contentTestSetup(2), [&](World& h) { h.editTerrain(0, b); });
    n.step();
    EXPECT_GT(n.host->terrainEdits().chunkCount(), 0u);
    EXPECT_GE(n.client->netStats().rejected, 1u);
    EXPECT_EQ(n.client->terrainEdits().chunkCount(), 0u);
    const auto blob = n.host->save();
    ASSERT_TRUE(n.host->tryLoad(blob).ok());
    const uint64_t rejected = n.client->netStats().rejected;
    n.step();
    EXPECT_GT(n.client->netStats().rejected, rejected);
    EXPECT_EQ(n.client->terrainEdits().chunkCount(), 0u);
}

namespace
{
// A space-state component (one state entity per space, bound NetIds)
struct TestSpaceZone
{
    static constexpr const char* name = "TestSpaceZone";
    static constexpr ComponentId id   = ComponentId(kFirstGameComponent);
    static constexpr uint32_t version = 1;
    int32_t level                     = 0;
    template <class V>
    void visit(V& v)
    {
        v("level", level);
    }
};
void spaceZoneComponents(Registry& r)
{
    r.registerSpaceState<TestSpaceZone>(Replicate::Host);
}
}  // namespace

// Regression: after a refused GenerationState the host's space-state entity of a space the client does not have (the
// host has 3 spaces, the flat client 1) ended the client process (SAIGA_EXIT_ERROR); now the spawn is refused
TEST(GenerationState, StateOfAMissingSpaceAfterARefusal)
{
    WorldSetup h = ContentTest::contentTestSetup(2);
    h.components = spaceZoneComponents;
    WorldSetup c = WorldSetup{};
    c.components = spaceZoneComponents;
    GenNet n(h, c, [](World& host) { host.get<TestSpaceZone>(host.spaceStateEntity(0)).level = 5; });
    expectRefused(n, "names a state entity this client does not have (after a refused GenerationState)");
    ASSERT_EQ(n.client->spaceCount(), 1u);
    // The state of the space both have is applied
    EXPECT_EQ(n.client->read<TestSpaceZone>(n.client->spaceStateEntity(0)).level, 5);
}

// Flat and hook worlds send no GenerationState; the non-integer warning is logged once
TEST(GenerationState, LegacyWorldsSendNothing)
{
    {
        GenNet n(WorldSetup{}, WorldSetup{});
        n.step();
        EXPECT_EQ(n.host->replication().stats().generationStatesSent, 0u);
        EXPECT_FALSE(n.client->generationChecked());
        EXPECT_FALSE(n.client->generationMismatch());
    }
    WorldSetup hook;
    SpaceTemplate t;
    t.generator                = 1;
    hook.config.spaceTemplates = {t};
    hook.generators            = [](GeneratorList& l)
    {
        l.add(1, std::make_shared<HookGenerator>(
                     "float_hook", [](uint64_t, int64_t x, int64_t) { return int32_t(x); }, std::nullopt, false));
    };
    const uint64_t log = engineLog().lastSequence();
    LoopbackOptions o;
    auto t3       = LoopbackTransport::createNetwork(3, o);
    WorldSetup hs = hook;
    hs.transport  = t3[0].get();
    World host(hs);
    std::vector<std::unique_ptr<World>> clients;
    for (uint8_t i = 1; i <= 2; ++i)
    {
        WorldSetup cs = hook;
        cs.transport  = t3[i].get();
        cs.role       = NetRole::Client;
        cs.localPeer  = i;
        clients.push_back(std::make_unique<World>(cs));
        for (int k = 0; k < 3; ++k)
        {
            host.step();
            for (auto& c : clients) c->step();
        }
    }
    EXPECT_EQ(host.replication().stats().generationStatesSent, 0u);
    const std::string lines   = logSince(log);
    const std::string warning = "generator 'float_hook' is not integer-only";
    const size_t first        = lines.find(warning);
    EXPECT_NE(first, std::string::npos) << lines;
    EXPECT_EQ(lines.find(warning, first + 1), std::string::npos) << "logged once";
}

// ---- chunk records with tombstones (S23b step 2) -------------------------------------------------------------------

namespace
{
InstanceKey key(uint32_t slot, uint32_t counter)
{
    InstanceKey k;
    k.generatorSlot = slot;
    k.counter       = counter;
    return k;
}

// The first occurrence of a byte pattern in a blob (npos: none)
size_t findBytes(const std::vector<uint8_t>& blob, const std::vector<uint8_t>& pattern)
{
    auto it = std::search(blob.begin(), blob.end(), pattern.begin(), pattern.end());
    return it == blob.end() ? std::string::npos : size_t(it - blob.begin());
}

std::vector<uint8_t> keyBytes(uint8_t slot, uint32_t counter)
{
    return {slot, uint8_t(counter), uint8_t(counter >> 8), uint8_t(counter >> 16), uint8_t(counter >> 24)};
}
}  // namespace

// CREC format 2 round trip: placement versions and tombstones; world.hash changes with a tombstone and is equal after
// save / load; diffSaves names the tombstones; the effective record of a chunk without one
TEST(ChunkRecords, Format2RoundTripHashAndDiff)
{
    World a(ContentTest::contentTestSetup(2));
    EXPECT_EQ(a.effectivePlacementVersion(chunkAt(9, 9)), 2);
    EXPECT_TRUE(a.effectiveRecord(chunkAt(9, 9)).tombstones.empty());
    const uint64_t h0 = a.hash();
    std::vector<ChunkId> notified;
    a.addRecordListener([&](const ChunkId& c) { notified.push_back(c); });
    a.contentRecordsForUpdate().injectForTests(chunkAt(4, 5), 2, {key(0, 17), key(1, 3)});
    a.contentRecordsForUpdate().injectForTests(chunkAt(7, 2), 2, {});
    a.onRecordChanged(chunkAt(4, 5));
    EXPECT_EQ(notified.size(), 1u);
    EXPECT_NE(a.hash(), h0);
    const uint64_t h1 = a.hash();
    a.contentRecordsForUpdate().injectForTests(chunkAt(4, 5), 2, {key(0, 18)});
    EXPECT_NE(a.hash(), h1);
    EXPECT_EQ(a.effectiveRecord(chunkAt(7, 2)).placementVersion, 2);
    EXPECT_TRUE(a.effectiveRecord(chunkAt(4, 5)).tombstoned(key(0, 18)));

    const auto blob = a.save();
    ParsedSave parsed;
    std::string error;
    ASSERT_TRUE(parseSave(blob, parsed, error)) << error;
    ASSERT_TRUE(parsed.chunkRecords);
    World b(ContentTest::contentTestSetup(2));
    std::vector<ChunkId> loadNotified;
    b.addRecordListener([&](const ChunkId& c) { loadNotified.push_back(c); });
    const LoadResult r = b.tryLoad(blob);
    ASSERT_TRUE(r.ok()) << r.error;
    EXPECT_EQ(b.contentRecords().records().size(), 2u);
    EXPECT_EQ(b.effectiveRecord(chunkAt(4, 5)), a.effectiveRecord(chunkAt(4, 5)));
    EXPECT_EQ(b.effectiveRecord(chunkAt(7, 2)), a.effectiveRecord(chunkAt(7, 2)));
    EXPECT_EQ(b.hash(), a.hash());
    EXPECT_EQ(World::diff(a, b), "");
    // The load notified both recorded chunks
    EXPECT_EQ(loadNotified, (std::vector<ChunkId>{chunkAt(4, 5), chunkAt(7, 2)}));
    // A tombstone difference is named
    World c(ContentTest::contentTestSetup(2));
    c.contentRecordsForUpdate().injectForTests(chunkAt(4, 5), 2, {key(0, 17)});
    const std::string d = World::diff(a, c);
    EXPECT_NE(d.find("record space 0 chunk (4, 5)"), std::string::npos) << d;
    EXPECT_NE(d.find("tombstones"), std::string::npos) << d;
}

// Format 1 (S23a's committed save) reads as records without tombstones
TEST(ChunkRecords, Format1LoadsWithoutTombstones)
{
    const std::string path = std::string(SAIGA_PROJECT_SOURCE_DIR) + "/data/tests/engine/save_s23a.bin";
    std::ifstream f(path, std::ios::binary);
    const std::vector<uint8_t> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    ASSERT_FALSE(blob.empty()) << path;
    World w(ContentTest::contentTestSetup(2));
    ASSERT_TRUE(w.tryLoad(blob).ok());
    ASSERT_FALSE(w.contentRecords().records().empty());
    for (const auto& kv : w.contentRecords().records()) EXPECT_TRUE(kv.second.tombstones.empty());
    // It saves as format 2
    ParsedSave parsed;
    std::string error;
    ASSERT_TRUE(parseSave(w.save(), parsed, error)) << error;
    EXPECT_TRUE(parsed.chunkRecords);
}

// Load errors (the world unchanged): unsorted keys, a slot out of range, a counter out of range, tombstones in a space
// whose generator has no slots
TEST(ChunkRecords, LoadErrors)
{
    World a(ContentTest::contentTestSetup(2));
    a.contentRecordsForUpdate().injectForTests(chunkAt(4, 5), 2, {key(0, 12345), key(0, 12346)});
    const auto blob                   = a.save();
    std::vector<uint8_t> pair         = keyBytes(0, 12345);
    const std::vector<uint8_t> second = keyBytes(0, 12346);
    pair.insert(pair.end(), second.begin(), second.end());
    const size_t at = findBytes(blob, pair);
    ASSERT_NE(at, std::string::npos);
    auto expectError = [&](std::vector<uint8_t> bad, const std::string& what)
    {
        World w(ContentTest::contentTestSetup(2));
        const LoadResult r = loadInto(w, bad);
        EXPECT_FALSE(r.committed) << what;
        EXPECT_NE(r.error.find(what), std::string::npos) << r.error;
    };
    {
        std::vector<uint8_t> bad = blob;
        std::copy(second.begin(), second.end(), bad.begin() + std::ptrdiff_t(at));
        const std::vector<uint8_t> first = keyBytes(0, 12345);
        std::copy(first.begin(), first.end(), bad.begin() + std::ptrdiff_t(at + 5));
        expectError(bad, "not sorted and unique");
    }
    {
        std::vector<uint8_t> bad = blob;
        bad[at + 5]              = 7;
        expectError(bad, "a tombstone of slot 7");
    }
    {
        std::vector<uint8_t> bad     = blob;
        const std::vector<uint8_t> k = keyBytes(0, 20000);
        std::copy(k.begin(), k.end(), bad.begin() + std::ptrdiff_t(at + 5));
        expectError(bad, "out of range");
    }
    {
        // The dungeon's AuthoredSpaceGenerator has no slots
        World d(ContentTest::contentTestSetup(2));
        d.contentRecordsForUpdate().injectForTests(chunkAt(0, 0, 2), 1, {key(0, 1)});
        expectError(d.save(), "tombstones in a space whose generator has no slots");
    }
}

// ---- tombstones (S23b step 3) --------------------------------------------------------------------------------------

namespace
{
// The accepted (in the placement), rejected (no candidate) and excluded (a candidate inside an exclusion) keys of a
// slot in a chunk at the effective placement version
struct SlotKeys
{
    std::vector<InstanceKey> accepted, rejected, excluded;
};
SlotKeys slotKeys(const World& w, const ChunkId& c, uint32_t slot)
{
    SlotKeys out;
    const SpaceGeneration& g = w.generation().space(c.space);
    const uint16_t v         = w.effectivePlacementVersion(c);
    const auto p             = w.generation().chunkPlacement(c, v);
    for (uint32_t n = 0; n < g.generator->candidatesPerChunk(slot); ++n)
    {
        Candidate cand;
        const InstanceKey k = key(slot, n);
        if (!g.generator->candidate(v, *g.plan, c, slot, n, cand))
            out.rejected.push_back(k);
        else if (findInstance(*p, k))
            out.accepted.push_back(k);
        else
            out.excluded.push_back(k);
    }
    return out;
}

// A system that tombstones g_stepKey in phase 4 (the death and flush tests)
ChunkId g_stepChunk;
InstanceKey g_stepKey;
bool g_stepTombstone = false;
void stepTombstoneSystem(World& w, double)
{
    if (!g_stepTombstone) return;
    g_stepTombstone = false;
    w.tombstone(g_stepChunk, g_stepKey);
    // Applied at the flush, not yet
    EXPECT_FALSE(w.isTombstoned(g_stepChunk, g_stepKey));
}
void stepTombstoneSystems(SystemList& l)
{
    l.add(Phase::Gameplay, "test_tombstone", stepTombstoneSystem);
}
WorldSetup stepTombstoneSetup()
{
    WorldSetup s = ContentTest::contentTestSetup(2);
    s.systems    = stepTombstoneSystems;
    return s;
}
}  // namespace

// A valid key: the record is created at the current version, S22's sequence advances (an entry without deltas, saved
// by TERR); idempotent; inside a step it applies at the phase flush
TEST(Tombstone, ApplyIdempotentAndInStep)
{
    World w(stepTombstoneSetup());
    const ChunkId c  = chunkAt(4, 5);
    const SlotKeys k = slotKeys(w, c, 0);
    ASSERT_GE(k.accepted.size(), 2u);
    ASSERT_FALSE(k.rejected.empty());
    ASSERT_FALSE(k.excluded.empty()) << "site 0's exclusion lies in chunk (4, 5)";
    int notified = 0;
    w.addRecordListener([&](const ChunkId&) { notified++; });
    std::string error;
    EXPECT_TRUE(w.checkTombstone(c, k.accepted[0], &error)) << error;
    w.tombstone(c, k.accepted[0]);
    ASSERT_TRUE(w.chunkRecord(c));
    EXPECT_EQ(w.chunkRecord(c)->placementVersion, 2);
    EXPECT_TRUE(w.isTombstoned(c, k.accepted[0]));
    EXPECT_FALSE(w.isTombstoned(c, k.accepted[1]));
    EXPECT_EQ(w.terrainEdits().sequence(c), 1u);
    EXPECT_FALSE(w.terrainEdits().hasDeltas(c));
    EXPECT_EQ(notified, 1);
    // Idempotent
    w.tombstone(c, k.accepted[0]);
    EXPECT_EQ(w.terrainEdits().sequence(c), 1u);
    EXPECT_EQ(notified, 1);
    // Inside a step: a command applied at the flush of phase 4
    g_stepChunk     = c;
    g_stepKey       = k.accepted[1];
    g_stepTombstone = true;
    w.step(1);
    EXPECT_TRUE(w.isTombstoned(c, k.accepted[1]));
    EXPECT_EQ(w.terrainEdits().sequence(c), 2u);
    EXPECT_EQ(w.chunkRecord(c)->tombstones, (std::vector<InstanceKey>{k.accepted[0], k.accepted[1]}));
    // TERR saves the tombstone-only entry; the load restores record and sequence
    World b(stepTombstoneSetup());
    ASSERT_TRUE(b.tryLoad(w.save()).ok());
    EXPECT_EQ(b.terrainEdits().sequence(c), 2u);
    EXPECT_EQ(b.effectiveRecord(c), w.effectiveRecord(c));
    EXPECT_EQ(b.hash(), w.hash());
}

// Invalid keys: the check reports them (the world API) and the call stops on the host, also inside a step; a client
// call stops
TEST(TombstoneDeathTest, InvalidKeysStop)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    World w(stepTombstoneSetup());
    const ChunkId c  = chunkAt(4, 5);
    const SlotKeys k = slotKeys(w, c, 0);
    std::string error;
    EXPECT_FALSE(w.checkTombstone(c, k.rejected[0], &error));
    EXPECT_NE(error.find("rejected or excluded"), std::string::npos) << error;
    EXPECT_FALSE(w.checkTombstone(c, k.excluded[0], &error));
    EXPECT_NE(error.find("rejected or excluded"), std::string::npos) << error;
    EXPECT_FALSE(w.checkTombstone(c, key(0, 16384), &error));
    EXPECT_NE(error.find("out of range"), std::string::npos) << error;
    EXPECT_FALSE(w.checkTombstone(c, key(2, 0), &error));
    EXPECT_NE(error.find("has no slot 2"), std::string::npos) << error;
    EXPECT_FALSE(w.checkTombstone(chunkAt(128, 0), k.accepted[0], &error));
    EXPECT_NE(error.find("outside the space"), std::string::npos) << error;
    EXPECT_FALSE(w.checkTombstone(chunkAt(0, 0, 2), key(0, 0), &error));
    EXPECT_NE(error.find("has no slot 0"), std::string::npos) << error;

    EXPECT_DEATH(w.tombstone(c, k.rejected[0]), "rejected or excluded");
    EXPECT_DEATH(w.tombstone(c, k.excluded[0]), "rejected or excluded");
    EXPECT_DEATH(w.tombstone(c, key(0, 16384)), "out of range");
    EXPECT_DEATH(w.tombstone(chunkAt(-1, 3), k.accepted[0]), "outside the space");
    auto inStep = [&](InstanceKey bad)
    {
        g_stepChunk     = c;
        g_stepKey       = bad;
        g_stepTombstone = true;
        w.step(1);
    };
    EXPECT_DEATH(inStep(k.excluded[0]), "rejected or excluded");
    EXPECT_DEATH(inStep(key(0, 99999)), "out of range");
    EXPECT_FALSE(w.isTombstoned(c, k.excluded[0]));

    // A client call stops (R-NET-2)
    auto t       = LoopbackTransport::createNetwork(2, LoopbackOptions());
    WorldSetup h = ContentTest::contentTestSetup(2);
    h.transport  = t[0].get();
    World host(h);
    WorldSetup cs = ContentTest::contentTestSetup(2);
    cs.transport  = t[1].get();
    cs.role       = NetRole::Client;
    cs.localPeer  = 1;
    World client(cs);
    EXPECT_FALSE(client.checkTombstone(c, k.accepted[0], &error));
    EXPECT_DEATH(client.tombstone(c, k.accepted[0]), "tombstone on a client");
}

// A terrain edit creates no chunk record (S22's sequence only), on a FlatGenerator-like hook space and on a test
// generator space; a tombstone after an edit continues S22's sequence
TEST(ChunkRecords, EditsCreateNoRecordAndSequencesContinue)
{
    World w(ContentTest::contentTestSetup(2));
    const ChunkId c = chunkAt(4, 5);
    TerrainBrush b;
    b.center   = Vec3(1150, 0, 1400);
    b.radius   = 6;
    b.strength = 1;
    w.editTerrain(0, b);
    ASSERT_EQ(w.terrainEdits().sequence(c), 1u);
    EXPECT_TRUE(w.contentRecords().records().empty());
    const SlotKeys k = slotKeys(w, c, 1);
    ASSERT_FALSE(k.accepted.empty());
    w.tombstone(c, k.accepted[0]);
    EXPECT_EQ(w.terrainEdits().sequence(c), 2u);

    // A world whose only generator is a hook (no slots): an edit, no record, no generation state
    WorldSetup hook;
    SpaceTemplate t;
    t.generator                = 1;
    t.sizeChunks               = 16;
    hook.config.spaceTemplates = {t};
    hook.generators            = [](GeneratorList& l)
    { l.add(1, std::make_shared<HookGenerator>("hills", [](uint64_t, int64_t x, int64_t) { return int32_t(x); })); };
    World hw(hook);
    hw.editTerrain(0, b);
    EXPECT_TRUE(hw.contentRecords().empty());
    EXPECT_FALSE(hw.writesGenerationState());
}

// The committed save of S23b: records (stamped, tombstoned, tombstone-only), an edited chunk; it loads and saves again
// with the same state
TEST(ChunkRecords, CommittedSaveLoads)
{
    const std::string path = std::string(SAIGA_PROJECT_SOURCE_DIR) + "/data/tests/engine/save_s23b.bin";
    const ChunkId c45 = chunkAt(4, 5), c20 = chunkAt(20, 30);
    const char* update = std::getenv("SAIGA_UPDATE_GOLDEN");
    if (update && std::string(update) == "1")
    {
        World w(ContentTest::contentTestSetup(2));
        stampSiteZero(w);
        TerrainBrush b;
        b.center   = Vec3(1110, 0, 1290);
        b.radius   = 8;
        b.strength = 2;
        w.editTerrain(0, b);
        w.tombstone(c45, slotKeys(w, c45, 0).accepted[0]);
        w.tombstone(c20, slotKeys(w, c20, 1).accepted[0]);
        w.step(3);
        const auto blob = w.save();
        std::ofstream(path, std::ios::binary)
            .write(reinterpret_cast<const char*>(blob.data()), std::streamsize(blob.size()));
        GTEST_SKIP() << "wrote " << path;
    }
    std::ifstream f(path, std::ios::binary);
    const std::vector<uint8_t> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    ASSERT_FALSE(blob.empty()) << path;
    World w(ContentTest::contentTestSetup(2));
    const LoadResult r = w.tryLoad(blob);
    ASSERT_TRUE(r.ok()) << r.error;
    EXPECT_TRUE(w.contentRecords().isStamped(c45));
    ASSERT_TRUE(w.chunkRecord(c45));
    EXPECT_EQ(w.chunkRecord(c45)->tombstones.size(), 1u);
    ASSERT_TRUE(w.chunkRecord(c20));
    EXPECT_EQ(w.chunkRecord(c20)->tombstones.size(), 1u);
    EXPECT_FALSE(w.contentRecords().isStamped(c20));
    EXPECT_EQ(w.terrainEdits().sequence(c20), 1u);
    EXPECT_FALSE(w.terrainEdits().hasDeltas(c20));
    EXPECT_TRUE(w.terrainEdits().hasDeltas(c45));
    EXPECT_EQ(w.terrainEdits().sequence(c45), 2u);
    World again(ContentTest::contentTestSetup(2));
    ASSERT_TRUE(again.tryLoad(w.save()).ok());
    EXPECT_EQ(again.hash(), w.hash());
}

// ---- tombstone and record messages (S23b step 4) -------------------------------------------------------------------

namespace
{
std::vector<uint8_t> tombstoneMessage(uint32_t epoch, SpaceId space, const std::vector<TombstoneEntry>& entries)
{
    std::vector<uint8_t> out;
    encodeTombstoneModification(epoch, space, entries, out);
    return out;
}
std::vector<uint8_t> recordMessage(uint32_t epoch, SpaceId space, const std::vector<ChunkRecordPart>& parts)
{
    std::vector<uint8_t> out;
    encodeChunkRecordState(epoch, space, parts, out);
    return out;
}
TombstoneEntry entry(int32_t x, int32_t z, uint16_t version, uint32_t sequence, InstanceKey k)
{
    TombstoneEntry e;
    e.x                = x;
    e.z                = z;
    e.placementVersion = version;
    e.sequence         = sequence;
    e.key              = k;
    return e;
}
ChunkRecordPart part(int32_t x, int32_t z, uint16_t version, std::vector<InstanceKey> keys)
{
    ChunkRecordPart p;
    p.x                = x;
    p.z                = z;
    p.placementVersion = version;
    p.keys             = std::move(keys);
    return p;
}
void deliverTombstones(World& c, const std::vector<uint8_t>& m)
{
    c.replication().onTombstoneModification(0, m.data(), m.size());
}
void deliverRecords(World& c, const std::vector<uint8_t>& m)
{
    c.replication().onChunkRecordState(0, m.data(), m.size());
}
}  // namespace

// Sequences: an old one is dropped, a gap logged and counted (and applied), another epoch dropped; two entries of one
// chunk in one message apply with consecutive sequences, a skip in the second is a gap only for that entry
TEST(TombstoneNet, SequencesGapsAndEpochs)
{
    const WorldSetup s = ContentTest::contentTestSetup(2);
    GenNet n(s, s);
    n.step();
    World& c          = *n.client;
    const ChunkId k45 = chunkAt(4, 5);
    // Two entries of one chunk: sequences 1 and 2
    deliverTombstones(c, tombstoneMessage(0, 0, {entry(4, 5, 2, 1, key(0, 10)), entry(4, 5, 2, 2, key(0, 11))}));
    EXPECT_EQ(c.terrainEdits().sequence(k45), 2u);
    EXPECT_TRUE(c.isTombstoned(k45, key(0, 10)));
    EXPECT_TRUE(c.isTombstoned(k45, key(0, 11)));
    EXPECT_EQ(c.netStats().modificationGaps, 0u);
    // An old sequence is dropped
    deliverTombstones(c, tombstoneMessage(0, 0, {entry(4, 5, 2, 2, key(0, 12))}));
    EXPECT_FALSE(c.isTombstoned(k45, key(0, 12)));
    EXPECT_EQ(c.replication().stats().tombstonesDuplicate, 1u);
    // The second entry skips: a gap only for it (3 applies, 5 is a gap)
    deliverTombstones(c, tombstoneMessage(0, 0, {entry(4, 5, 2, 3, key(0, 12)), entry(4, 5, 2, 5, key(0, 13))}));
    EXPECT_EQ(c.netStats().modificationGaps, 1u);
    EXPECT_EQ(c.terrainEdits().sequence(k45), 5u);
    EXPECT_TRUE(c.isTombstoned(k45, key(0, 13)));
    // Another epoch: dropped, not counted as rejected
    const uint64_t rejected = c.netStats().rejected;
    deliverTombstones(c, tombstoneMessage(7, 0, {entry(4, 5, 2, 6, key(0, 14))}));
    EXPECT_FALSE(c.isTombstoned(k45, key(0, 14)));
    EXPECT_EQ(c.netStats().rejected, rejected);
    EXPECT_EQ(c.replication().stats().recordsDroppedEpoch, 1u);
}

// Rejected, counted, nothing applied
TEST(TombstoneNet, RejectionsApplyNothing)
{
    const WorldSetup s = ContentTest::contentTestSetup(2);
    GenNet n(s, s);
    n.step();
    World& c            = *n.client;
    auto expectRejected = [&](const std::vector<uint8_t>& m, bool records, const std::string& what)
    {
        const uint64_t before = c.netStats().rejected;
        const uint64_t hash   = c.hash();
        if (records)
            deliverRecords(c, m);
        else
            deliverTombstones(c, m);
        EXPECT_EQ(c.netStats().rejected, before + 1) << what;
        EXPECT_EQ(c.hash(), hash) << what;
        EXPECT_TRUE(c.contentRecords().records().empty()) << what;
    };
    expectRejected(tombstoneMessage(0, 0, {entry(128, 5, 2, 1, key(0, 1))}), false, "a chunk outside the space");
    expectRejected(tombstoneMessage(0, 0, {entry(4, 5, 3, 1, key(0, 1))}), false, "a version above the current one");
    expectRejected(tombstoneMessage(0, 0, {entry(4, 5, 2, 1, key(2, 1))}), false, "a slot beyond the table");
    expectRejected(tombstoneMessage(0, 0, {entry(4, 5, 2, 1, key(1, 1024))}), false, "a counter out of range");
    expectRejected(tombstoneMessage(0, 2, {entry(0, 0, 1, 1, key(0, 1))}), false, "keys in a space without slots");
    expectRejected(tombstoneMessage(0, 0, {entry(4, 5, 2, 1, key(0, 1)), entry(4, 6, 2, 1, key(9, 1))}), false,
                   "one invalid entry among valid ones");
    expectRejected(recordMessage(0, 0, {part(4, 5, 2, {key(0, 5), key(0, 4)})}), true, "unsorted keys");
    expectRejected(recordMessage(0, 0, {part(4, 5, 3, {key(0, 5)})}), true, "a version above the current one");
    expectRejected(recordMessage(0, 0, {part(4, 5, 1, {key(0, 5)})}), true, "a version below the pin");
    expectRejected(recordMessage(0, 2, {part(0, 0, 1, {key(0, 5)})}), true, "keys in a space without slots");
    expectRejected(recordMessage(0, 0, {part(4, 5, 2, {key(0, 5)}), part(4, 5, 1, {key(0, 6)})}), true,
                   "parts of one chunk with different versions");
    // A part without keys needs no slots (the dungeon at its current version)
    deliverRecords(c, recordMessage(0, 2, {part(0, 0, 1, {})}));
    EXPECT_TRUE(c.chunkRecord(chunkAt(0, 0, 2)));
    // A version differing from the chunk's effective one (its record's)
    {
        const uint64_t before = c.netStats().rejected;
        deliverTombstones(c, tombstoneMessage(0, 0, {entry(4, 5, 1, 1, key(0, 1))}));
        deliverRecords(c, recordMessage(0, 2, {part(0, 0, 2, {})}));
        EXPECT_EQ(c.netStats().rejected, before + 2);
        EXPECT_EQ(c.chunkRecord(chunkAt(0, 0, 2))->placementVersion, 1);
    }
    // A record message after a refused GenerationState
    GenNet refused(ContentTest::contentTestSetup(2), ContentTest::contentTestSetup(1));
    refused.step();
    ASSERT_TRUE(refused.client->generationMismatch());
    const uint64_t before = refused.client->netStats().rejected;
    deliverTombstones(*refused.client, tombstoneMessage(0, 0, {entry(4, 5, 1, 1, key(0, 1))}));
    deliverRecords(*refused.client, recordMessage(0, 0, {part(4, 5, 1, {key(0, 1)})}));
    EXPECT_EQ(refused.client->netStats().rejected, before + 2);
    EXPECT_TRUE(refused.client->contentRecords().records().empty());
}

// The live path: host tombstones reach the client in application order with the terrain edits (sequences equal); a
// late joiner after a terrain edit on a flat space has no rejections and equal sequences (E10)
TEST(TombstoneNet, LiveAndFlatJoiner)
{
    const WorldSetup s = ContentTest::contentTestSetup(2);
    GenNet n(s, s);
    n.step();
    const ChunkId c  = chunkAt(4, 5);
    const SlotKeys k = slotKeys(*n.host, c, 1);
    TerrainBrush b;
    b.center   = Vec3(1150, 0, 1400);
    b.radius   = 6;
    b.strength = 1;
    n.host->editTerrain(0, b);
    n.host->tombstone(c, k.accepted[0]);
    n.host->editTerrain(0, b);
    n.host->tombstone(c, k.accepted[1]);
    n.step();
    EXPECT_EQ(n.client->terrainEdits().sequence(c), 4u);
    EXPECT_EQ(n.client->effectiveRecord(c), n.host->effectiveRecord(c));
    EXPECT_EQ(n.client->netStats().modificationGaps, 0u);
    EXPECT_EQ(n.client->netStats().rejected, 0u);
    // Idempotent: nothing is sent again
    const uint64_t sent = n.host->replication().stats().tombstoneMessagesSent;
    n.host->tombstone(c, k.accepted[1]);
    n.step();
    EXPECT_EQ(n.host->replication().stats().tombstoneMessagesSent, sent);

    // A flat world: an edit before the join, no rejections, equal sequences
    GenNet flat(WorldSetup{}, WorldSetup{}, [&](World& h) { h.editTerrain(0, b); });
    flat.step();
    EXPECT_EQ(flat.client->netStats().rejected, 0u);
    EXPECT_EQ(flat.client->terrainEdits().chunkCount(), flat.host->terrainEdits().chunkCount());
    EXPECT_GT(flat.client->terrainEdits().chunkCount(), 0u);
}

// A late joiner gets the records of chunks tombstoned before it joined, the ChunkRecordState before the
// TerrainChunkState (E5); the tombstones pending in the joining step arrive in the snapshot, not again live
TEST(TombstoneNet, LateJoinerSnapshotOrder)
{
    const WorldSetup s = ContentTest::contentTestSetup(2);
    const ChunkId c    = chunkAt(4, 5);
    InstanceKey pending;
    GenNet n(s, s,
             [&](World& h)
             {
                 TerrainBrush b;
                 b.center   = Vec3(1150, 0, 1400);
                 b.radius   = 6;
                 b.strength = 1;
                 // Everything pending in the host state list of the joining step (the loopback peer connects at
                 // once: the host's first step is the join)
                 h.editTerrain(0, b);
                 const SlotKeys k = slotKeys(h, c, 1);
                 h.tombstone(c, k.accepted[0]);
                 h.editTerrain(0, b);
                 pending = k.accepted[1];
                 h.tombstone(c, pending);
             });
    // The record arrives before the chunk's terrain state
    int recordBeforeTerrain = -1;
    n.client->addRecordListener(
        [&](const ChunkId& changed)
        {
            if (changed == c && recordBeforeTerrain < 0)
                recordBeforeTerrain = n.client->terrainEdits().sequence(c) == 0 ? 1 : 0;
        });
    n.step();
    EXPECT_EQ(recordBeforeTerrain, 1);
    EXPECT_TRUE(n.client->isTombstoned(c, pending));
    EXPECT_EQ(n.client->effectiveRecord(c), n.host->effectiveRecord(c));
    EXPECT_EQ(n.client->terrainEdits().sequence(c), n.host->terrainEdits().sequence(c));
    EXPECT_EQ(n.client->replication().stats().tombstonesApplied, 0u) << "the pending tombstone came in the snapshot";
    EXPECT_EQ(n.client->replication().stats().tombstonesDuplicate, 0u);
    EXPECT_EQ(n.client->netStats().modificationGaps, 0u);
    EXPECT_EQ(n.client->netStats().rejected, 0u);
}

// A host load (Resync): the client clears records and S22's store together; a tombstone made after the save is gone,
// one in the save comes back with the ChunkRecordState; equal sequences afterwards
TEST(TombstoneNet, HostLoadResyncs)
{
    const WorldSetup s = ContentTest::contentTestSetup(2);
    GenNet n(s, s);
    n.step();
    const ChunkId c  = chunkAt(4, 5);
    const SlotKeys k = slotKeys(*n.host, c, 0);
    n.host->tombstone(c, k.accepted[0]);
    n.step();
    const auto blob = n.host->save();
    n.host->tombstone(c, k.accepted[1]);
    n.step();
    ASSERT_TRUE(n.client->isTombstoned(c, k.accepted[1]));
    std::vector<ChunkId> changes;
    n.client->addRecordListener([&](const ChunkId& x) { changes.push_back(x); });
    ASSERT_TRUE(n.host->tryLoad(blob).ok());
    n.step();
    EXPECT_GE(changes.size(), 2u) << "cleared, then the snapshot";
    EXPECT_TRUE(n.client->isTombstoned(c, k.accepted[0]));
    EXPECT_FALSE(n.client->isTombstoned(c, k.accepted[1]));
    EXPECT_EQ(n.client->effectiveRecord(c), n.host->effectiveRecord(c));
    EXPECT_EQ(n.client->terrainEdits().sequence(c), n.host->terrainEdits().sequence(c));
    EXPECT_EQ(n.client->netStats().rejected, 0u);
}

// Bounds: 4096 tombstones in one step arrive in one message, 4097 in two; 3000 injected records in three
// ChunkRecordStates; one record of 10000 keys in two, united on the client
TEST(TombstoneNet, MessageBounds)
{
    const WorldSetup s = ContentTest::contentTestSetup(2);
    GenNet n(s, s);
    n.step();
    // Accepted test_tree keys of the chunks around (10, 10), in ChunkId then key order
    std::vector<std::pair<ChunkId, InstanceKey>> keys;
    for (int32_t x = 10; keys.size() < 4096 + 4097; ++x)
        for (int32_t z = 10; z < 14; ++z)
            for (const InstanceKey& k : slotKeys(*n.host, chunkAt(x, z), 0).accepted)
                keys.push_back({chunkAt(x, z), k});
    for (size_t i = 0; i < 4096; ++i) n.host->tombstone(keys[i].first, keys[i].second);
    uint64_t sent = n.host->replication().stats().tombstoneMessagesSent;
    n.step();
    EXPECT_EQ(n.host->replication().stats().tombstoneMessagesSent - sent, 1u);
    EXPECT_EQ(n.client->replication().stats().tombstonesApplied, 4096u);
    for (size_t i = 4096; i < 4096 + 4097; ++i) n.host->tombstone(keys[i].first, keys[i].second);
    sent = n.host->replication().stats().tombstoneMessagesSent;
    n.step();
    EXPECT_EQ(n.host->replication().stats().tombstoneMessagesSent - sent, 2u);
    EXPECT_EQ(n.client->replication().stats().tombstonesApplied, 4096u + 4097u);
    EXPECT_EQ(n.client->netStats().modificationGaps, 0u);

    GenNet many(s, s,
                [](World& h)
                {
                    for (int i = 0; i < 3000; ++i)
                        h.contentRecordsForUpdate().injectForTests(chunkAt(i % 128, i / 128), 2, {key(0, 1)});
                });
    many.step();
    EXPECT_EQ(many.host->replication().stats().recordStatesSent, 3u);
    EXPECT_EQ(many.client->contentRecords().records().size(), 3000u);
    EXPECT_EQ(many.client->netStats().rejected, 0u);

    GenNet big(s, s,
               [](World& h)
               {
                   std::vector<InstanceKey> k;
                   for (uint32_t i = 0; i < 10000; ++i) k.push_back(key(0, i));
                   h.contentRecordsForUpdate().injectForTests(chunkAt(4, 5), 2, k);
               });
    big.step();
    EXPECT_EQ(big.host->replication().stats().recordStatesSent, 2u);
    ASSERT_TRUE(big.client->chunkRecord(chunkAt(4, 5)));
    EXPECT_EQ(big.client->chunkRecord(chunkAt(4, 5))->tombstones.size(), 10000u);
}
