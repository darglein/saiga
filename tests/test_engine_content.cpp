/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The content generation (docs/engine_design.md §12.1, §15.3, R-CONTENT-1; S23a): the fixed-point toolkit and the
// integer lint (this unit is compiled with trapping UBSan on the Clang CI job, SAIGA_CONTENT_UBSAN).

#include "saiga/engine/content/gen/FixedMath.h"
#include "saiga/engine/content/gen/IntNoise.h"
#include "saiga/engine/content/gen/TrigTable.h"
#include "saiga/engine/core/Hash.h"

#include "gtest/gtest.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <set>

#include "content_lint.h"

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
