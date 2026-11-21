/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Terraforming (docs/engine_design.md §12.1, §12.2; S22) without a device: the integer brush kernel, validateBrush,
// the modification state, World::editTerrain, its events, saves and the visual path on the recording sink.

#include "saiga/core/math/TerrainSurface.h"
#include "saiga/engine/core/Hash.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/terrain/BrushKernel.h"
#include "saiga/engine/terrain/TerrainBrush.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "terrain_test_hooks.h"

using namespace Saiga;
using namespace Saiga::Engine;
using namespace TerrainTestHooks;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);

TerrainBrush brush(TerrainBrush::Op op, double x, double z, double radius, double strength, double y = 0)
{
    TerrainBrush b;
    b.op       = op;
    b.center   = Vec3(x, y, z);
    b.radius   = radius;
    b.strength = strength;
    return b;
}

// One kernel evaluation over a field function (space samples)
struct KernelRun
{
    QuantizedBrush q;
    SampleRect affected, read;
    std::vector<int32_t> pre, out;
    uint64_t changed = 0;
    int32_t before(int64_t i, int64_t k) const { return pre[size_t((k - read.k0) * read.width() + (i - read.i0))]; }
    int32_t after(int64_t i, int64_t k) const
    {
        return out[size_t((k - affected.k0) * affected.width() + (i - affected.i0))];
    }
};

KernelRun runKernel(const TerrainBrush& b, int spacing, int64_t maxSample,
                    const std::function<int32_t(int64_t, int64_t)>& field)
{
    KernelRun r;
    r.q        = BrushKernel::quantize(b);
    r.affected = BrushKernel::affectedRect(r.q, spacing, maxSample);
    r.read     = BrushKernel::readRect(r.q, r.affected, maxSample);
    for (int64_t k = r.read.k0; k <= r.read.k1; ++k)
        for (int64_t i = r.read.i0; i <= r.read.i1; ++i) r.pre.push_back(field(i, k));
    r.out.assign(size_t(r.affected.width() * r.affected.height()), 0);
    if (!r.affected.empty())
        r.changed = BrushKernel::evaluate(r.q, spacing, r.affected, r.read, r.pre.data(), r.out.data());
    return r;
}

int32_t noise(int64_t i, int64_t k)
{
    uint64_t h = uint64_t(i) * 0x9E3779B97F4A7C15ull ^ uint64_t(k) * 0xC2B2AE3D27D4EB4Full;
    h ^= h >> 29;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 32;
    return int32_t(h % 8192) - 4096 + int32_t(i * 37 - k * 11);
}

SpaceTemplate space16(int spacing = 1)
{
    return testSpaceTemplate(16, spacing, integerTestHook);
}
}  // namespace

// ---- kernel (step 1)

TEST(BrushKernel, Quantization)
{
    QuantizedBrush q = BrushKernel::quantize(brush(TerrainBrush::Add, -0.001, 3.999, 0.001, 0.25));
    EXPECT_EQ(q.cx, -1);
    EXPECT_EQ(q.cz, 1023);
    EXPECT_EQ(q.r, 1);  // max(1, floor(0.256))
    EXPECT_EQ(q.s, 256);
    q = BrushKernel::quantize(brush(TerrainBrush::Flatten, -10.5, 7.25, 12.3, 0.5, -3.0001));
    EXPECT_EQ(q.cx, -2688);
    EXPECT_EQ(q.cz, 1856);
    EXPECT_EQ(q.r, int64_t(std::floor(12.3 * 256)));
    EXPECT_EQ(q.cy, -3073);
    EXPECT_EQ(q.f, 32768);
    EXPECT_EQ(q.s, 0);
    q = BrushKernel::quantize(brush(TerrainBrush::Smooth, 0, 0, 128, 1));
    EXPECT_EQ(q.r, 32768);
    EXPECT_EQ(q.f, 65536);
}

TEST(BrushKernel, WeightValues)
{
    // Center on sample (100, 100) at 1 m, radius 8 m
    const QuantizedBrush q = BrushKernel::quantize(brush(TerrainBrush::Add, 100, 100, 8, 1));
    EXPECT_EQ(BrushKernel::weight(q, 1, 100, 100), 65536);
    // Half the radius: t2 = 16384, w = 36864
    EXPECT_EQ(BrushKernel::weight(q, 1, 104, 100), 36864);
    // At and beyond the radius: not affected
    EXPECT_EQ(BrushKernel::weight(q, 1, 108, 100), -1);
    EXPECT_EQ(BrushKernel::weight(q, 1, 106, 106), -1);
    // Radius 64 m, 63 m from the center: t2 = 63^2 / 64^2 * 65536 = 63504, w = (2032^2) >> 16 = 63
    const QuantizedBrush big = BrushKernel::quantize(brush(TerrainBrush::Add, 0, 0, 64, 1));
    EXPECT_EQ(BrushKernel::weight(big, 1, 63, 0), 63);
    // Near the rim u < 256 gives w = 0: d = 8 m, r = 8.004 m (2049 / 256 m): t2 = floor(2048^2 * 65536 / 2049^2) =
    // 65472, u = 64, w = 0
    const TerrainBrush rimBrush = brush(TerrainBrush::Add, 0, 0, 8.004, 64);
    const QuantizedBrush rim    = BrushKernel::quantize(rimBrush);
    EXPECT_EQ(rim.r, 2049);
    EXPECT_EQ(BrushKernel::weight(rim, 1, 8, 0), 0);
    const KernelRun run = runKernel(rimBrush, 1, 4096, [](int64_t, int64_t) { return 0; });
    EXPECT_TRUE(run.affected.contains(8, 0));
    EXPECT_EQ(run.after(8, 0), 0);  // affected but unchanged
}

TEST(BrushKernel, AddSubtractCenterAndSymmetry)
{
    for (TerrainBrush::Op op : {TerrainBrush::Add, TerrainBrush::Subtract})
    {
        const KernelRun r = runKernel(brush(op, 500, 600, 10, 1.5), 1, 4096, noise);
        const int64_t s   = 1536;
        EXPECT_EQ(int64_t(r.after(500, 600)) - r.before(500, 600), op == TerrainBrush::Add ? s : -s);
        // Mirrored samples around a center on a sample change by equal amounts
        for (int d = 1; d < 10; ++d)
        {
            const int32_t a = r.after(500 + d, 600) - r.before(500 + d, 600);
            EXPECT_EQ(a, r.after(500 - d, 600) - r.before(500 - d, 600));
            EXPECT_EQ(a, r.after(500, 600 + d) - r.before(500, 600 + d));
            EXPECT_EQ(a, r.after(500, 600 - d) - r.before(500, 600 - d));
            EXPECT_EQ(r.after(500 + d / 2, 600 + d / 3) - r.before(500 + d / 2, 600 + d / 3),
                      r.after(500 - d / 2, 600 - d / 3) - r.before(500 - d / 2, 600 - d / 3));
        }
        // Counts changed samples, not affected ones: a 10 m disc at 1 m has 305 samples strictly inside
        uint64_t inside = 0;
        for (int64_t k = r.affected.k0; k <= r.affected.k1; ++k)
            for (int64_t i = r.affected.i0; i <= r.affected.i1; ++i)
                inside += BrushKernel::weight(r.q, 1, i, k) >= 0 ? 1 : 0;
        EXPECT_EQ(inside, 305u);
        EXPECT_LE(r.changed, inside);
        EXPECT_GT(r.changed, 250u);
    }
}

TEST(BrushKernel, FlattenReachesTargetAndNeverOvershoots)
{
    const double target = 2.5;
    const KernelRun r   = runKernel(brush(TerrainBrush::Flatten, 300, 300, 16, 1, target), 1, 4096, noise);
    EXPECT_EQ(r.after(300, 300), 2560);
    for (int64_t k = r.affected.k0; k <= r.affected.k1; ++k)
        for (int64_t i = r.affected.i0; i <= r.affected.i1; ++i)
        {
            const int64_t h = r.before(i, k), n = r.after(i, k), cy = 2560;
            if (h <= cy)
            {
                EXPECT_GE(n, h);
                EXPECT_LE(n, cy);
            }
            else
            {
                EXPECT_LE(n, h);
                EXPECT_GE(n, cy);
            }
        }
    // Half strength: never crosses either
    const KernelRun h = runKernel(brush(TerrainBrush::Flatten, 300.3, 299.7, 16, 0.5, -1.0), 1, 4096, noise);
    for (int64_t k = h.affected.k0; k <= h.affected.k1; ++k)
        for (int64_t i = h.affected.i0; i <= h.affected.i1; ++i)
        {
            const int64_t b = h.before(i, k), n = h.after(i, k), cy = -1024;
            EXPECT_TRUE((b <= cy && n >= b && n <= cy) || (b >= cy && n <= b && n >= cy));
        }
}

TEST(BrushKernel, SmoothKeepsConstantAndRamp)
{
    const KernelRun c =
        runKernel(brush(TerrainBrush::Smooth, 200, 200, 12, 1), 2, 2048, [](int64_t, int64_t) { return 777; });
    EXPECT_EQ(c.changed, 0u);
    // A linear ramp: the interior (every 3 x 3 neighbourhood inside the space) is unchanged
    const KernelRun r = runKernel(brush(TerrainBrush::Smooth, 200, 200, 12, 1), 2, 2048,
                                  [](int64_t i, int64_t k) { return int32_t(i * 13 - k * 7 + 5); });
    EXPECT_EQ(r.changed, 0u);
    // At the space border the clamped indices are not a ramp any more: the border samples change
    const KernelRun b = runKernel(brush(TerrainBrush::Smooth, 0, 100, 12, 1), 1, 4096,
                                  [](int64_t i, int64_t k) { return int32_t(i * 1300 - k * 7 + 5); });
    EXPECT_GT(b.changed, 0u);
    for (int64_t k = b.affected.k0; k <= b.affected.k1; ++k)
        for (int64_t i = b.affected.i0 + 1; i <= b.affected.i1; ++i) EXPECT_EQ(b.after(i, k), b.before(i, k));
}

TEST(BrushKernel, IterationOrderDoesNotMatter)
{
    for (TerrainBrush::Op op : {TerrainBrush::Add, TerrainBrush::Subtract, TerrainBrush::Flatten, TerrainBrush::Smooth})
    {
        const KernelRun r =
            runKernel(brush(op, 1000.37, 999.61, 9.5, op <= TerrainBrush::Subtract ? 2 : 0.8, 1.5), 1, 4096, noise);
        // Reversed iteration over the same pre-edit copy
        for (int64_t k = r.affected.k1; k >= r.affected.k0; --k)
            for (int64_t i = r.affected.i1; i >= r.affected.i0; --i)
                EXPECT_EQ(BrushKernel::evaluateSample(r.q, 1, r.read, r.pre.data(), i, k), r.after(i, k));
    }
}

TEST(BrushKernel, ClippedAtTheSpaceBorder)
{
    // Spacing 2, 16 chunks: samples 0..2048
    const KernelRun a = runKernel(brush(TerrainBrush::Add, 0, 0, 10, 1), 2, 2048, noise);
    EXPECT_EQ(a.affected.i0, 0);
    EXPECT_EQ(a.affected.k0, 0);
    EXPECT_EQ(a.affected.i1, 5);
    EXPECT_EQ(a.affected.k1, 5);
    EXPECT_EQ(a.after(0, 0) - a.before(0, 0), 1024);
    const KernelRun b = runKernel(brush(TerrainBrush::Smooth, 4096, 4096, 10, 1), 2, 2048, noise);
    EXPECT_EQ(b.affected.i1, 2048);
    EXPECT_EQ(b.affected.k1, 2048);
    EXPECT_EQ(b.affected.i0, 2043);
    EXPECT_EQ(b.read.i0, 2042);
    EXPECT_EQ(b.read.i1, 2048);
    // Entirely outside: empty
    const KernelRun c = runKernel(brush(TerrainBrush::Add, -20, 50, 10, 1), 2, 2048, noise);
    EXPECT_TRUE(c.affected.empty());
    EXPECT_EQ(c.changed, 0u);
}

TEST(BrushKernel, SmoothRoundsOnceToTheNearest)
{
    // Against the exact rational step (sum / 9 - h) * g / 65536: error <= 0.5 units per sample, no sign bias
    std::mt19937_64 rng(1234);
    std::uniform_int_distribution<int32_t> dist(-200000, 200000);
    std::vector<int32_t> field(size_t(512) * 512);
    for (int32_t& v : field) v = dist(rng);
    auto f           = [&](int64_t i, int64_t k) { return field[size_t(k * 512 + i)]; };
    int64_t errorSum = 0;
    uint64_t count   = 0;
    for (double radius : {4.0, 16.0})
        for (double strength : {0.1, 0.5, 1.0})
        {
            uint64_t combo = 0;
            for (int c = 0; combo < 2000; ++c)
            {
                const double x    = 40 + (c * 37) % 420 + 0.31 * (c % 7);
                const double z    = 40 + (c * 53) % 420 + 0.17 * (c % 5);
                const KernelRun r = runKernel(brush(TerrainBrush::Smooth, x, z, radius, strength), 1, 511, f);
                for (int64_t k = r.affected.k0; k <= r.affected.k1; ++k)
                    for (int64_t i = r.affected.i0; i <= r.affected.i1; ++i)
                    {
                        const int64_t w = BrushKernel::weight(r.q, 1, i, k);
                        if (w < 0) continue;
                        const int64_t g = (r.q.f * w) >> 16;
                        int64_t sum     = 0;
                        for (int dk = -1; dk <= 1; ++dk)
                            for (int di = -1; di <= 1; ++di) sum += r.before(i + di, k + dk);
                        const int64_t h = r.before(i, k), n = r.after(i, k);
                        const int64_t lhs = 2 * 9 * 65536 * (n - h) - 2 * (sum - 9 * h) * g;
                        EXPECT_LE(std::abs(lhs), 9 * 65536);
                        errorSum += 9 * 65536 * (n - h) - (sum - 9 * h) * g;
                        combo++;
                    }
            }
            count += combo;
        }
    EXPECT_GE(count, 10000u);
    const double mean = double(errorSum) / (9.0 * 65536.0 * double(count));
    EXPECT_LE(std::abs(mean), 0.02) << mean;
}

TEST(BrushKernel, ClampDecidedAfterTheClamp)
{
    const int32_t L = kTerrainHeightLimit;
    // Add at +2^23 and Subtract at -2^23 change nothing
    const KernelRun a =
        runKernel(brush(TerrainBrush::Add, 100, 100, 6, 2), 1, 4096, [L](int64_t, int64_t) { return L; });
    EXPECT_EQ(a.changed, 0u);
    const KernelRun s =
        runKernel(brush(TerrainBrush::Subtract, 100, 100, 6, 2), 1, 4096, [L](int64_t, int64_t) { return -L; });
    EXPECT_EQ(s.changed, 0u);
    // Add 1 m (s = 1024) at 2^23 - 100 gives exactly 2^23 at the center and counts as changed
    const KernelRun c =
        runKernel(brush(TerrainBrush::Add, 100, 100, 6, 1), 1, 4096, [L](int64_t, int64_t) { return L - 100; });
    EXPECT_EQ(c.after(100, 100), L);
    EXPECT_GT(c.changed, 0u);
    // A sample beyond the limit at zero weight stays unchanged (pure kernel: the engine clamps every base)
    const KernelRun z =
        runKernel(brush(TerrainBrush::Add, 0, 0, 8.004, 64), 1, 4096, [L](int64_t, int64_t) { return L + 5; });
    EXPECT_EQ(z.after(8, 0), L + 5);
    EXPECT_EQ(z.after(0, 0), L);  // changed: clamped from L + 5 + 65536
}

TEST(BrushKernel, ValidateBrush)
{
    const SpaceTemplate sp = space16();  // 4096 m
    std::string error;
    const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
    auto check = [&](TerrainBrush b, const char* field)
    {
        error.clear();
        EXPECT_FALSE(validateBrush(b, sp, &error)) << field;
        EXPECT_NE(error.find(field), std::string::npos) << error;
    };
    for (double bad : {nan, inf, -inf})
    {
        for (int f = 0; f < 3; ++f)
        {
            TerrainBrush b = brush(TerrainBrush::Add, 10, 10, 5, 1);
            b.center(f)    = bad;
            check(b, f == 0 ? "center.x" : (f == 1 ? "center.y" : "center.z"));
        }
        check(brush(TerrainBrush::Add, 10, 10, bad, 1), "radius");
        check(brush(TerrainBrush::Smooth, 10, 10, 5, bad), "strength");
    }
    check(brush(TerrainBrush::Add, 10, 10, 0, 1), "radius");
    check(brush(TerrainBrush::Add, 10, 10, -1, 1), "radius");
    check(brush(TerrainBrush::Add, 10, 10, 128.0001, 1), "radius");
    check(brush(TerrainBrush::Add, 10, 10, 5, 64.001), "strength");
    check(brush(TerrainBrush::Subtract, 10, 10, 5, -0.1), "strength");
    check(brush(TerrainBrush::Flatten, 10, 10, 5, 1.01), "strength");
    check(brush(TerrainBrush::Smooth, 10, 10, 5, 2), "strength");
    TerrainBrush op = brush(TerrainBrush::Add, 10, 10, 5, 1);
    op.op           = TerrainBrush::Count;
    check(op, "op");
    op.op = TerrainBrush::Op(200);
    check(op, "op");
    check(brush(TerrainBrush::Add, -128.001, 10, 5, 1), "center.x");
    check(brush(TerrainBrush::Add, 10, 4096 + 128.001, 5, 1), "center.z");
    check(brush(TerrainBrush::Add, 1e300, 10, 5, 1), "center.x");
    check(brush(TerrainBrush::Add, 10, -1e300, 5, 1), "center.z");
    check(brush(TerrainBrush::Flatten, 10, 10, 5, 1, 8192.001), "center.y");
    check(brush(TerrainBrush::Flatten, 10, 10, 5, 1, -1e300), "center.y");
    // The bounds themselves are accepted; y beyond 8192 m only matters for Flatten
    EXPECT_TRUE(validateBrush(brush(TerrainBrush::Add, -128, 4096 + 128, 128, 64, 1e6), sp, &error)) << error;
    EXPECT_TRUE(validateBrush(brush(TerrainBrush::Subtract, 4224, -128, 0.001, 0), sp, &error)) << error;
    EXPECT_TRUE(validateBrush(brush(TerrainBrush::Flatten, 0, 0, 1, 1, -8192), sp, &error)) << error;
    EXPECT_TRUE(validateBrush(brush(TerrainBrush::Smooth, 0, 0, 1, 1, 8192), sp, &error)) << error;
    EXPECT_TRUE(validateBrush(brush(TerrainBrush::Flatten, 0, 0, 1, 0, 8192), sp, &error)) << error;
}

// The committed result hash (R-DET-1): integer arithmetic only, equal on GCC, Clang and MSVC. A change of this value
// is a change of the kernel (§12.2)
TEST(BrushKernel, CommittedHash)
{
    Hasher h;
    const TerrainBrush brushes[] = {
        brush(TerrainBrush::Add, 300.3, 299.9, 17.7, 3.3),      brush(TerrainBrush::Subtract, 280, 310.5, 9, 63),
        brush(TerrainBrush::Flatten, 305, 305, 25, 0.77, -2.2), brush(TerrainBrush::Smooth, 290.1, 300.2, 12.5, 0.9),
        brush(TerrainBrush::Add, 0.4, 0.6, 5, 0.001),           brush(TerrainBrush::Smooth, 511, 511, 4, 1),
    };
    for (int spacing : {1, 2})
        for (const TerrainBrush& b : brushes)
        {
            const KernelRun r = runKernel(b, spacing, 511, noise);
            h.addU64(r.changed);
            for (int32_t v : r.out) h.addI32(v);
        }
    EXPECT_EQ(h.value(), 5726413041445852036ull);
}
