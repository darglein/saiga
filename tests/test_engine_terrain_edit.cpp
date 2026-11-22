/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Terraforming (docs/engine_design.md §12.1, §12.2; S22) without a device: the integer brush kernel, validateBrush,
// the modification state, World::editTerrain, its events, saves and the visual path on the recording sink.

#include "saiga/core/math/TerrainSurface.h"
#include "saiga/engine/api/WorldApi.h"
#include "saiga/engine/core/Hash.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/core/LogRing.h"
#include "saiga/engine/net/LoopbackTransport.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/net/Varint.h"
#include "saiga/engine/save/SaveDiff.h"
#include "saiga/engine/save/SaveFormat.h"
#include "saiga/engine/sync/RecordingRenderSink.h"
#include "saiga/engine/terrain/BrushKernel.h"
#include "saiga/engine/terrain/TerrainBrush.h"
#include "saiga/engine/terrain/TerrainDeltaCodec.h"
#include "saiga/engine/terrain/TerrainDirtyQueue.h"
#include "saiga/engine/terrain/TerrainEdit.h"
#include "saiga/engine/terrain/TerrainEditStore.h"
#include "saiga/engine/terrain/TerrainGeneration.h"
#include "saiga/engine/terrain/TerrainStore.h"
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/engine/terrain/TerrainVisualPatch.h"
#include "saiga/engine/tools/WorldTools.h"
#include "saiga/engine/world/Events.h"
#include "saiga/engine/world/World.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <zstd.h>

#include "mcp_test_helper.h"
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

ChunkId chunk(int32_t x, int32_t z, SpaceId space = 0)
{
    ChunkId c;
    c.space = space;
    c.x     = x;
    c.z     = z;
    return c;
}

WorldSetup editSetup(int spacing, uint64_t seed = 7)
{
    WorldSetup s;
    s.seed      = seed;
    s.profiling = false;
    s.config.spaceTemplates.push_back(terraformTestTemplate(spacing));
    return s;
}

// 2^23 everywhere (the clamp test)
int32_t maxHook(uint64_t, int64_t, int64_t)
{
    return kTerrainHeightLimit;
}

size_t pendingEvents(const World& w)
{
    return w.events.find(TerrainEditedEvent::id)->pendingCount();
}

// Every allocated block of the space holds a non-zero delta (blocks only for changed samples)
void expectBlocksNonZero(const World& w, SpaceId space)
{
    const TerrainEditStore& st = w.terrainEdits();
    size_t found               = 0;
    for (int64_t bk = 0; bk <= st.maxSample(space) / 16; ++bk)
        for (int64_t bi = 0; bi <= st.maxSample(space) / 16; ++bi)
        {
            const TerrainEditBlock* b = st.block(space, bi * 16, bk * 16);
            if (!b) continue;
            found++;
            bool any = false;
            for (int32_t v : b->delta) any = any || v != 0;
            EXPECT_TRUE(any) << "block " << bi << ", " << bk;
        }
    EXPECT_EQ(found, st.blockCount(space));
}

// Reserve, then write (the order of every path)
void put(TerrainEditStore& st, SpaceId space, int64_t i, int64_t k, int32_t v)
{
    st.reserveFor(space, i, k, v);
    st.writeDelta(space, i, k, v);
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
    // static: usable in the lambdas without a capture on every compiler (MSVC requires one for a local constexpr)
    static constexpr int32_t L = kTerrainHeightLimit;
    // Add at +2^23 and Subtract at -2^23 change nothing
    const KernelRun a =
        runKernel(brush(TerrainBrush::Add, 100, 100, 6, 2), 1, 4096, [](int64_t, int64_t) { return L; });
    EXPECT_EQ(a.changed, 0u);
    const KernelRun s =
        runKernel(brush(TerrainBrush::Subtract, 100, 100, 6, 2), 1, 4096, [](int64_t, int64_t) { return -L; });
    EXPECT_EQ(s.changed, 0u);
    // Add 1 m (s = 1024) at 2^23 - 100 gives exactly 2^23 at the center and counts as changed
    const KernelRun c =
        runKernel(brush(TerrainBrush::Add, 100, 100, 6, 1), 1, 4096, [](int64_t, int64_t) { return L - 100; });
    EXPECT_EQ(c.after(100, 100), L);
    EXPECT_GT(c.changed, 0u);
    // A sample beyond the limit at zero weight stays unchanged (pure kernel: the engine clamps every base)
    const KernelRun z =
        runKernel(brush(TerrainBrush::Add, 0, 0, 8.004, 64), 1, 4096, [](int64_t, int64_t) { return L + 5; });
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

// ---- the modification state (step 2)

TEST(TerrainEditStore, BlocksOnlyForNonZeroChangedDeltas)
{
    TerrainEditStore st({terraformTestTemplate(1)});
    EXPECT_EQ(st.chunkSamples(0), 256);
    EXPECT_EQ(st.maxSample(0), 4096);
    // A zero (or the current value) allocates nothing
    st.reserveFor(0, 100, 100, 0);
    EXPECT_FALSE(st.writeDelta(0, 100, 100, 0));
    EXPECT_EQ(st.blockCount(0), 0u);
    put(st, 0, 100, 100, 5);
    EXPECT_EQ(st.blockCount(0), 1u);
    EXPECT_EQ(st.delta(0, 100, 100), 5);
    EXPECT_EQ(st.delta(0, 101, 100), 0);
    put(st, 0, 111, 111, -3);  // the same block (96..111)
    EXPECT_EQ(st.blockCount(0), 1u);
    st.reserveFor(0, 100, 100, 5);  // equal value: nothing
    put(st, 0, 112, 100, 0);        // zero in a missing block
    EXPECT_EQ(st.blockCount(0), 1u);
    // Back to zero: the block stays allocated and reads 0
    put(st, 0, 100, 100, 0);
    put(st, 0, 111, 111, 0);
    EXPECT_EQ(st.blockCount(0), 1u);
    EXPECT_EQ(st.nonZeroSamples(0), 0u);
    EXPECT_GT(st.memoryBytes(), sizeof(TerrainEditBlock));
}

TEST(TerrainEditStore, NonZeroCountsAndRevision)
{
    TerrainEditStore st({terraformTestTemplate(2)});  // S = 128
    // A border sample of chunks (0,0), (1,0), (0,1), (1,1) written before any entry exists
    put(st, 0, 128, 128, 7);
    put(st, 0, 128, 50, 9);  // border of (0,0) and (1,0)
    // An entry created over them counts them (dense scan)
    st.bumpSequence(chunk(0, 0));
    EXPECT_EQ(st.sequence(chunk(0, 0)), 1u);
    EXPECT_EQ(st.entry(chunk(0, 0))->nonZeroSamples, 2u);
    EXPECT_TRUE(st.hasDeltas(chunk(0, 0)));
    st.setSequence(chunk(1, 1), 4);
    EXPECT_EQ(st.entry(chunk(1, 1))->nonZeroSamples, 1u);
    st.bumpSequence(chunk(1, 0));
    EXPECT_EQ(st.entry(chunk(1, 0))->nonZeroSamples, 2u);
    // Writes update every entry whose closed range contains the sample
    const uint64_t r00 = st.revision(chunk(0, 0)), r11 = st.revision(chunk(1, 1)), r10 = st.revision(chunk(1, 0));
    put(st, 0, 128, 128, 0);
    EXPECT_EQ(st.entry(chunk(0, 0))->nonZeroSamples, 1u);
    EXPECT_EQ(st.entry(chunk(1, 1))->nonZeroSamples, 0u);
    EXPECT_FALSE(st.hasDeltas(chunk(1, 1)));
    EXPECT_EQ(st.sequence(chunk(1, 1)), 4u);  // keeps its state
    EXPECT_GT(st.revision(chunk(0, 0)), r00);
    EXPECT_GT(st.revision(chunk(1, 1)), r11);
    EXPECT_GT(st.revision(chunk(1, 0)), r10);
    // An interior sample of (1, 0) only
    const uint64_t a = st.revision(chunk(0, 0)), b = st.revision(chunk(1, 0));
    put(st, 0, 200, 20, -2);
    EXPECT_EQ(st.revision(chunk(0, 0)), a);
    EXPECT_GT(st.revision(chunk(1, 0)), b);
    EXPECT_EQ(st.entry(chunk(1, 0))->nonZeroSamples, 2u);
    // bumpSequence does not change the revision; writing the current value changes nothing
    const uint64_t c = st.revision(chunk(1, 0));
    st.bumpSequence(chunk(1, 0));
    EXPECT_EQ(st.revision(chunk(1, 0)), c);
    EXPECT_FALSE(st.writeDelta(0, 200, 20, -2));
    EXPECT_EQ(st.revision(chunk(1, 0)), c);
    // bumpSequence alone creates an entry without deltas
    st.bumpSequence(chunk(5, 5));
    EXPECT_EQ(st.sequence(chunk(5, 5)), 1u);
    EXPECT_FALSE(st.hasDeltas(chunk(5, 5)));
    EXPECT_EQ(st.revision(chunk(5, 5)), 0u);
    EXPECT_EQ(st.chunkCount(), 4u);
    // A reserved entry node is inserted without allocation and counts the existing deltas
    TerrainEditStore::EntryNode node = st.makeEntry(chunk(0, 1));
    EXPECT_EQ(st.entry(chunk(0, 1)), nullptr);
    st.insertEntry(std::move(node));
    EXPECT_EQ(st.sequence(chunk(0, 1)), 0u);
    EXPECT_EQ(st.entry(chunk(0, 1))->nonZeroSamples, 0u);
}

TEST(TerrainEditStore, DenseDeltaAndContentHash)
{
    TerrainEditStore st({terraformTestTemplate(2)});
    const int64_t N = 129;
    for (int64_t k = 120; k <= 136; ++k)
        for (int64_t i = 0; i <= 10; ++i) put(st, 0, 128 + i - 5, k, int32_t(i * 100 + k));
    st.bumpSequence(chunk(0, 0));
    st.bumpSequence(chunk(1, 0));
    st.bumpSequence(chunk(0, 1));
    std::vector<int32_t> a, b, c;
    st.denseDelta(chunk(0, 0), a);
    st.denseDelta(chunk(1, 0), b);
    st.denseDelta(chunk(0, 1), c);
    ASSERT_EQ(a.size(), size_t(N * N));
    // The shared border column i = 128 agrees
    for (int64_t k = 0; k < N; ++k) EXPECT_EQ(a[size_t(k * N + 128)], b[size_t(k * N + 0)]);
    // ... and the shared row k = 128
    for (int64_t i = 0; i < N; ++i) EXPECT_EQ(a[size_t(128 * N + i)], c[size_t(0 * N + i)]);
    EXPECT_EQ(a[size_t(125 * N + 123)], 125);
    EXPECT_EQ(b[size_t(136 * 0 + 120 * N + 5)], 1000 + 120);
    uint32_t nz = 0;
    for (int32_t v : a) nz += v != 0 ? 1 : 0;
    EXPECT_EQ(st.entry(chunk(0, 0))->nonZeroSamples, nz);
    // The content hash is FNV-1a over the dense delta, cached until a delta changes
    Hasher h;
    for (int32_t v : a) h.addI32(v);
    EXPECT_EQ(st.contentHash(chunk(0, 0)), h.value());
    EXPECT_FALSE(st.entry(chunk(0, 0))->hashDirty);
    const uint64_t neighbour = st.contentHash(chunk(1, 0));
    put(st, 0, 0, 0, 1);  // inside (0, 0) only
    EXPECT_TRUE(st.entry(chunk(0, 0))->hashDirty);
    EXPECT_FALSE(st.entry(chunk(1, 0))->hashDirty);
    EXPECT_NE(st.contentHash(chunk(0, 0)), h.value());
    EXPECT_EQ(st.contentHash(chunk(1, 0)), neighbour);
}

TEST(TerrainEditStore, TerrainSampleAndHeightOverDeltas)
{
    for (int spacing : {1, 2})
    {
        World w(editSetup(spacing));
        TerrainEditStore& st = w.terrainEditsForUpdate();
        const int64_t S      = 256 / spacing;
        // Deltas over a patch around the chunk corner (S, S), incl. one that clamps
        std::mt19937 rng(3);
        std::uniform_int_distribution<int32_t> d(-5000, 5000);
        for (int64_t k = S - 6; k <= S + 6; ++k)
            for (int64_t i = S - 6; i <= S + 6; ++i) put(st, 0, i, k, d(rng));
        put(st, 0, S + 2, S + 2, (1 << 24));
        auto expected = [&](int64_t i, int64_t k)
        {
            const int64_t v = int64_t(w.terrain().sample(0, i, k)) + st.delta(0, i, k);
            return clampTerrainHeight(int32_t(std::min<int64_t>(v, int64_t(1) << 30)));
        };
        for (int64_t k = S - 7; k <= S + 7; ++k)
            for (int64_t i = S - 7; i <= S + 7; ++i) EXPECT_EQ(w.terrainSample(0, i, k), expected(i, k));
        EXPECT_EQ(w.terrainSample(0, S + 2, S + 2), kTerrainHeightLimit);
        // terrainHeight: the §12.2 triangle rule over the edited samples, at samples and random points incl. diagonals
        std::uniform_real_distribution<double> u(-6.0 * spacing, 6.0 * spacing);
        for (int n = 0; n < 400; ++n)
        {
            double x = 256 + u(rng), z = 256 + u(rng);
            if (n % 4 == 0) z = x;  // on a cell diagonal (exact for dyadic values)
            if (n % 4 == 1)
            {
                x = std::floor(x / spacing) * spacing;
                z = std::floor(z / spacing) * spacing;
            }
            const double gx = x / spacing, gz = z / spacing;
            const int64_t ci = int64_t(std::floor(gx)), cj = int64_t(std::floor(gz));
            const double fx = gx - double(ci), fz = gz - double(cj);
            const int32_t h00 = expected(ci, cj), h10 = expected(ci + 1, cj), h01 = expected(ci, cj + 1),
                          h11 = expected(ci + 1, cj + 1);
            EXPECT_EQ(w.terrainHeight(0, x, z), terrainSurfaceDouble(h00, h10, h01, h11, fx, fz) * kHeightUnit);
        }
    }
}

TEST(TerrainEditStore, FlatSpaceHeightWithDeltas)
{
    // D1: a space without terrain reads 0 until it has a delta
    WorldSetup s;
    s.profiling = false;
    World w(s);
    EXPECT_EQ(w.terrainHeight(0, 100, 100), 0.0);
    put(w.terrainEditsForUpdate(), 0, 50, 50, 2048);  // spacing 2: sample (50, 50) at (100, 100) m
    EXPECT_EQ(w.terrainSample(0, 50, 50), 2048);
    EXPECT_EQ(w.terrainHeight(0, 100, 100), 2.0);
    EXPECT_EQ(w.terrainHeight(0, 101, 100), 1.0);
}

// ---- edits (step 3)

TEST(TerrainEdit, CornerEditChangesSharedSamplesOnce)
{
    World w(editSetup(1));
    const int32_t before      = w.terrainSample(0, 256, 256);
    const TerrainEditResult r = applyTerrainEdit(w, 0, brush(TerrainBrush::Add, 256, 256, 8, 1));
    ASSERT_EQ(r.chunks.size(), 4u);
    EXPECT_EQ(r.chunks[0].chunk, chunk(0, 0));
    EXPECT_EQ(r.chunks[1].chunk, chunk(0, 1));
    EXPECT_EQ(r.chunks[2].chunk, chunk(1, 0));
    EXPECT_EQ(r.chunks[3].chunk, chunk(1, 1));
    for (const auto& c : r.chunks)
    {
        EXPECT_EQ(c.sequence, 1u);
        EXPECT_EQ(w.terrainEdits().sequence(c.chunk), 1u);
        EXPECT_TRUE(w.terrainEdits().hasDeltas(c.chunk));
    }
    // The shared sample changed once (one copy)
    EXPECT_EQ(w.terrainSample(0, 256, 256), before + 1024);
    EXPECT_EQ(w.terrainEdits().delta(0, 256, 256), 1024);
    // The box: 8 m strictly inside at 1 m
    EXPECT_EQ(r.i0, 249);
    EXPECT_EQ(r.i1, 263);
    EXPECT_EQ(r.k0, 249);
    EXPECT_EQ(r.k1, 263);
    // The parts are the box clipped per chunk (chunk-local); (0, 0) holds 249..256
    EXPECT_EQ(r.chunks[0].x0, 249);
    EXPECT_EQ(r.chunks[0].w, 8);
    EXPECT_EQ(r.chunks[3].x0, 0);
    EXPECT_EQ(r.chunks[3].w, 8);
    // denseDelta of neighbours agrees on the shared border
    std::vector<int32_t> a, b;
    w.terrainEdits().denseDelta(chunk(0, 0), a);
    w.terrainEdits().denseDelta(chunk(1, 0), b);
    for (int k = 0; k < 257; ++k) EXPECT_EQ(a[size_t(k * 257 + 256)], b[size_t(k * 257)]);
    // One send-list entry with four parts; one event (between steps: at once, not in the change list)
    ASSERT_EQ(w.replication().hostStateList().size(), 1u);
    EXPECT_EQ(w.replication().hostStateList()[0].parts.size(), 4u);
    EXPECT_EQ(pendingEvents(w), 1u);
    EXPECT_TRUE(w.pendingTerrainChanges().empty());
    EXPECT_EQ(w.terrainEdits().stats().edits, 1u);
    EXPECT_EQ(w.terrainEdits().stats().chunksEdited, 4u);
    EXPECT_EQ(w.terrainEdits().stats().changedSamples, r.changedSamples);
    expectBlocksNonZero(w, 0);
    // The step sends (no peers) and clears the list; the event is readable in it
    w.step();
    EXPECT_TRUE(w.replication().hostStateList().empty());
    ASSERT_EQ(w.events.read<TerrainEditedEvent>().size(), 1u);
    const TerrainEditedEvent& e = w.events.read<TerrainEditedEvent>()[0];
    EXPECT_EQ(e.x0, 249.0);
    EXPECT_EQ(e.x1, 263.0);
    EXPECT_EQ(e.z0, 249.0);
    EXPECT_EQ(e.z1, 263.0);
}

TEST(TerrainEdit, InteriorEditChangesOneChunk)
{
    World w(editSetup(2));
    const TerrainEditResult r = applyTerrainEdit(w, 0, brush(TerrainBrush::Subtract, 600, 300, 20, 2));
    ASSERT_EQ(r.chunks.size(), 1u);
    EXPECT_EQ(r.chunks[0].chunk, chunk(2, 1));
    EXPECT_EQ(w.terrainEdits().chunkCount(), 1u);
    EXPECT_EQ(w.terrainEdits().sequence(chunk(2, 1)), 1u);
    // The parts' box equals the changed box (global sample 300, 150 at the center)
    EXPECT_EQ(r.chunks[0].x0 + 2 * 128, r.i0);
    EXPECT_EQ(r.chunks[0].z0 + 128, r.k0);
}

TEST(TerrainEdit, NoOpEditsLeaveNoState)
{
    auto expectNothing = [](World& w, const TerrainEditResult& r)
    {
        EXPECT_TRUE(r.chunks.empty());
        EXPECT_EQ(r.changedSamples, 0u);
        EXPECT_EQ(w.terrainEdits().chunkCount(), 0u);
        EXPECT_EQ(w.terrainEdits().blockCount(0), 0u);
        EXPECT_EQ(pendingEvents(w), 0u);
        EXPECT_TRUE(w.replication().hostStateList().empty());
    };
    // Flatten on a field at the target and Smooth on a constant field (a hook-less open world: base 0)
    {
        WorldSetup s;
        s.profiling = false;
        World w(s);
        expectNothing(w, applyTerrainEdit(w, 0, brush(TerrainBrush::Flatten, 500, 500, 30, 1, 0)));
        expectNothing(w, applyTerrainEdit(w, 0, brush(TerrainBrush::Smooth, 500, 500, 30, 1)));
        EXPECT_EQ(w.terrainEdits().stats().noopEdits, 2u);
        EXPECT_EQ(w.terrainEdits().stats().edits, 2u);
    }
    // Add on a space at the height limit everywhere
    {
        WorldSetup s;
        s.profiling = false;
        s.config.spaceTemplates.push_back(testSpaceTemplate(16, 2, maxHook));
        World w(s);
        expectNothing(w, applyTerrainEdit(w, 0, brush(TerrainBrush::Add, 500, 500, 30, 10)));
        EXPECT_EQ(w.terrainSample(0, 250, 250), kTerrainHeightLimit);
    }
}

TEST(TerrainEdit, BlocksOnlyForChangedSamplesAndReturnToZero)
{
    WorldSetup s;
    s.profiling = false;
    World w(s);  // flat, spacing 2
    // A mound inside block (2, 2), then a Smooth whose read ring reaches into the next blocks
    applyTerrainEdit(w, 0, brush(TerrainBrush::Add, 72, 72, 5, 1));
    const size_t blocks = w.terrainEdits().blockCount(0);
    EXPECT_EQ(blocks, 1u);
    applyTerrainEdit(w, 0, brush(TerrainBrush::Smooth, 72, 72, 7, 1));
    expectBlocksNonZero(w, 0);
    // Back to zero: Add then Subtract of the same brush cancel exactly
    World v(s);
    applyTerrainEdit(v, 0, brush(TerrainBrush::Add, 1000, 1000, 9, 1.5));
    applyTerrainEdit(v, 0, brush(TerrainBrush::Subtract, 1000, 1000, 9, 1.5));
    EXPECT_EQ(v.terrainEdits().sequence(chunk(3, 3)), 2u);
    EXPECT_FALSE(v.terrainEdits().hasDeltas(chunk(3, 3)));
    EXPECT_EQ(v.terrainEdits().nonZeroSamples(0), 0u);
    EXPECT_EQ(v.terrainHeight(0, 1000, 1000), 0.0);
}

TEST(TerrainEdit, SampleAndHeightAfterEdits)
{
    for (int spacing : {1, 2})
    {
        World w(editSetup(spacing));
        std::mt19937 rng(11);
        std::uniform_real_distribution<double> pos(200, 312), rad(2, 30), str(0, 1);
        std::vector<int32_t> base;
        for (int n = 0; n < 12; ++n)
        {
            const TerrainBrush::Op op = TerrainBrush::Op(n % 4);
            const double strength     = op <= TerrainBrush::Subtract ? 4 * str(rng) : str(rng);
            applyTerrainEdit(w, 0, brush(op, pos(rng), pos(rng), rad(rng), strength, 1.0 + str(rng)));
        }
        const int64_t lo = 180 / spacing, hi = 330 / spacing;
        auto expected = [&](int64_t i, int64_t k)
        {
            const int64_t v = int64_t(w.terrain().sample(0, i, k)) + w.terrainEdits().delta(0, i, k);
            return clampTerrainHeight(int32_t(v));
        };
        for (int64_t k = lo; k <= hi; ++k)
            for (int64_t i = lo; i <= hi; ++i) ASSERT_EQ(w.terrainSample(0, i, k), expected(i, k));
        std::uniform_real_distribution<double> u(190, 320);
        for (int n = 0; n < 500; ++n)
        {
            const double x = u(rng), z = n % 3 == 0 ? x : u(rng);
            const double gx = x / spacing, gz = z / spacing;
            const int64_t ci = int64_t(std::floor(gx)), cj = int64_t(std::floor(gz));
            const double fx = gx - double(ci), fz = gz - double(cj);
            EXPECT_EQ(w.terrainHeight(0, x, z),
                      terrainSurfaceDouble(expected(ci, cj), expected(ci + 1, cj), expected(ci, cj + 1),
                                           expected(ci + 1, cj + 1), fx, fz) *
                          kHeightUnit);
        }
        expectBlocksNonZero(w, 0);
    }
}

TEST(TerrainEdit, GatherFromResidentTilesEqualsTheHook)
{
    World a(editSetup(2)), b(editSetup(2));
    RecordingRenderSink sink;
    a.setRenderSink(&sink);
    ASSERT_TRUE(a.terrain().completeView(0, Vec3(256, 30, 256)));
    // A rectangle across four level-0 tiles
    SampleRect r;
    r.i0 = 100;
    r.k0 = 110;
    r.i1 = 150;
    r.k1 = 140;
    std::vector<int32_t> baseA, heightA, baseB, heightB;
    const uint64_t ha = a.terrain().baseHookSamples(), hb = b.terrain().baseHookSamples();
    gatherSamples(a, 0, r, baseA, heightA);
    gatherSamples(b, 0, r, baseB, heightB);
    EXPECT_EQ(a.terrain().baseHookSamples() - ha, 0u);
    EXPECT_EQ(b.terrain().baseHookSamples() - hb, uint64_t(r.width() * r.height()));
    EXPECT_EQ(baseA, baseB);
    EXPECT_EQ(heightA, heightB);
    // An edit on the resident tiles reads no hook
    const uint64_t h0 = a.terrain().baseHookSamples();
    applyTerrainEdit(a, 0, brush(TerrainBrush::Smooth, 256, 256, 20, 1));
    applyTerrainEdit(a, 0, brush(TerrainBrush::Add, 256, 256, 20, 1));
    EXPECT_EQ(a.terrain().baseHookSamples() - h0, 0u);
    a.setRenderSink(nullptr);
}

TEST(TerrainEdit, SpaceWithoutTerrain)
{
    // D1: an open world without a hook is flat at 0; an edit changes terrainHeight
    WorldSetup s;
    s.profiling = false;
    World w(s);
    EXPECT_EQ(w.terrainHeight(0, 300, 300), 0.0);
    w.editTerrain(0, brush(TerrainBrush::Add, 300, 300, 10, 2));
    EXPECT_EQ(w.terrainHeight(0, 300, 300), 2.0);
    EXPECT_GT(w.terrainHeight(0, 305, 300), 0.0);
    EXPECT_EQ(w.terrainHeight(0, 400, 300), 0.0);
}

// Regression (PR review of S22): TerrainEditStore divided by the baseSpacing of an enclosed space, which is ignored and
// not validated (0 stopped the World constructor with a division by zero)
TEST(TerrainEdit, EnclosedSpaceWithZeroSpacing)
{
    WorldSetup s         = editSetup(2);
    SpaceTemplate cellar = SpaceTemplate::enclosed("cellar");
    cellar.baseSpacing   = 0;
    s.config.spaceTemplates.push_back(cellar);
    World w(s);
    EXPECT_EQ(w.terrainEdits().spaceCount(), 2u);
    EXPECT_EQ(w.terrainEdits().nonZeroSamples(1), 0u);
    EXPECT_GT(applyTerrainEdit(w, 0, brush(TerrainBrush::Add, 300, 300, 10, 2)).changedSamples, 0u);
}

// Regression (PR review of S22): an edit of a chunk whose sequence is UINT32_MAX (a save accepts it) wrapped the
// sequence to 0, which saves and messages reject; the edit is refused and changes nothing
TEST(TerrainEdit, ExhaustedSequenceRefusesTheEdit)
{
    World w(editSetup(2));
    w.terrainEditsForUpdate().setSequence(chunk(1, 1), std::numeric_limits<uint32_t>::max());
    const std::vector<uint8_t> before = w.save();
    const TerrainEditResult r         = applyTerrainEdit(w, 0, brush(TerrainBrush::Add, 300, 300, 10, 2));
    EXPECT_EQ(r.changedSamples, 0u);
    EXPECT_TRUE(r.chunks.empty());
    EXPECT_EQ(w.terrainEdits().sequence(chunk(1, 1)), std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(w.terrainEdits().nonZeroSamples(0), 0u);
    EXPECT_TRUE(w.replication().hostStateList().empty());
    EXPECT_EQ(pendingEvents(w), 0u);
    EXPECT_EQ(w.save(), before);
    // Other chunks stay editable
    EXPECT_GT(applyTerrainEdit(w, 0, brush(TerrainBrush::Add, 600, 600, 10, 2)).changedSamples, 0u);
}

// ---- editTerrain in the step (step 3)

namespace
{
struct StepHooks
{
    std::function<void(World&)> phase3, phase4a, phase4b, phase5;
};
StepHooks g_hooks;
void phase3System(World& w, double)
{
    if (g_hooks.phase3) g_hooks.phase3(w);
}
void phase4aSystem(World& w, double)
{
    if (g_hooks.phase4a) g_hooks.phase4a(w);
}
void phase4bSystem(World& w, double)
{
    if (g_hooks.phase4b) g_hooks.phase4b(w);
}
void phase5System(World& w, double)
{
    if (g_hooks.phase5) g_hooks.phase5(w);
}
WorldSetup hookedSetup(int spacing)
{
    WorldSetup s = editSetup(spacing);
    s.systems    = [](SystemList& l)
    {
        l.add(Phase::Behaviour, "test_terrain3", phase3System, RunOn::Both);
        l.add(Phase::Gameplay, "test_terrain4a", phase4aSystem, RunOn::Both);
        l.add(Phase::Gameplay, "test_terrain4b", phase4bSystem, RunOn::Both);
        l.add(Phase::Physics, "test_terrain5", phase5System, RunOn::Both);
    };
    return s;
}
class TerrainEditWorld : public ::testing::Test
{
   protected:
    void SetUp() override { g_hooks = StepHooks(); }
    void TearDown() override { g_hooks = StepHooks(); }
};
}  // namespace

TEST_F(TerrainEditWorld, CommandAppliesAtThePhaseFlush)
{
    World w(hookedSetup(2));
    const double base = w.terrainHeight(0, 500, 500);
    double seen4b = 0, seen5 = 0;
    size_t listed5  = 0;
    g_hooks.phase4a = [](World& world)
    {
        if (world.stepIndex() == 0) world.editTerrain(0, brush(TerrainBrush::Add, 500, 500, 10, 3));
    };
    g_hooks.phase4b = [&](World& world) { seen4b = world.terrainHeight(0, 500, 500); };
    g_hooks.phase5  = [&](World& world)
    {
        if (world.stepIndex() != 0) return;
        seen5   = world.terrainHeight(0, 500, 500);
        listed5 = world.pendingTerrainChanges().size();
    };
    w.step();
    EXPECT_EQ(seen4b, base);
    EXPECT_EQ(seen5, base + 3);
    // Listed in the step, emitted at the commit: one event, readable in the next step
    EXPECT_EQ(listed5, 1u);
    EXPECT_TRUE(w.pendingTerrainChanges().empty());
    EXPECT_EQ(pendingEvents(w), 1u);
    size_t read     = 0;
    g_hooks.phase4b = [&](World& world) { read = world.events.read<TerrainEditedEvent>().size(); };
    w.step();
    EXPECT_EQ(read, 1u);
    EXPECT_EQ(pendingEvents(w), 0u);
}

TEST_F(TerrainEditWorld, TwoEditsInRecordingOrderAndBetweenSteps)
{
    World w(hookedSetup(2));
    // Add, then Flatten at strength 1 to a target: the center ends at the target only in this order
    g_hooks.phase4a = [](World& world)
    {
        if (world.stepIndex() != 0) return;
        world.editTerrain(0, brush(TerrainBrush::Add, 700, 700, 10, 5));
        world.editTerrain(0, brush(TerrainBrush::Flatten, 700, 700, 10, 1, 1.25));
    };
    w.step();
    EXPECT_EQ(w.terrainSample(0, 350, 350), 1280);
    EXPECT_EQ(w.terrainEdits().sequence(chunk(2, 2)), 2u);
    // Between steps: at once
    w.editTerrain(0, brush(TerrainBrush::Add, 700, 700, 10, 1));
    EXPECT_EQ(w.terrainSample(0, 350, 350), 1280 + 1024);
    EXPECT_EQ(w.terrainEdits().sequence(chunk(2, 2)), 3u);
}

TEST_F(TerrainEditWorld, AbortedStepKeepsTheChange)
{
    // Regression: the event of an edit applied in a finished phase of an aborted step was lost (§7.6)
    World w(hookedSetup(2));
    int aborts = 0, editsDone = 0;
    std::vector<size_t> readPerStep;
    g_hooks.phase3 = [&](World& world)
    {
        if (editsDone == 0)
        {
            world.editTerrain(0, brush(TerrainBrush::Add, 300, 300, 6, 1));
            editsDone++;
        }
    };
    g_hooks.phase4a = [&](World& world)
    {
        readPerStep.push_back(world.events.read<TerrainEditedEvent>().size());
        if (aborts < 2)
        {
            aborts++;
            throw std::runtime_error("abort phase 4");
        }
    };
    EXPECT_THROW(w.step(), std::runtime_error);
    // The edit stays applied; its change is kept
    EXPECT_EQ(w.terrainEdits().sequence(chunk(1, 1)), 1u);
    ASSERT_EQ(w.pendingTerrainChanges().size(), 1u);
    EXPECT_EQ(pendingEvents(w), 0u);
    // The retried step emits it before beginStep: readable there (and the retry aborts too)
    EXPECT_THROW(w.step(), std::runtime_error);
    EXPECT_TRUE(w.pendingTerrainChanges().empty());
    EXPECT_EQ(pendingEvents(w), 1u);  // turned back into a pending event
    w.step();
    w.step();
    ASSERT_EQ(readPerStep.size(), 4u);
    EXPECT_EQ(readPerStep[0], 0u);
    EXPECT_EQ(readPerStep[1], 1u);
    EXPECT_EQ(readPerStep[2], 1u);
    EXPECT_EQ(readPerStep[3], 0u);
    // The edit's send-list entry was kept by the aborted steps and cleared by the completed one
    EXPECT_TRUE(w.replication().hostStateList().empty());
}

TEST_F(TerrainEditWorld, ViolationsStopTheProgram)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    // An invalid brush, an unknown space, an enclosed space: at the call, between steps and from a phase-4 system
    WorldSetup s = hookedSetup(2);
    s.config.spaceTemplates.push_back(SpaceTemplate::enclosed("cellar"));
    EXPECT_DEATH(
        {
            World w(s);
            w.editTerrain(0, brush(TerrainBrush::Add, 10, 10, 200, 1));
        },
        "radius");
    EXPECT_DEATH(
        {
            World w(s);
            w.editTerrain(7, brush(TerrainBrush::Add, 10, 10, 2, 1));
        },
        "unknown space");
    EXPECT_DEATH(
        {
            World w(s);
            w.editTerrain(1, brush(TerrainBrush::Add, 10, 10, 2, 1));
        },
        "enclosed");
    EXPECT_DEATH(
        {
            World w(s);
            g_hooks.phase4a = [](World& world) { world.editTerrain(0, brush(TerrainBrush::Smooth, 10, 10, 2, 3)); };
            w.step();
        },
        "strength");
    EXPECT_DEATH(
        {
            World w(s);
            g_hooks.phase4a = [](World& world) { world.editTerrain(1, brush(TerrainBrush::Add, 10, 10, 2, 1)); };
            w.step();
        },
        "enclosed");
    // On a client
    auto net = LoopbackTransport::createNetwork(2);
    EXPECT_DEATH(
        {
            WorldSetup c = hookedSetup(2);
            c.transport  = net[1].get();
            c.role       = NetRole::Client;
            c.localPeer  = 1;
            World w(c);
            w.editTerrain(0, brush(TerrainBrush::Add, 10, 10, 2, 1));
        },
        "client");
    EXPECT_DEATH(
        {
            WorldSetup c = hookedSetup(2);
            c.transport  = net[1].get();
            c.role       = NetRole::Client;
            c.localPeer  = 1;
            World w(c);
            g_hooks.phase4a = [](World& world) { world.editTerrain(0, brush(TerrainBrush::Add, 10, 10, 2, 1)); };
            w.step();
        },
        "client");
}

// ---- codec, saves and loads (step 4)

TEST(TerrainDeltaCodec, RoundTripsAndBounds)
{
    std::mt19937 rng(5);
    std::uniform_int_distribution<int32_t> d(-kMaxTerrainDelta, kMaxTerrainDelta), small(-3000, 3000);
    for (size_t count : {size_t(1), size_t(17 * 9), size_t(257 * 257)})
        for (bool zstd : {false, true})
        {
            std::vector<int32_t> v(count);
            for (size_t i = 0; i < count; ++i) v[i] = i % 7 == 0 ? d(rng) : small(rng);
            v[0] = count > 1 ? kMaxTerrainDelta : -kMaxTerrainDelta;
            std::vector<uint8_t> bytes;
            TerrainDeltaCodec::encode(v.data(), v.size(), zstd, bytes);
            std::vector<int32_t> back(count);
            std::string e;
            ASSERT_TRUE(TerrainDeltaCodec::decode(bytes.data(), bytes.size(), zstd, count, back.data(), &e)) << e;
            EXPECT_EQ(back, v);
            // Bytes left over, a short count, a long count
            std::vector<uint8_t> more = bytes;
            more.push_back(0);
            EXPECT_FALSE(TerrainDeltaCodec::decode(more.data(), more.size(), zstd, count, back.data(), &e));
            if (count > 1)
            {
                EXPECT_FALSE(TerrainDeltaCodec::decode(bytes.data(), bytes.size(), zstd, count - 1, back.data(), &e));
            }
            std::vector<int32_t> big(count + 1);
            EXPECT_FALSE(TerrainDeltaCodec::decode(bytes.data(), bytes.size(), zstd, count + 1, big.data(), &e));
        }
    // Every rectangle decodes on its own: the predictor starts at 0
    const int32_t a[] = {1000, 1001}, b[] = {1000, 999};
    std::vector<uint8_t> ea, eb;
    TerrainDeltaCodec::encode(a, 2, false, ea);
    TerrainDeltaCodec::encode(b, 2, false, eb);
    EXPECT_EQ(ea[0], eb[0]);
    EXPECT_EQ(ea.size(), eb.size());
    // A delta beyond 2^24 (raw: zigzag of 2^24 + 1)
    std::vector<uint8_t> over;
    writeVarint(over, zigzag(int64_t(kMaxTerrainDelta) + 1));
    int32_t one = 0;
    std::string e;
    EXPECT_FALSE(TerrainDeltaCodec::decode(over.data(), over.size(), false, 1, &one, &e));
    EXPECT_NE(e.find("2^24"), std::string::npos) << e;
    // The zstd bound: a frame larger than count * 4 bytes
    std::vector<int32_t> many(400, 5000000);
    std::vector<uint8_t> z;
    TerrainDeltaCodec::encode(many.data(), many.size(), true, z);
    std::vector<int32_t> out(100);
    EXPECT_FALSE(TerrainDeltaCodec::decode(z.data(), z.size(), true, 100, out.data(), &e));
    // A truncated zstd stream
    std::vector<int32_t> fine(400);
    EXPECT_TRUE(TerrainDeltaCodec::decode(z.data(), z.size(), true, 400, fine.data(), &e));
    EXPECT_FALSE(TerrainDeltaCodec::decode(z.data(), z.size() - 3, true, 400, fine.data(), &e));
}

namespace
{
// The sections of a save blob: (tag, payload)
std::vector<std::pair<uint32_t, std::vector<uint8_t>>> sections(const std::vector<uint8_t>& blob)
{
    std::vector<std::pair<uint32_t, std::vector<uint8_t>>> out;
    size_t pos = 8;
    while (pos + 12 <= blob.size())
    {
        uint32_t tag = 0;
        uint64_t len = 0;
        std::memcpy(&tag, blob.data() + pos, 4);
        std::memcpy(&len, blob.data() + pos + 4, 8);
        out.push_back({tag, std::vector<uint8_t>(blob.begin() + std::ptrdiff_t(pos + 12),
                                                 blob.begin() + std::ptrdiff_t(pos + 12 + len))});
        pos += 12 + size_t(len);
    }
    return out;
}
// The blob with its TERR payload replaced (inserted before ENTS if absent)
std::vector<uint8_t> withTerr(const std::vector<uint8_t>& blob, const std::vector<uint8_t>& payload)
{
    std::vector<uint8_t> out(blob.begin(), blob.begin() + 8);
    bool done = false;
    for (auto& [tag, bytes] : sections(blob))
    {
        if (tag == kTagTerrain || (tag == kTagEntities && !done))
        {
            const uint32_t t = kTagTerrain;
            const uint64_t n = payload.size();
            out.insert(out.end(), reinterpret_cast<const uint8_t*>(&t), reinterpret_cast<const uint8_t*>(&t) + 4);
            out.insert(out.end(), reinterpret_cast<const uint8_t*>(&n), reinterpret_cast<const uint8_t*>(&n) + 8);
            out.insert(out.end(), payload.begin(), payload.end());
            done = true;
            if (tag == kTagTerrain) continue;
        }
        const uint64_t n = bytes.size();
        out.insert(out.end(), reinterpret_cast<const uint8_t*>(&tag), reinterpret_cast<const uint8_t*>(&tag) + 4);
        out.insert(out.end(), reinterpret_cast<const uint8_t*>(&n), reinterpret_cast<const uint8_t*>(&n) + 8);
        out.insert(out.end(), bytes.begin(), bytes.end());
    }
    return out;
}
struct TerrChunk
{
    int32_t x = 0, z = 0;
    uint32_t sequence = 1;
    uint8_t flags     = 1;
    std::vector<int32_t> deltas;   // N * N with flags & 1
    std::vector<uint8_t> encoded;  // overrides the encoding when not empty
};
template <class T>
void put(std::vector<uint8_t>& out, T v)
{
    for (size_t i = 0; i < sizeof(T); ++i) out.push_back(uint8_t(uint64_t(v) >> (8 * i)));
}
std::vector<uint8_t> terrPayload(uint8_t spacing, uint16_t N, const std::vector<TerrChunk>& chunks, SpaceId space = 0)
{
    std::vector<uint8_t> out;
    put<uint32_t>(out, 1);  // the section's version
    put<uint32_t>(out, 1);
    put<uint16_t>(out, space);
    put<uint8_t>(out, spacing);
    put<uint16_t>(out, N);
    put<uint32_t>(out, uint32_t(chunks.size()));
    for (const TerrChunk& c : chunks)
    {
        put<int32_t>(out, c.x);
        put<int32_t>(out, c.z);
        put<uint32_t>(out, c.sequence);
        put<uint8_t>(out, c.flags);
        if (!(c.flags & 1)) continue;
        std::vector<uint8_t> e = c.encoded;
        if (e.empty()) TerrainDeltaCodec::encode(c.deltas.data(), c.deltas.size(), true, e);
        put<uint64_t>(out, uint64_t(e.size()));
        out.insert(out.end(), e.begin(), e.end());
    }
    return out;
}
}  // namespace

TEST(TerrainEditSave, RoundTripAndHash)
{
    World a(editSetup(1));
    const std::vector<uint8_t> empty = a.save();
    const uint64_t emptyHash         = a.hash();
    // No TERR without chunks with state; a no-op edit leaves the bytes and the hash
    applyTerrainEdit(a, 0, brush(TerrainBrush::Flatten, 100, 100, 5, 0));
    EXPECT_EQ(a.save(), empty);
    EXPECT_EQ(a.hash(), emptyHash);
    for (auto& [tag, bytes] : sections(empty)) EXPECT_NE(tag, kTagTerrain);
    // Edits in two chunks (one across their border), a returned-to-zero chunk and a bumpSequence-only chunk
    applyTerrainEdit(a, 0, brush(TerrainBrush::Add, 256, 100, 12, 2));
    EXPECT_NE(a.hash(), emptyHash);
    applyTerrainEdit(a, 0, brush(TerrainBrush::Smooth, 250, 110, 12, 1));
    applyTerrainEdit(a, 0, brush(TerrainBrush::Add, 2000, 2000, 6, 1));
    applyTerrainEdit(a, 0, brush(TerrainBrush::Subtract, 2000, 2000, 6, 1));
    a.terrainEditsForUpdate().bumpSequence(chunk(9, 3));
    EXPECT_FALSE(a.terrainEdits().hasDeltas(chunk(7, 7)));
    const std::vector<uint8_t> blob = a.save();
    // The section at its D7 position: after SCHM, before ENTS
    std::vector<uint32_t> tags;
    for (auto& [tag, bytes] : sections(blob)) tags.push_back(tag);
    ASSERT_GE(tags.size(), 4u);
    EXPECT_EQ(tags[1], kTagSchema);
    EXPECT_EQ(tags[2], kTagTerrain);
    EXPECT_EQ(tags[3], kTagEntities);
    // Load into a fresh world: equal bytes and hash, no hook, blocks only with non-zero deltas
    World b(editSetup(1));
    const uint64_t hooks = b.terrain().baseHookSamples();
    const LoadResult r   = b.tryLoad(blob);
    ASSERT_TRUE(r.ok()) << r.error;
    EXPECT_EQ(b.terrain().baseHookSamples(), hooks);
    EXPECT_EQ(b.save(), blob);
    EXPECT_EQ(b.hash(), a.hash());
    EXPECT_EQ(b.terrainEdits().sequence(chunk(7, 7)), 2u);
    EXPECT_FALSE(b.terrainEdits().hasDeltas(chunk(7, 7)));
    EXPECT_EQ(b.terrainEdits().sequence(chunk(9, 3)), 1u);
    EXPECT_EQ(b.terrainEdits().sequence(chunk(0, 0)), 2u);
    EXPECT_EQ(b.terrainEdits().sequence(chunk(1, 0)), 2u);
    for (int64_t k = 80; k <= 130; ++k)
        for (int64_t i = 230; i <= 280; ++i) ASSERT_EQ(b.terrainSample(0, i, k), a.terrainSample(0, i, k));
    expectBlocksNonZero(b, 0);
    EXPECT_LT(b.terrainEdits().blockCount(0), a.terrainEdits().blockCount(0));  // a keeps the zeroed blocks
    // Loading the empty save clears the state
    ASSERT_TRUE(b.tryLoad(empty).ok());
    EXPECT_EQ(b.terrainEdits().chunkCount(), 0u);
    EXPECT_EQ(b.hash(), emptyHash);
    // diffSaves names a differing chunk with its first differing sample
    World c(editSetup(1));
    ASSERT_TRUE(c.tryLoad(blob).ok());
    applyTerrainEdit(c, 0, brush(TerrainBrush::Add, 2010, 2000, 3, 1));
    const SaveDiff d = diffSaves(blob, c.save());
    ASSERT_TRUE(d.error.empty()) << d.error;
    bool found = false;
    for (const SaveDifference& x : d.differences)
        if (x.kind == "terrain")
        {
            found = true;
            EXPECT_NE(x.describe().find("chunk (7, 7)"), std::string::npos) << x.describe();
        }
    EXPECT_TRUE(found);
    const SaveDiff d2 = diffSaves(blob, a.save());
    EXPECT_EQ(d2.total, 0u);
    applyTerrainEdit(a, 0, brush(TerrainBrush::Add, 260, 100, 3, 1));
    const SaveDiff d3 = diffSaves(blob, a.save());
    bool samples      = false;
    for (const SaveDifference& x : d3.differences)
        samples = samples || (x.kind == "terrain" && x.field.find("samples differ, first") != std::string::npos);
    EXPECT_TRUE(samples);
}

// Regression (PR review of S22): diffSaves compared N (the base spacing) only of chunks with deltas on both sides, so
// two saves of different spacings with delta-free chunks reported no terrain difference
TEST(TerrainEditSave, DiffReportsTheLayoutOfDeltaFreeChunks)
{
    World a(editSetup(1)), b(editSetup(2));
    a.terrainEditsForUpdate().bumpSequence(chunk(3, 3));
    b.terrainEditsForUpdate().bumpSequence(chunk(3, 3));
    const SaveDiff d = diffSaves(a.save(), b.save());
    ASSERT_TRUE(d.error.empty()) << d.error;
    bool layout = false;
    for (const SaveDifference& x : d.differences)
        layout = layout || (x.kind == "terrain" && x.a == json11::Json("N 257") && x.b == json11::Json("N 129"));
    EXPECT_TRUE(layout);
}

TEST(TerrainEditSave, SpaceWithoutTerrainRoundTrip)
{
    WorldSetup s;
    s.profiling = false;
    World a(s);
    a.editTerrain(0, brush(TerrainBrush::Subtract, 300, 300, 10, 2));
    World b(s);
    ASSERT_TRUE(b.tryLoad(a.save()).ok());
    EXPECT_EQ(b.terrainHeight(0, 300, 300), -2.0);
    EXPECT_EQ(b.hash(), a.hash());
}

TEST(TerrainEditSave, TamperedSectionsAreRefused)
{
    World a(editSetup(2));
    applyTerrainEdit(a, 0, brush(TerrainBrush::Add, 600, 600, 10, 1));
    const std::vector<uint8_t> blob = a.save();
    const int N                     = 129;
    std::vector<int32_t> good(size_t(N) * N, 0);
    good[size_t(50 * N + 50)]    = 77;
    std::vector<int32_t> border  = good;
    border[size_t(50 * N + 128)] = 5;  // a sample shared with chunk (x + 1, z)
    std::vector<int32_t> zero(size_t(N) * N, 0);
    std::vector<int32_t> tooBig = good;
    tooBig[3]                   = kMaxTerrainDelta + 1;
    std::vector<uint8_t> tooBigEncoded;
    {
        // The codec asserts on encode: write the varints by hand, then zstd
        std::vector<uint8_t> raw;
        int64_t prev = 0;
        for (int32_t v : tooBig)
        {
            writeVarint(raw, zigzag(int64_t(v) - prev));
            prev = v;
        }
        tooBigEncoded.resize(ZSTD_compressBound(raw.size()));
        tooBigEncoded.resize(ZSTD_compress(tooBigEncoded.data(), tooBigEncoded.size(), raw.data(), raw.size(), 3));
    }
    std::vector<uint8_t> badZstd = {1, 2, 3, 4, 5};
    struct Case
    {
        const char* name;
        std::vector<uint8_t> payload;
    };
    const std::vector<Case> cases = {
        {"wrong N", terrPayload(2, 257, {{1, 1, 1, 1, good, {}}})},
        {"wrong spacing", terrPayload(1, 257, {{1, 1, 1, 1, std::vector<int32_t>(257 * 257, 1), {}}})},
        {"out of range", terrPayload(2, 129, {{16, 1, 1, 1, good, {}}})},
        {"negative chunk", terrPayload(2, 129, {{-1, 1, 1, 1, good, {}}})},
        {"unsorted", terrPayload(2, 129, {{2, 1, 1, 1, good, {}}, {1, 1, 1, 1, good, {}}})},
        {"duplicate", terrPayload(2, 129, {{1, 1, 1, 1, good, {}}, {1, 1, 1, 1, good, {}}})},
        {"sequence 0", terrPayload(2, 129, {{1, 1, 0, 1, good, {}}})},
        {"unknown flags", terrPayload(2, 129, {{1, 1, 1, 3, good, {}}})},
        {"all zero", terrPayload(2, 129, {{1, 1, 1, 1, zero, {}}})},
        {"bad zstd", terrPayload(2, 129, {{1, 1, 1, 1, good, badZstd}})},
        {"delta beyond 2^24", terrPayload(2, 129, {{1, 1, 1, 1, good, tooBigEncoded}})},
        {"border without neighbour", terrPayload(2, 129, {{1, 1, 1, 1, border, {}}})},
        {"border without the neighbour's deltas",
         terrPayload(2, 129, {{1, 1, 1, 1, border, {}}, {2, 1, 1, 0, {}, {}}})},
        {"unequal border", terrPayload(2, 129, {{1, 1, 1, 1, border, {}}, {2, 1, 1, 1, good, {}}})},
        {"unknown space", terrPayload(2, 129, {{1, 1, 1, 1, good, {}}}, 3)},
    };
    World b(editSetup(2));
    applyTerrainEdit(b, 0, brush(TerrainBrush::Smooth, 1000, 1000, 10, 1));
    applyTerrainEdit(b, 0, brush(TerrainBrush::Add, 1000, 1000, 10, 1));
    const uint64_t before                 = b.hash();
    const std::vector<uint8_t> beforeSave = b.save();
    for (const Case& c : cases)
    {
        const LoadResult r = b.tryLoad(withTerr(blob, c.payload));
        EXPECT_FALSE(r.committed) << c.name;
        EXPECT_NE(r.error.find("TERR"), std::string::npos) << c.name << ": " << r.error;
        EXPECT_EQ(b.hash(), before) << c.name;
    }
    EXPECT_EQ(b.save(), beforeSave);
    // The control: an equal border loads
    std::vector<int32_t> right = good;
    right[size_t(50 * N + 0)]  = 5;
    const LoadResult ok =
        b.tryLoad(withTerr(blob, terrPayload(2, 129, {{1, 1, 1, 1, border, {}}, {2, 1, 2, 1, right, {}}})));
    EXPECT_TRUE(ok.ok()) << ok.error;
    EXPECT_EQ(b.terrainEdits().delta(0, 256, 128 + 50), 5);
    EXPECT_EQ(b.terrainEdits().sequence(chunk(2, 1)), 2u);
}

TEST_F(TerrainEditWorld, HostLoadClearsTheListsAndEmitsNothing)
{
    World a(hookedSetup(2));
    a.editTerrain(0, brush(TerrainBrush::Add, 400, 400, 8, 1));  // one pending event, saved with the world
    const std::vector<uint8_t> blob = a.save();
    World b(hookedSetup(2));
    bool thrown     = false;
    g_hooks.phase3  = [](World& w) { w.editTerrain(0, brush(TerrainBrush::Add, 100, 100, 8, 1)); };
    g_hooks.phase4a = [&](World&)
    {
        if (!thrown)
        {
            thrown = true;
            throw std::runtime_error("abort");
        }
    };
    EXPECT_THROW(b.step(), std::runtime_error);
    EXPECT_EQ(b.pendingTerrainChanges().size(), 1u);
    EXPECT_EQ(b.replication().hostStateList().size(), 1u);
    ASSERT_TRUE(b.tryLoad(blob).ok());
    EXPECT_TRUE(b.pendingTerrainChanges().empty());
    EXPECT_TRUE(b.replication().hostStateList().empty());
    EXPECT_EQ(pendingEvents(b), 1u);  // the saved one, nothing added by the load
    EXPECT_EQ(b.hash(), a.hash());
    EXPECT_EQ(b.terrainEdits().sequence(chunk(0, 0)), 0u);
}

// ---- world API and MCP tools (step 7)

namespace
{
using Json = json11::Json;
Json callTool(McpServer& server, const std::string& name, const Json& args)
{
    return server.handleMessage(McpRequest("tools/call", Json::object{{"name", name}, {"arguments", args}}));
}
Json toolJson(const Json& response)
{
    std::string err;
    return Json::parse(response["result"]["content"][0]["text"].string_value(), err);
}
std::string toolError(const Json& response)
{
    return response["result"]["content"][0]["text"].string_value();
}
}  // namespace

TEST_F(TerrainEditWorld, ToolsThroughHandleMessage)
{
    WorldSetup s = hookedSetup(1);
    s.config.spaceTemplates.push_back(SpaceTemplate::enclosed("cellar"));
    World w(s);
    McpServer server("test");
    auto api = registerWorldTools(server, w);
    Poller poller(server);
    // edit_terrain on a chunk corner: four parts with chunk-local boxes, the box in metres
    Json r =
        callTool(server, "edit_terrain",
                 Json::object{{"op", "add"}, {"center", Json::array{256, 0, 256}}, {"radius", 8}, {"strength", 1}});
    ASSERT_FALSE(r["result"]["isError"].bool_value()) << r.dump();
    Json e = toolJson(r);
    EXPECT_GT(e["changed_samples"].number_value(), 100);
    ASSERT_EQ(e["chunks"].array_items().size(), 4u);
    EXPECT_EQ(e["chunks"][0]["chunk"], Json(Json::array{0, 0}));
    EXPECT_EQ(e["chunks"][0]["sequence"].number_value(), 1);
    EXPECT_EQ(e["chunks"][0]["box"], Json(Json::array{249, 249, 8, 8}));
    EXPECT_EQ(e["chunks"][3]["box"], Json(Json::array{0, 0, 8, 8}));
    EXPECT_EQ(e["box"], Json(Json::array{249, 249, 263, 263}));
    // Logged under api
    bool logged = false;
    for (const LogRing::Line& l : engineLog().since(0))
        logged = logged || (l.category == "api" && l.message.find("edit_terrain") != std::string::npos);
    EXPECT_TRUE(logged);
    // Nothing changed: empty chunks
    r = callTool(server, "edit_terrain",
                 Json::object{{"op", "smooth"}, {"center", Json::array{3000, 3000}}, {"radius", 4}, {"strength", 1}});
    ASSERT_FALSE(r["result"]["isError"].bool_value()) << r.dump();
    EXPECT_EQ(toolJson(r)["changed_samples"].number_value(), 0);
    EXPECT_TRUE(toolJson(r)["chunks"].array_items().empty());
    // terrain_height: the exact value and the edited flag
    Json h = toolJson(callTool(server, "terrain_height", Json::object{{"x", 256}, {"z", 256}}));
    EXPECT_EQ(h["height"].number_value(), w.terrainHeight(0, 256, 256));
    EXPECT_TRUE(h["edited"].bool_value());
    EXPECT_EQ(h["samples"][0], Json(Json::array{256, 256, double(w.terrainSample(0, 256, 256))}));
    Json f = toolJson(callTool(server, "terrain_height", Json::object{{"x", 1000.5}, {"z", 1000.25}}));
    EXPECT_FALSE(f["edited"].bool_value());
    EXPECT_EQ(f["height"].number_value(), w.terrainHeight(0, 1000.5, 1000.25));
    // terrain_edits: paged, has_deltas
    w.terrainEditsForUpdate().bumpSequence(chunk(9, 9));
    Json l = toolJson(callTool(server, "terrain_edits", Json::object{{"limit", 3}}));
    ASSERT_EQ(l["chunks"].array_items().size(), 3u);
    EXPECT_EQ(l["total"].number_value(), 5);
    EXPECT_EQ(l["next"].number_value(), 3);
    EXPECT_TRUE(l["chunks"][0]["has_deltas"].bool_value());
    Json l2 = toolJson(callTool(server, "terrain_edits", Json::object{{"limit", 3}, {"after", 3}}));
    ASSERT_EQ(l2["chunks"].array_items().size(), 2u);
    EXPECT_TRUE(l2["next"].is_null());
    EXPECT_EQ(l2["chunks"][1]["chunk"], Json(Json::array{9, 9}));
    EXPECT_FALSE(l2["chunks"][1]["has_deltas"].bool_value());
    EXPECT_EQ(l2["chunks"][1]["sequence"].number_value(), 1);
    // Errors: an invalid brush (field and bound), a center out of the space, an enclosed space, an unknown op
    auto expectError = [&](const Json& args, const std::string& part)
    {
        const Json x = callTool(server, "edit_terrain", args);
        EXPECT_TRUE(x["result"]["isError"].bool_value()) << args.dump();
        EXPECT_NE(toolError(x).find(part), std::string::npos) << toolError(x);
    };
    expectError(Json::object{{"op", "add"}, {"center", Json::array{1, 0, 1}}, {"radius", 200}, {"strength", 1}},
                "radius");
    expectError(Json::object{{"op", "flatten"}, {"center", Json::array{1, 0, 1}}, {"radius", 2}, {"strength", 2}},
                "strength");
    expectError(Json::object{{"op", "add"}, {"center", Json::array{9000, 0, 1}}, {"radius", 2}, {"strength", 1}},
                "center.x");
    expectError(
        Json::object{{"space", 1}, {"op", "add"}, {"center", Json::array{1, 0, 1}}, {"radius", 2}, {"strength", 1}},
        "enclosed");
    expectError(Json::object{{"op", "raise"}, {"center", Json::array{1, 0, 1}}, {"radius", 2}, {"strength", 1}}, "op");
    // Inside a step: refused (the API is called from a system)
    std::string inStep;
    g_hooks.phase4a = [&](World& world)
    {
        inStep =
            api->call("edit_terrain",
                      Json::object{{"op", "add"}, {"center", Json::array{1, 0, 1}}, {"radius", 2}, {"strength", 1}})
                .error;
    };
    w.step();
    g_hooks.phase4a = nullptr;
    EXPECT_NE(inStep.find("between steps"), std::string::npos) << inStep;
    // Read-only registration has no edit_terrain
    McpServer observer("observer");
    auto readOnly = registerWorldTools(observer, w, WorldToolsOptions{false});
    Json list     = observer.handleMessage(McpRequest("tools/list", Json::object{}));
    std::set<std::string> names;
    for (auto& t : list["result"]["tools"].array_items()) names.insert(t["name"].string_value());
    EXPECT_FALSE(names.count("edit_terrain"));
    EXPECT_TRUE(names.count("terrain_height"));
    EXPECT_TRUE(names.count("terrain_edits"));
}

// Regression (PR review of S22): apply took "op" as the tool name and removed it, so edit_terrain (whose brush argument
// is op too) could not run in a batch; the envelope {op, args} carries the arguments
TEST_F(TerrainEditWorld, EditTerrainInsideApply)
{
    World w(hookedSetup(2));
    McpServer server("test");
    auto api = registerWorldTools(server, w);
    Poller poller(server);
    const Json edit = Json::object{
        {"op", "edit_terrain"},
        {"args", Json::object{{"op", "add"}, {"center", Json::array{300, 0, 300}}, {"radius", 6}, {"strength", 1}}}};
    const Json height = Json::object{{"op", "terrain_height"}, {"x", 300}, {"z", 300}};
    Json r            = callTool(server, "apply", Json::object{{"operations", Json::array{edit, height}}});
    ASSERT_FALSE(r["result"]["isError"].bool_value()) << r.dump();
    const Json results = toolJson(r)["results"];
    EXPECT_GT(results[0]["changed_samples"].number_value(), 0);
    EXPECT_TRUE(results[1]["edited"].bool_value());
    EXPECT_EQ(w.terrainEdits().sequence(chunk(1, 1)), 1u);
    // args with other arguments beside it: refused before anything is applied
    const Json mixed = Json::object{{"op", "terrain_height"}, {"x", 1}, {"args", Json::object{{"z", 1}}}};
    r                = callTool(server, "apply", Json::object{{"operations", Json::array{edit, mixed}}});
    EXPECT_TRUE(r["result"]["isError"].bool_value()) << r.dump();
    EXPECT_NE(toolError(r).find("args"), std::string::npos) << toolError(r);
    EXPECT_EQ(w.terrainEdits().sequence(chunk(1, 1)), 1u);
}

TEST_F(TerrainEditWorld, ToolsOnAClientAnswerWithAnError)
{
    auto net     = LoopbackTransport::createNetwork(2);
    WorldSetup c = hookedSetup(2);
    c.transport  = net[1].get();
    c.role       = NetRole::Client;
    c.localPeer  = 1;
    World w(c);
    WorldApi api(w);
    const ApiResult r =
        api.call("edit_terrain",
                 Json::object{{"op", "add"}, {"center", Json::array{10, 0, 10}}, {"radius", 2}, {"strength", 1}});
    EXPECT_FALSE(r.ok);
    EXPECT_NE(r.error.find("client"), std::string::npos) << r.error;
    EXPECT_TRUE(api.call("terrain_height", Json::object{{"x", 1}, {"z", 1}}).ok);
}

// ---- the visual path (step 8)

// Regression (PR review of S22): a merge that grew a box checked only the boxes after it, so an earlier box inside the
// grown one stayed separate (its samples were patched twice)
TEST(TerrainDirtyQueue, GrownBoxAbsorbsEarlierBoxes)
{
    auto rect = [](int64_t i0, int64_t k0, int64_t i1, int64_t k1)
    {
        SampleRect r;
        r.i0 = i0;
        r.k0 = k0;
        r.i1 = i1;
        r.k1 = k1;
        return r;
    };
    TerrainDirtyQueue q;
    q.addTileBox(rect(0, 0, 0, 0));
    q.addTileBox(rect(2, 0, 2, 2));
    ASSERT_EQ(q.tileBoxes().size(), 2u);
    q.addTileBox(rect(0, 2, 1, 2));
    ASSERT_EQ(q.tileBoxes().size(), 1u);
    EXPECT_EQ(q.tileBoxes()[0].i0, 0);
    EXPECT_EQ(q.tileBoxes()[0].k0, 0);
    EXPECT_EQ(q.tileBoxes()[0].i1, 2);
    EXPECT_EQ(q.tileBoxes()[0].k1, 2);
    // Isolated boxes stay separate, in insertion order
    q.addTileBox(rect(10, 10, 11, 11));
    q.addTileBox(rect(20, 0, 20, 0));
    ASSERT_EQ(q.tileBoxes().size(), 3u);
    EXPECT_EQ(q.tileBoxes()[1].i0, 10);
    EXPECT_EQ(q.tileBoxes()[2].i0, 20);
}

namespace
{
// The merged heights a tile of the sink must hold: terrainSample at its coincident samples
std::vector<int32_t> expectedTile(const World& w, const RegionId& r)
{
    const int64_t S = 256 / w.spaceTemplate(r.space).baseSpacing, N = S + 1, span = int64_t(1) << r.level;
    std::vector<int32_t> out(size_t(N * N));
    for (int64_t b = 0; b < N; ++b)
        for (int64_t a = 0; a < N; ++a)
            out[size_t(b * N + a)] = w.terrainSample(r.space, r.x * span * S + a * span, r.z * span * S + b * span);
    return out;
}
std::vector<int32_t> expectedPyramid(const World& w, SpaceId space)
{
    const int64_t P    = 8 * int64_t(w.spaceTemplate(space).sizeChunks) + 1,
                  step = 32 / w.spaceTemplate(space).baseSpacing;
    std::vector<int32_t> out(size_t(P * P));
    for (int64_t b = 0; b < P; ++b)
        for (int64_t a = 0; a < P; ++a) out[size_t(b * P + a)] = w.terrainSample(space, a * step, b * step);
    return out;
}
// Every tile and the pyramid of the sink equal the merged heights; min / max cover the heights
void expectSinkMerged(const World& w, const RecordingRenderSink& sink)
{
    ASSERT_TRUE(sink.hasTerrainSet());
    for (const auto& [r, t] : sink.terrainTiles())
    {
        EXPECT_EQ(*t.heights, expectedTile(w, r)) << "L" << int(r.level) << " " << r.x << " " << r.z;
        EXPECT_LE(t.minHeight, *std::min_element(t.heights->begin(), t.heights->end()));
        EXPECT_GE(t.maxHeight, *std::max_element(t.heights->begin(), t.heights->end()));
    }
    EXPECT_EQ(*sink.terrainPyramid()->heights, expectedPyramid(w, sink.terrainSpace()));
}
std::map<RegionId, std::vector<int32_t>> storeTiles(const World& w, SpaceId space = 0)
{
    std::map<RegionId, std::vector<int32_t>> out;
    for (const auto& [r, t] : w.terrain().store(space).tiles()) out[r] = *t.data.heights;
    return out;
}
const Vec3 kCamera(300, 60, 300);
}  // namespace

TEST(TerrainEditVisual, EditPatchesTheSentTilesAndThePyramid)
{
    World w(editSetup(2));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    ASSERT_TRUE(w.terrain().completeView(0, kCamera));
    const auto before                      = storeTiles(w);
    const std::vector<int32_t> pyramidBase = *w.terrain().pyramid(0)->heights;
    applyTerrainEdit(w, 0, brush(TerrainBrush::Add, 300, 300, 40, 3));
    applyTerrainEdit(w, 0, brush(TerrainBrush::Subtract, 256, 260, 20, 2));  // across chunk borders
    EXPECT_GT(w.terrain().dirtyQueue().size(), 0u);
    const auto counts = sink.counters();
    const uint64_t h0 = w.terrain().baseHookSamples();
    w.terrain().updateView(0, kCamera);
    // No hook on the main thread: the bases come from the store's own tiles and pyramid
    EXPECT_EQ(w.terrain().baseHookSamples(), h0);
    EXPECT_GT(sink.counters().patchTerrainTile, counts.patchTerrainTile);
    EXPECT_GT(sink.counters().patchTerrainPyramid, counts.patchTerrainPyramid);
    EXPECT_EQ(sink.counters().setTerrainTile, counts.setTerrainTile);
    EXPECT_EQ(w.terrain().dirtyQueue().size(), 0u);
    EXPECT_GT(w.terrain().stats().patchedSamples, 0u);
    expectSinkMerged(w, sink);
    // The store keeps the base (bitwise)
    EXPECT_EQ(storeTiles(w), before);
    EXPECT_EQ(*w.terrain().pyramid(0)->heights, pyramidBase);
    w.setRenderSink(nullptr);
}

TEST(TerrainEditVisual, SetThenPatchInOneFrame)
{
    // A tile resident but not sent when the edit happens (uploadsPerFrame 0): the frame that sends it merged also
    // patches the queued box over it, after the send
    World w(editSetup(2));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    ASSERT_TRUE(w.terrain().completeView(0, kCamera));
    TerrainViewSettings v = w.terrain().viewSettings();
    v.uploadsPerFrame     = 0;
    w.terrain().setViewSettings(v);
    const Vec3 far(1300, 60, 1300);
    RegionId t;
    t.level = 0;
    t.x     = 5;
    t.z     = 5;
    ASSERT_FALSE(sink.terrainTiles().count(t));
    for (int i = 0; i < 5000 && !w.terrain().store(0).tile(t); ++i)
    {
        w.terrain().updateView(0, far);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(w.terrain().store(0).tile(t));
    ASSERT_FALSE(sink.terrainTiles().count(t));
    applyTerrainEdit(w, 0, brush(TerrainBrush::Add, 5 * 256 + 100, 5 * 256 + 100, 30, 4));
    v.uploadsPerFrame = 64;
    w.terrain().setViewSettings(v);
    sink.setLogging(true);
    const uint64_t merged = w.terrain().stats().mergedSends;
    w.terrain().updateView(0, far);
    ASSERT_TRUE(sink.terrainTiles().count(t));
    const std::vector<std::string>& log = sink.log();
    const auto set                      = std::find(log.begin(), log.end(), "setTerrainTile L0 5 5");
    const auto patch                    = std::find(log.begin(), log.end(), "patchTerrainTile L0 5 5");
    ASSERT_NE(set, log.end());
    ASSERT_NE(patch, log.end());
    EXPECT_LT(set - log.begin(), patch - log.begin());
    EXPECT_GT(w.terrain().stats().mergedSends, merged);
    // Requested before the edit, sent after it: merged, exact min / max
    const TerrainTileData& got = sink.terrainTiles().at(t);
    EXPECT_EQ(*got.heights, expectedTile(w, t));
    EXPECT_EQ(got.minHeight, *std::min_element(got.heights->begin(), got.heights->end()));
    EXPECT_EQ(got.maxHeight, *std::max_element(got.heights->begin(), got.heights->end()));
    w.setRenderSink(nullptr);
}

TEST(TerrainEditVisual, NewSinkReceivesMergedData)
{
    World w(editSetup(2));
    RecordingRenderSink a;
    w.setRenderSink(&a);
    ASSERT_TRUE(w.terrain().completeView(0, kCamera));
    applyTerrainEdit(w, 0, brush(TerrainBrush::Add, 300, 300, 40, 3));
    w.setRenderSink(nullptr);
    applyTerrainEdit(w, 0, brush(TerrainBrush::Smooth, 310, 290, 30, 1));  // without a sink: nothing queued
    EXPECT_EQ(w.terrain().dirtyQueue().size(), 0u);
    RecordingRenderSink b;
    w.setRenderSink(&b);
    EXPECT_EQ(b.counters().patchTerrainTile, 0u);
    expectSinkMerged(w, b);
    w.setRenderSink(nullptr);
}

// Regression (PR review of S22): the merge at the send point added a delta once per chunk whose closed range contains
// the sample, so a coarse tile or the pyramid had 2x the delta on a chunk border and 4x on a corner
TEST(TerrainEditVisual, NewSinkReceivesMergedDataOnAChunkCorner)
{
    World w(editSetup(2));
    ASSERT_TRUE(w.terrain().completeView(0, kCamera));
    applyTerrainEdit(w, 0, brush(TerrainBrush::Add, 256, 256, 40, 3));
    ASSERT_NE(w.terrainSample(0, 128, 128), w.terrain().sample(0, 128, 128));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    bool coarse = false;
    for (const auto& kv : sink.terrainTiles()) coarse = coarse || kv.first.level > 0;
    EXPECT_TRUE(coarse);
    expectSinkMerged(w, sink);
    w.setRenderSink(nullptr);
}

// Regression (implementation review of S22): without a sink the masks of edited chunks were never made (the mask queue
// drops its entries without a sink) and completeView waited for its whole timeout; without a sink the store's base
// masks count
TEST(TerrainEditVisual, CompleteViewWithoutSinkAcceptsTheBaseMasksOfEditedChunks)
{
    World w(editSetup(2));
    applyTerrainEdit(w, 0, brush(TerrainBrush::Add, 300, 300, 40, 3));
    const auto start = std::chrono::steady_clock::now();
    EXPECT_TRUE(w.terrain().completeView(0, kCamera, 60));
    EXPECT_LT(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), 30.0);
    EXPECT_GT(w.terrain().stats().masksDesired, 0);
    EXPECT_TRUE(w.terrain().viewComplete(0, kCamera));
    ChunkId edited;
    edited.x = 1;
    edited.z = 1;
    EXPECT_NE(w.terrain().store(0).mask(edited), nullptr);
}

TEST(TerrainEditVisual, BudgetSpreadsALargeEditByRows)
{
    World w(editSetup(2));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    ASSERT_TRUE(w.terrain().completeView(0, kCamera));
    TerrainViewSettings v = w.terrain().viewSettings();
    v.editSamplesPerFrame = 1000;
    w.terrain().setViewSettings(v);
    applyTerrainEdit(w, 0, brush(TerrainBrush::Add, 300, 300, 100, 5));
    const uint64_t before = w.terrain().stats().patchedSamples;
    int frames            = 0;
    while (w.terrain().dirtyQueue().size() > 0 && frames < 1000)
    {
        w.terrain().updateView(0, kCamera);
        EXPECT_LE(w.terrain().stats().patchedSamplesLastUpdate, 1000u);
        frames++;
    }
    EXPECT_GT(frames, 5);
    expectSinkMerged(w, sink);
    // Every sample once: the coincident samples of the box in the sent tiles and the pyramid
    const int64_t i0 = 100, i1 = 200;  // the box at 2 m: (300 +- 100) / 2, strictly inside
    int64_t total = 0;
    for (const auto& [r, t] : sink.terrainTiles())
    {
        const int64_t span = int64_t(1) << r.level, ti0 = r.x * span * 128, tk0 = r.z * span * 128;
        auto count = [&](int64_t lo, int64_t hi, int64_t t0)
        {
            const int64_t a = std::max(lo, t0), b = std::min(hi, t0 + span * 128);
            return a > b ? int64_t(0) : (b - t0) / span - (a - t0 + span - 1) / span + 1;
        };
        total += std::max<int64_t>(0, count(i0 + 1, i1 - 1, ti0)) * std::max<int64_t>(0, count(i0 + 1, i1 - 1, tk0));
    }
    const int64_t py = (199 / 16) - (101 + 15) / 16 + 1;
    total += py * py;
    EXPECT_EQ(int64_t(w.terrain().stats().patchedSamples - before), total);
    // completeView drains the whole queue at once
    applyTerrainEdit(w, 0, brush(TerrainBrush::Subtract, 320, 280, 100, 5));
    ASSERT_TRUE(w.terrain().completeView(0, kCamera));
    EXPECT_EQ(w.terrain().dirtyQueue().size(), 0u);
    expectSinkMerged(w, sink);
    w.setRenderSink(nullptr);
}

TEST(TerrainEditVisual, EditInAnotherSpaceQueuesOnlyPyramidBoxes)
{
    WorldSetup s         = editSetup(2);
    SpaceTemplate second = terraformTestTemplate(2);
    second.name          = "second";
    s.config.spaceTemplates.push_back(second);
    World w(s);
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    ASSERT_TRUE(w.terrain().completeView(0, kCamera));
    applyTerrainEdit(w, 1, brush(TerrainBrush::Add, 300, 300, 40, 3));
    EXPECT_TRUE(w.terrain().dirtyQueue().tileBoxes().empty());
    EXPECT_EQ(w.terrain().dirtyQueue().pyramidBoxCount(1), 1u);
    w.terrain().updateView(0, kCamera);
    EXPECT_EQ(w.terrain().dirtyQueue().pyramidBoxCount(1), 1u);  // kept for its space
    // The switch sends space 1's set merged
    ASSERT_TRUE(w.terrain().completeView(1, kCamera));
    EXPECT_EQ(sink.terrainSpace(), 1);
    expectSinkMerged(w, sink);
    // An edit in the view space after the switch: tile boxes
    applyTerrainEdit(w, 1, brush(TerrainBrush::Add, 330, 300, 10, 1));
    EXPECT_FALSE(w.terrain().dirtyQueue().tileBoxes().empty());
    w.terrain().updateView(1, kCamera);
    expectSinkMerged(w, sink);
    w.setRenderSink(nullptr);
}

TEST(TerrainEditVisual, HostLoadRepatchesTheEditedChunks)
{
    World w(editSetup(2));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    ASSERT_TRUE(w.terrain().completeView(0, kCamera));
    applyTerrainEdit(w, 0, brush(TerrainBrush::Add, 300, 300, 30, 3));
    const std::vector<uint8_t> blob = w.save();
    applyTerrainEdit(w, 0, brush(TerrainBrush::Subtract, 600, 200, 30, 3));
    applyTerrainEdit(w, 0, brush(TerrainBrush::Smooth, 300, 300, 30, 1));
    w.terrain().updateView(0, kCamera);
    const uint64_t evicted = w.terrain().stats().evicted;
    ASSERT_TRUE(w.tryLoad(blob).ok());
    EXPECT_GT(w.terrain().dirtyQueue().size(), 0u);
    w.terrain().updateView(0, kCamera);
    EXPECT_EQ(w.terrain().stats().evicted, evicted);
    expectSinkMerged(w, sink);
    w.setRenderSink(nullptr);
}

// ---- the edited grass masks (step 9)

namespace
{
RegionId level0(int32_t x, int32_t z)
{
    RegionId r;
    r.level = 0;
    r.x     = x;
    r.z     = z;
    return r;
}
// The mask of the merged level-0 heights (the default thresholds, no sea level)
std::vector<uint8_t> expectedMask(const World& w, const ChunkId& c)
{
    TerrainTileData t;
    auto h                       = std::make_shared<std::vector<int32_t>>(expectedTile(w, level0(c.x, c.z)));
    t.heights                    = h;
    t.minHeight                  = *std::min_element(h->begin(), h->end());
    t.maxHeight                  = *std::max_element(h->begin(), h->end());
    const TerrainViewSettings& v = w.terrain().viewSettings();
    return *generateGrassMask(t, w.terrain().layout(c.space), w.spaceTemplate(c.space).seaLevel, v.grassMinNormalY,
                              v.grassMinAboveSea)
                .density;
}
// updateView until the edits' visual work is done (patches, mask jobs, requests)
void settle(World& w, const Vec3& camera)
{
    for (int i = 0; i < 5000; ++i)
    {
        w.terrain().updateView(0, camera);
        const TerrainSystem::Stats& s = w.terrain().stats();
        if (s.maskJobsInFlight == 0 && s.maskQueue == 0 && s.pending == 0 && w.terrain().dirtyQueue().size() == 0)
            return;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ADD_FAILURE() << "the terrain did not settle";
}
}  // namespace

TEST(TerrainEditGrass, EditedMasksFromMergedHeights)
{
    World w(editSetup(2));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    ASSERT_TRUE(w.terrain().completeView(0, kCamera));
    const ChunkId c = chunk(1, 1);
    ASSERT_TRUE(sink.grassMasks().count(c));
    const std::vector<uint8_t> base = *sink.grassMasks().at(c).density;
    // A steep crater on the chunk corner: four edited chunks, at most editMasksPerFrame submissions per frame
    applyTerrainEdit(w, 0, brush(TerrainBrush::Subtract, 256, 256, 10, 8));
    applyTerrainEdit(w, 0, brush(TerrainBrush::Subtract, 300, 300, 8, 6));
    int maxJobs = 0;
    for (int i = 0; i < 5000; ++i)
    {
        w.terrain().updateView(0, kCamera);
        maxJobs                       = std::max(maxJobs, w.terrain().stats().maskJobsLastUpdate);
        const TerrainSystem::Stats& s = w.terrain().stats();
        if (s.maskJobsInFlight == 0 && s.maskQueue == 0 && w.terrain().dirtyQueue().size() == 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_GE(maxJobs, 1);
    EXPECT_LE(maxJobs, 2);
    EXPECT_GE(w.terrain().stats().maskRecomputes, 4u);
    for (const ChunkId& e : {chunk(0, 0), chunk(1, 0), chunk(0, 1), chunk(1, 1)})
    {
        ASSERT_TRUE(sink.grassMasks().count(e));
        EXPECT_EQ(*sink.grassMasks().at(e).density, expectedMask(w, e)) << e.x << ", " << e.z;
        EXPECT_TRUE(w.terrain().visualPatch().masks.count(e));
    }
    // The crater wall lost grass
    const std::vector<uint8_t>& now = *sink.grassMasks().at(c).density;
    int lost                        = 0;
    for (size_t i = 0; i < now.size(); ++i) lost += (base[i] > 0 && now[i] == 0) ? 1 : 0;
    EXPECT_GT(lost, 10);
    // The store keeps base masks only
    EXPECT_EQ(*w.terrain().store(0).mask(c)->density, base);
    // A new sink receives the cached edited masks
    w.setRenderSink(nullptr);
    RecordingRenderSink b;
    w.setRenderSink(&b);
    ASSERT_TRUE(b.grassMasks().count(c));
    EXPECT_EQ(*b.grassMasks().at(c).density, now);
    // ... and after another edit (stale cache) nothing for that chunk until its job completed: never the base mask
    w.setRenderSink(nullptr);
    applyTerrainEdit(w, 0, brush(TerrainBrush::Add, 330, 330, 5, 3));
    RecordingRenderSink d;
    w.setRenderSink(&d);
    EXPECT_FALSE(d.grassMasks().count(c));
    // completeView waits for the mask jobs
    ASSERT_TRUE(w.terrain().completeView(0, kCamera));
    ASSERT_TRUE(d.grassMasks().count(c));
    EXPECT_EQ(*d.grassMasks().at(c).density, expectedMask(w, c));
    w.setRenderSink(nullptr);
}

TEST(TerrainEditGrass, ReturnToZeroSendsTheBaseMask)
{
    World w(editSetup(2));
    RecordingRenderSink sink;
    w.setRenderSink(&sink);
    ASSERT_TRUE(w.terrain().completeView(0, kCamera));
    const ChunkId c                 = chunk(1, 1);
    const std::vector<uint8_t> base = *sink.grassMasks().at(c).density;
    applyTerrainEdit(w, 0, brush(TerrainBrush::Subtract, 300, 300, 8, 6));
    settle(w, kCamera);
    EXPECT_NE(*sink.grassMasks().at(c).density, base);
    applyTerrainEdit(w, 0, brush(TerrainBrush::Add, 300, 300, 8, 6));
    EXPECT_FALSE(w.terrainEdits().hasDeltas(c));
    settle(w, kCamera);
    EXPECT_FALSE(w.terrain().visualPatch().masks.count(c));
    EXPECT_EQ(*sink.grassMasks().at(c).density, base);
    w.setRenderSink(nullptr);
}

namespace
{
// A world on the gated hook with one pool worker: armed, the worker blocks in the first hook call of a tile job, so
// a mask job submitted after it waits in the queue
struct GatedGrass
{
    std::unique_ptr<World> w;
    RecordingRenderSink sink;
    GatedGrass()
    {
        WorldSetup s                               = editSetup(2);
        s.config.spaceTemplates[0].prototypeHeight = latchedHook;
        w                                          = std::make_unique<World>(s);
        TerrainViewSettings v                      = w->terrain().viewSettings();
        v.workers                                  = 1;
        w->terrain().setViewSettings(v);
        w->setRenderSink(&sink);
        EXPECT_TRUE(w->terrain().completeView(0, kCamera));
        // Block the worker: more tiles wanted
        armLatch();
        v.residencyRange0 = 1000;
        w->terrain().setViewSettings(v);
        w->terrain().updateView(0, kCamera);
        for (int i = 0; i < 2000 && latchState().callsBlocked == 0; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        EXPECT_GT(latchState().callsBlocked.load(), 0);
    }
    ~GatedGrass()
    {
        openLatch();
        w->setRenderSink(nullptr);
        w.reset();
    }
    // An edit of chunk (1, 1) and the update that submits its mask job (waiting behind the blocked tile job)
    void submit()
    {
        applyTerrainEdit(*w, 0, brush(TerrainBrush::Subtract, 300, 300, 8, 6));
        for (int i = 0; i < 4 && !jobFor(chunk(1, 1)); ++i) w->terrain().updateView(0, kCamera);
        EXPECT_TRUE(jobFor(chunk(1, 1)));
    }
    bool jobFor(const ChunkId&) const { return w->terrain().stats().maskJobsInFlight > 0; }
};
}  // namespace

TEST(TerrainEditGrass, ASecondEditDropsTheFirstResult)
{
    GatedGrass g;
    g.submit();
    const uint64_t dropped = g.w->terrain().stats().maskResultsDropped;
    applyTerrainEdit(*g.w, 0, brush(TerrainBrush::Add, 304, 300, 4, 2));
    openLatch();
    settle(*g.w, kCamera);
    EXPECT_GT(g.w->terrain().stats().maskResultsDropped, dropped);
    EXPECT_EQ(*g.sink.grassMasks().at(chunk(1, 1)).density, expectedMask(*g.w, chunk(1, 1)));
}

TEST(TerrainEditGrass, ABumpedSequenceKeepsTheResult)
{
    // F1: the masks key on the terrain revision, not on the sequence (S23b's tombstones bump it)
    GatedGrass g;
    g.submit();
    const uint64_t dropped = g.w->terrain().stats().maskResultsDropped;
    const uint64_t done    = g.w->terrain().stats().maskRecomputes;
    g.w->terrainEditsForUpdate().bumpSequence(chunk(1, 1));
    openLatch();
    settle(*g.w, kCamera);
    EXPECT_EQ(g.w->terrain().stats().maskResultsDropped, dropped);
    EXPECT_GT(g.w->terrain().stats().maskRecomputes, done);
    EXPECT_EQ(*g.sink.grassMasks().at(chunk(1, 1)).density, expectedMask(*g.w, chunk(1, 1)));
}

TEST(TerrainEditGrass, AResultForAChunkThatLeftIsDiscarded)
{
    GatedGrass g;
    g.submit();
    const uint64_t dropped = g.w->terrain().stats().maskResultsDropped;
    const uint64_t done    = g.w->terrain().stats().maskRecomputes;
    // The camera moves away: chunk (1, 1) leaves the grass radius
    const Vec3 far(2000, 60, 2000);
    g.w->terrain().updateView(0, far);
    openLatch();
    settle(*g.w, far);
    EXPECT_EQ(g.w->terrain().stats().maskResultsDropped, dropped);
    EXPECT_EQ(g.w->terrain().stats().maskRecomputes, done);
    EXPECT_FALSE(g.w->terrain().visualPatch().masks.count(chunk(1, 1)));
    EXPECT_FALSE(g.sink.grassMasks().count(chunk(1, 1)));
    EXPECT_EQ(g.w->terrain().stats().maskQueue, 0);
}
