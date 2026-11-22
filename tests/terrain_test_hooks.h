/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Shared terrain test helpers (S20; S22 and S23 add theirs): cheap integer height hooks, the gated latchedHook, and the
// engine-only templates (inside SAIGA_TEST_ENGINE, which saiga_test defines for the tests that link saiga_engine).

#pragma once

#include "saiga/core/math/TerrainSurface.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/scene/TerrainTiles.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include <condition_variable>

#ifdef SAIGA_TEST_ENGINE
#    include "saiga/engine/world/SpaceTemplate.h"
#endif

namespace TerrainTestHooks
{
using namespace Saiga;

// pi (M_PI is not standard C++: MSVC has it only with _USE_MATH_DEFINES)
constexpr double kPi = 3.14159265358979323846;

// A triangle wave of period p (p even) with amplitude p / 2: |((v mod p) - p / 2)|
inline int64_t triangleWave(int64_t v, int64_t p)
{
    int64_t m = v % p;
    if (m < 0) m += p;
    return m < p / 2 ? m : p - m;
}

// y = 0 everywhere
inline int32_t flatHook(uint64_t, int64_t, int64_t)
{
    return 0;
}

// A slope along x (1/64) with ridges every 40 m along x + z (4 m high), plus a seed offset: the walker scenario
inline int32_t slopeRidgeHook(uint64_t seed, int64_t x, int64_t z)
{
    const int64_t h = x * 16 + triangleWave(x + z, 40) * 205 + int64_t(seed % 7) * 1024;
    return int32_t(h);
}

// Periodic in 4096 m in x and z (so periodic in 28 672 m): hills up to about 60 m and a valley below the sea level 6 m
// along x = 2000 (mod 4096; the origin-offset test's terrain + water view)
inline int32_t periodic4096Hook(uint64_t, int64_t x, int64_t z)
{
    const int64_t a = triangleWave(x + 2096, 4096) * 20;  // 0..40960 units (40 m), 0 at x = 2000
    const int64_t b = triangleWave(z, 1024) * 30;         // 0..15360 (15 m)
    const int64_t c = triangleWave(x + 2 * z, 256) * 24;  // 0..3072 (3 m)
    const int64_t d = triangleWave(3 * x - z, 64) * 40;   // 0..1280 (1.25 m)
    return int32_t(a + b + c + d - 8 * 1024);
}

// High frequency: every sample differs by up to 6 m from its neighbours (the crack test: coarser levels subsample it)
inline int32_t highFrequencyHook(uint64_t, int64_t x, int64_t z)
{
    uint64_t h = uint64_t(x) * 0x9E3779B97F4A7C15ull ^ uint64_t(z) * 0xC2B2AE3D27D4EB4Full;
    h ^= h >> 29;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 32;
    return int32_t(h % 6144) + 20 * 1024;
}

// Spikes: a small table of (x, z, height); 0 elsewhere
struct Spike
{
    int64_t x = 0, z = 0;
    int32_t height = 0;
};
inline std::vector<Spike>& spikeTable()
{
    static std::vector<Spike> table;
    return table;
}
inline int32_t spikeHook(uint64_t, int64_t x, int64_t z)
{
    for (const Spike& s : spikeTable())
        if (s.x == x && s.z == z) return s.height;
    return 0;
}

// A cheap plane with a seed-dependent offset (S22's render and edit tests through terraformTestTemplate)
inline int32_t integerTestHook(uint64_t seed, int64_t x, int64_t z)
{
    return int32_t(int64_t(seed % 1000) * 64 + x * 8 + z * 4 + 2048);
}

// ---- the gated hook (file-static state): latchedHook blocks on a latch only when called off the thread that armed it
// (the pool's workers), so the main thread's synchronous paths and its share of completeView never block. openLatch()
// before the world is destroyed. sleepAtOrigin mode: no latch, but 100 ms of sleep when called at sample (0, 0) (once
// per pyramid and per tile containing the origin). The heights are integerTestHook's.
struct LatchState
{
    std::mutex mutex;
    std::condition_variable cv;
    bool armed = false, open = true;
    // Read by the workers without the mutex
    std::atomic<bool> sleepAtOrigin{false};
    std::thread::id armingThread;
    // Calls on the arming thread, calls that blocked (or would have) on another thread
    std::atomic<int> callsOnArmingThread{0}, callsBlocked{0};
};
inline LatchState& latchState()
{
    static LatchState state;
    return state;
}
// Closes the latch for every other thread (the pool's workers)
inline void armLatch()
{
    LatchState& s = latchState();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.armed               = true;
    s.open                = false;
    s.armingThread        = std::this_thread::get_id();
    s.callsOnArmingThread = 0;
    s.callsBlocked        = 0;
}
inline void openLatch()
{
    LatchState& s = latchState();
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        s.open  = true;
        s.armed = false;
    }
    s.cv.notify_all();
}
// Arms the latch for its scope and opens it at the end (also when a failed assertion returns early: a closed latch
// would block the pool's workers and hang the world's destruction)
struct ScopedLatch
{
    ScopedLatch() { armLatch(); }
    ~ScopedLatch() { openLatch(); }
    ScopedLatch(const ScopedLatch&)            = delete;
    ScopedLatch& operator=(const ScopedLatch&) = delete;
};
inline void setSleepAtOrigin(bool on)
{
    latchState().sleepAtOrigin = on;
}
inline int32_t latchedHook(uint64_t seed, int64_t x, int64_t z)
{
    LatchState& s = latchState();
    if (s.sleepAtOrigin && x == 0 && z == 0) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    {
        std::unique_lock<std::mutex> lock(s.mutex);
        if (s.armed && std::this_thread::get_id() == s.armingThread)
            s.callsOnArmingThread++;
        else if (s.armed && !s.open)
        {
            s.callsBlocked++;
            s.cv.wait(lock, [&s]() { return s.open; });
        }
    }
    return integerTestHook(seed, x, z);
}

// ---- the renderer side (no engine): layouts, test tiles and pyramids, the upload wait

// A layout with the default budgets (48 tiles per level, 16 grass layers)
inline Vulkan::TerrainLayout testLayout(int32_t sizeChunks, int baseSpacing)
{
    Vulkan::TerrainLayout l;
    l.sizeChunks  = sizeChunks;
    l.baseSpacing = baseSpacing;
    return l;
}
inline Vulkan::TerrainTileData testTile(TerrainHeightHook hook, const Vulkan::TerrainLayout& layout,
                                        const RegionId& region, uint64_t seed = 1)
{
    return Vulkan::generateTerrainTile(TerrainHeightSource::fromHook(hook, seed), layout, region);
}
inline Vulkan::TerrainPyramidData testPyramid(TerrainHeightHook hook, const Vulkan::TerrainLayout& layout,
                                              std::optional<int32_t> seaLevel = std::nullopt, uint64_t seed = 1)
{
    return Vulkan::generateTerrainPyramid(TerrainHeightSource::fromHook(hook, seed), layout, seaLevel);
}
// Renders frames until no terrain upload is pending (at most maxFrames); returns the frames rendered
inline int renderUntilUploaded(Vulkan::RenderScene& scene, int maxFrames = 64)
{
    int frames = 0;
    while (frames < maxFrames && scene.terrainUploadsPending() > 0)
    {
        scene.renderFrame([](vk::CommandBuffer) {}, [](vk::CommandBuffer) {}, nullptr);
        frames++;
    }
    return frames;
}

// The surface height of the pyramid (32 m samples, the integer surface rule) at (x, z), clamped to the space
inline double pyramidSurfaceHeight(const Vulkan::TerrainLayout& layout, const Vulkan::TerrainPyramidData& pyramid,
                                   double x, double z)
{
    const int P       = layout.pyramidSamplesPerSide();
    const double size = double(layout.sizeChunks) * 256;
    const double gx = std::min(std::max(x, 0.0), size) / 32, gz = std::min(std::max(z, 0.0), size) / 32;
    const int i = std::min(int(gx), P - 2), j = std::min(int(gz), P - 2);
    auto h = [&](int a, int b) { return (*pyramid.heights)[size_t(b) * size_t(P) + size_t(a)]; };
    return terrainSurfaceDouble(h(i, j), h(i + 1, j), h(i, j + 1), h(i + 1, j + 1), gx - i, gz - j) * kHeightUnit;
}

// The streamed tiles of the default rule (§4.3, TerrainSystem::computeDesired) for a camera: per level the tiles within
// 400 * 2^L m of TerrainPass's LOD distance sqrt(dxz^2 + hc^2) (dxz the horizontal distance to the tile, hc the camera
// height above ground: the given ground height, else the pyramid's surface), the nearest tilesPerLevel
inline std::vector<RegionId> desiredTestTiles(const Vulkan::TerrainLayout& layout,
                                              const Vulkan::TerrainPyramidData& pyramid, const Vec3& camera,
                                              SpaceId space = 0, int firstLevel = 0,
                                              std::optional<double> ground = std::nullopt)
{
    std::vector<RegionId> out;
    const double hc =
        std::abs(camera.y() - (ground ? *ground : pyramidSurfaceHeight(layout, pyramid, camera.x(), camera.z())));
    for (int k = firstLevel; k < layout.streamedLevels(); ++k)
    {
        const double range = 400.0 * double(1 << k);
        const int64_t size = int64_t(256) << k;
        const int32_t W    = layout.tilesPerSide(k);
        std::vector<std::pair<double, RegionId>> level;
        for (int32_t tz = 0; tz < W; ++tz)
            for (int32_t tx = 0; tx < W; ++tx)
            {
                const double dx = std::max({0.0, double(tx * size) - camera.x(), camera.x() - double((tx + 1) * size)});
                const double dz = std::max({0.0, double(tz * size) - camera.z(), camera.z() - double((tz + 1) * size)});
                const double d  = std::sqrt(dx * dx + dz * dz + hc * hc);
                if (d > range) continue;
                RegionId r;
                r.space = space;
                r.level = uint8_t(k);
                r.x     = tx;
                r.z     = tz;
                level.push_back({d, r});
            }
        std::stable_sort(level.begin(), level.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        if (int(level.size()) > layout.tilesPerLevel) level.resize(size_t(layout.tilesPerLevel));
        for (auto& e : level) out.push_back(e.second);
    }
    return out;
}

// Fills a RenderScene without an engine: the pyramid, the desired tiles of the camera (desiredTestTiles), with
// withMasks the prototypeGrassMasks (default thresholds 0.82, 0.4 m) of the level-0 chunks within 80 m, then waits for
// the uploads
inline void fillTestTerrain(Vulkan::RenderScene& scene, const Vulkan::TerrainLayout& layout,
                            const TerrainHeightSource& source, std::optional<int32_t> seaLevel, const Vec3& camera,
                            bool withMasks, SpaceId space = 0)
{
    const Vulkan::TerrainPyramidData pyramid = Vulkan::generateTerrainPyramid(source, layout, seaLevel);
    scene.setTerrainPyramid(space, pyramid);
    for (const RegionId& r : desiredTestTiles(layout, pyramid, camera, space))
    {
        const Vulkan::TerrainTileData tile = Vulkan::generateTerrainTile(source, layout, r);
        scene.setTerrainTile(r, tile);
        if (withMasks && r.level == 0)
        {
            const Vec3 bmin(double(r.x) * 256, -1e9, double(r.z) * 256),
                bmax(double(r.x + 1) * 256, 1e9, double(r.z + 1) * 256);
            const Vec3 p = camera.cwiseMax(bmin).cwiseMin(bmax);
            if (Vec2(p.x() - camera.x(), p.z() - camera.z()).norm() > 80) continue;
            ChunkId c;
            c.space = space;
            c.x     = r.x;
            c.z     = r.z;
            scene.setGrassMask(c, Vulkan::prototypeGrassMask(tile, layout, seaLevel, 0.82, 0.4));
        }
    }
    renderUntilUploaded(scene, 256);
}

#ifdef SAIGA_TEST_ENGINE
// A space template for the tests: open world, the hook and sea level given
inline Engine::SpaceTemplate testSpaceTemplate(int32_t sizeChunks, int baseSpacing, TerrainHeightHook hook,
                                               std::optional<int32_t> seaLevel = std::nullopt)
{
    Engine::SpaceTemplate t;
    t.name            = "test";
    t.sizeChunks      = sizeChunks;
    t.baseSpacing     = baseSpacing;
    t.prototypeHeight = hook;
    t.seaLevel        = seaLevel;
    return t;
}
#endif

}  // namespace TerrainTestHooks
