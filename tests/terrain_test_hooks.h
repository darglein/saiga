/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Shared terrain test helpers (S20; S22 and S23 add theirs): cheap integer height hooks, the gated latchedHook, and the
// engine-only templates (inside SAIGA_TEST_ENGINE, which saiga_test defines for the tests that link saiga_engine).

#pragma once

#include "saiga/core/math/TerrainSurface.h"

#include <algorithm>
#include <atomic>
#include <chrono>
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

// Periodic in 4096 m in x and z (so periodic in 28 672 m): hills from 0 to about 60 m and a valley below the sea level
// 6 m (the origin-offset test)
inline int32_t periodic4096Hook(uint64_t, int64_t x, int64_t z)
{
    const int64_t a = triangleWave(x, 4096) * 20;         // 0..40960 units (40 m)
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
    bool armed = false, open = true, sleepAtOrigin = false;
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
