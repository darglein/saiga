/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/json11.hpp"
#include "saiga/vulkan/world/HeightField.h"
#include "saiga/vulkan/world/WorldRenderer.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
/**
 * Benchmarks of the world prototype (docs/engine_design.md §17.7): a fixed camera path over warmupFrames +
 * measuredFrames frames; the unsmoothed frame intervals, CPU work times and GPU pass times of the measured frames are
 * summarized as median and p95.
 *
 * Usage (game_world --benchmark <name>):
 *   WorldBenchmark bench(name, world, gpuName);
 *   per frame, before world.transfer(): pose = bench.currentPose(); set the camera; world.transfer(...);
 *   if (bench.finished()) { json = bench.result(); WorldBenchmark::writeResult("benchmark_" + name + ".json", json);
 *   quit; }
 * The sample ignores SAIGA_SCREENSHOT in benchmark mode (the automatic screenshot would close the window at frame 60).
 *
 * Moves to engine/app/ in S13.
 */
class SAIGA_VULKAN_API WorldBenchmark
{
   public:
    static constexpr int warmupFrames   = 60;
    static constexpr int measuredFrames = 600;

    struct Pose
    {
        vec3 eye    = vec3(0, 0, 0);
        vec3 target = vec3(0, 0, -1);
    };
    struct Stat
    {
        double median = 0, p95 = 0;
    };

    // The known benchmarks: "flyover", "forest"
    static const std::vector<std::string>& names();
    static bool known(const std::string& name);

    /**
     * The camera of measured frame i (0 <= i < measuredFrames; warm-up frames use i = 0):
     * - flyover: one revolution in measuredFrames frames on a circle of radius 3 km around the world center, 300 m
     *   above the ground below the camera, looking at the center (on the ground);
     * - forest: 2 m above max(terrain, water level), from center - 300 m to center + 300 m along x, looking along +x.
     */
    static Pose pose(const std::string& name, const HeightField& hf, int measuredFrame);

    // Median and p95 (nearest rank) of the values; zeros for an empty list.
    static Stat summarize(std::vector<double> values);

    // Frame intervals of a stalled frame: longer than 4 x the median and longer than 50 ms (a minimized window, a swap
    // chain recreation, another program). A run with a stalled frame is not comparable.
    static int countStalls(const std::vector<double>& intervalsMs);

    // Registers a frame callback at the world. The callback only uses the shared sample state (never this object); it
    // removes itself when the benchmark is finished or the benchmark object was destroyed.
    WorldBenchmark(const std::string& name, WorldRenderer& world, const std::string& gpuName);
    WorldBenchmark(const WorldBenchmark&)            = delete;
    WorldBenchmark& operator=(const WorldBenchmark&) = delete;

    // The camera of the frame that the next world.transfer() prepares.
    Pose currentPose() const;

    // Adds the times of one frame (called by the frame callback; public for tests). Samples outside of the measured
    // frames are ignored.
    void addSample(const WorldRenderer::FrameTimes& times);

    // All measured CPU samples are in, and the GPU samples of the last measured frame (or a few frames more).
    bool finished() const;

    /**
     * {gpu, quality, build: {type, asserts, archnative}, frames, gpu_frames, width, height, cpu_ms: {median, p95},
     * cpu_work_ms: {median, p95}, gpu_ms: {<pass>: {median, p95}}, stalled_frames}. cpu_ms is the frame interval
     * (FrameTimes::cpuMs), stalled_frames the number of measured frames whose interval is a stall (countStalls),
     * cpu_work_ms the CPU time of transfer() + render(). gpu_ms is null if no GPU times were measured (no timestamp
     * support).
     */
    json11::Json result() const;
    // Writes a result (computed once by result()) to the file. Returns false if it could not be written.
    static bool writeResult(const std::string& file, const json11::Json& json);

    // The frame that the next world.transfer() prepares is a measured frame (not warm-up).
    bool measuring() const;

    // The size of the rendered images (the swap chain extent of the measured frames), set by the caller
    int width = 0, height = 0;

   private:
    struct State
    {
        std::vector<double> cpuMs, cpuWorkMs;
        std::array<std::vector<double>, WorldRenderer::PASS_COUNT> gpuPassMs;
        int lastCpuFrame = 0, lastGpuFrame = 0;
        void add(const WorldRenderer::FrameTimes& times);
        bool finished() const;
    };
    std::string name, gpuName;
    WorldRenderer* world;
    std::shared_ptr<State> state;
};

}  // namespace Vulkan
}  // namespace Saiga
