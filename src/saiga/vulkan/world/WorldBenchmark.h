/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"
#include "saiga/core/util/json11.hpp"
#include "saiga/export.h"

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
/**
 * Benchmarks of the world prototype (docs/engine_design.md §17.7): a fixed camera path over warmupFrames +
 * measuredFrames frames; the unsmoothed frame intervals, CPU work times and GPU times of the measured frames are
 * summarized as median and p95. S19a: fed by the game (samples/games/world/main.cpp) with its own samples (no engine
 * Profiler in saiga_vulkan): the frames are RenderScene's frame numbers (frames() + 1 is the frame being recorded).
 *
 * Usage (game_world --benchmark <name>):
 *   WorldBenchmark bench(name, heightField, gpuName);
 *   per frame before recording frame f: pose = bench.currentPose(f); set the camera;
 *   bench.addSample({f - 1, interval, work, {}});   // the CPU times of the previous frame
 *   bench.addSample({g, -1, -1, gpuMs});            // the GPU times of a completed frame g (S03 keys, gpuKeys())
 *   if (bench.finished()) { json = bench.result(); WorldBenchmark::writeResult("benchmark_" + name + ".json", json); }
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
        // mean, p99 (nearest rank), max and the mean of the largest 1% (ceil(n / 100) values: 1000 / it is the "1% low"
        // frame rate of frame intervals)
        double mean = 0, p99 = 0, max = 0, worst1Percent = 0;
    };
    // One frame's times (each part optional)
    struct Sample
    {
        uint64_t frame = 0;  // RenderScene's frame number (1, 2, ...)
        // The frame interval (ms) from this frame's renderFrame to the next one's (the S03 cpu_ms; -1: none)
        double intervalMs = -1;
        // The CPU work of the frame (ms): the app's frame/render/recording (-1: none); not comparable with the S03
        // value (the world renderer's transfer() + render())
        double workMs = -1;
        // The GPU times (ms) by the S03 keys of gpuKeys() (empty: no GPU part)
        std::map<std::string, double> sectionMs;
        // (After sectionMs: aggregate initializers like {g, -1, -1, gpuMs} keep their meaning)
        // The app's CPU work of the frame (ms): the frame without the render's waits for the frame slot, the swap chain
        // image and the present, of every attempt of the frame (-1: none); the "CPU busy" of the engine statistics
        double frameWorkMs = -1;
        // The CPU's waits of the frame (ms): the frame slot, the image acquire, the present ({-1, ...}: none)
        std::array<double, 3> waitPartsMs = {-1, -1, -1};
        // S20: the frame's terrain stream time (EngineApp::lastTerrainStreamMs: its completeView calls), subtracted
        // from intervalMs (the frame interval, the stalls and the frame rate exclude it) and reported separately
        double terrainStreamMs = 0;
    };
    // The ground under (x, z) for the camera paths (S20: max(terrainHeight, sea level))
    using Ground = std::function<double(double x, double z)>;

    // The known benchmarks: "flyover", "forest", "terraform" (S22: the edit schedule lives in the game)
    static const std::vector<std::string>& names();
    static bool known(const std::string& name);
    // The S03 keys of gpu_ms: "upload + cull", "terrain + water", "trees", "grass", "sky" (from RenderScene's
    // sections: transfer + cull, depth.terrain + scene.terrain, scene.instances, scene.grass, scene.sky; a sum of
    // sections can count their overlap twice: a section begins at TOP_OF_PIPE and ends at BOTTOM_OF_PIPE)
    static const std::vector<std::string>& gpuKeys();

    /**
     * The camera of measured frame i (0 <= i < measuredFrames; warm-up frames use i = 0):
     * - flyover: one revolution in measuredFrames frames on a circle of radius 3 km around the world center, 300 m
     *   above the ground below the camera, looking at the center (on the ground);
     * - forest: 2 m above max(terrain, water level), from center - 300 m to center + 300 m along x, looking along +x;
     * - terraform (S22): static, eye (3968, g + 300, 3940) looking at (3968, g, 3968), g = the ground at (3968, 3968).
     */
    // S20: the same paths over a ground callback (terrain or water, whichever is higher; the centre (4096, 4096) of the
    // prototype's 8 km area until S27)
    static Pose pose(const std::string& name, const Ground& ground, int measuredFrame);

    // Median and p95 (nearest rank) of the values; zeros for an empty list.
    static Stat summarize(std::vector<double> values);

    // Frame intervals of a stalled frame: longer than 4 x the median and longer than 50 ms (a minimized window, a swap
    // chain recreation, another program). A run with a stalled frame is not comparable.
    static int countStalls(const std::vector<double>& intervalsMs);

    // S20: over a ground callback
    WorldBenchmark(const std::string& name, Ground ground, const std::string& gpuName);
    WorldBenchmark(const WorldBenchmark&)            = delete;
    WorldBenchmark& operator=(const WorldBenchmark&) = delete;

    // The camera of RenderScene frame `frame` (warm-up frames 1..warmupFrames, then the measured frames)
    Pose currentPose(uint64_t frame) const;
    // The frame is a measured frame (not warm-up)
    bool measuring(uint64_t frame) const;

    // Adds the times of one frame. Samples outside of the measured frames are ignored, as are parts of frames older
    // than the newest one given (each part is delivered once, in frame order).
    void addSample(const Sample& sample);

    // All measured CPU samples are in, and the GPU samples of the last measured frame (or the CPU samples of a few
    // frames more).
    bool finished() const;
    // Finished with the CPU samples of every measured frame: a run that missed one (finished() only proves that the
    // samples reached the last frame) is incomplete, a failure rather than a result
    bool complete() const { return finished() && cpuMs.size() == size_t(measuredFrames); }

    /**
     * {gpu, quality, build: {type, asserts, archnative}, frames, gpu_frames, width, height, cpu_ms: {median, p95},
     * cpu_work_ms: {median, p95}, gpu_ms: {<S03 key>: {median, p95}}, stalled_frames}. cpu_ms is the frame interval,
     * stalled_frames the number of measured frames whose interval is a stall (countStalls), cpu_work_ms the app's
     * recording and submit (S19a). gpu_ms is null if no GPU times were measured (no timestamp support).
     * game_world adds step and recording (the engine's step and command recording times) and gpu_frame_ms: {gpu,
     * <section>: {median, p95}} (the app's gpu profile: the whole GPU frame and RenderScene's sections and
     * sub-sections of the measured frames, S17-S19a).
     */
    json11::Json result() const;
    /**
     * A readable summary of a result (the benchmark's console output; the file keeps the JSON): the run, the frame
     * rate with its 1% low and the long frames, a table of the frame interval, CPU work and GPU (mean, median, p99,
     * max) with what limits the frame rate, the largest GPU passes and, with a baseline (the GPU's entry of the
     * benchmark in data/tests/benchmarks.json), the change of the frame interval and the GPU passes.
     */
    static std::string summaryText(const json11::Json& result, const json11::Json& baseline = json11::Json());
    // Writes a result (computed once by result()) to the file. Returns false if it could not be written.
    static bool writeResult(const std::string& file, const json11::Json& json);
    // The build of saiga: {type, asserts, archnative} (also used by sample_vulkan_scene_benchmark)
    static json11::Json buildInfo();

    // The size of the rendered images (the swap chain extent of the measured frames), set by the caller
    int width = 0, height = 0;
    // S20: the terrain block of the result ({uploads, upload_mb, resident_tiles, cpu_mb, gpu_mb}), set by the game
    void setTerrainInfo(const json11::Json& info) { terrainInfo = info; }

   private:
    std::string name, gpuName;
    Ground groundFn;
    json11::Json terrainInfo;
    std::vector<double> streamMs;
    std::vector<double> cpuMs, cpuWorkMs, frameWorkMs;
    std::vector<std::array<double, 3>> waitParts;
    std::map<std::string, std::vector<double>> gpuMs;
    int gpuFrames         = 0;
    uint64_t lastCpuFrame = 0, lastGpuFrame = 0;
};

}  // namespace Vulkan
}  // namespace Saiga
