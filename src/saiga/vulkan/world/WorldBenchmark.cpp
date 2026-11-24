/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/world/WorldBenchmark.h"

#include "saiga/core/util/assert.h"
#include "saiga/saiga_buildconfig.h"
#include "saiga/vulkan/world/PrototypeTerrain.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>

namespace Saiga
{
namespace Vulkan
{
namespace
{
// The first and the last measured frame number (RenderScene's frame numbers count from 1)
constexpr uint64_t firstMeasured = WorldBenchmark::warmupFrames + 1;
constexpr uint64_t lastMeasured  = WorldBenchmark::warmupFrames + WorldBenchmark::measuredFrames;
// GPU times arrive a few frames late; if the last measured frame has no GPU sample, stop this much later
constexpr uint64_t gpuGraceFrames = 10;

bool measured(uint64_t frame)
{
    return frame >= firstMeasured && frame <= lastMeasured;
}

}  // namespace

// The build configuration of saiga: the CMake configuration (SAIGA_BUILD_CONFIG, src/saiga/vulkan/CMakeLists.txt),
// SAIGA_ASSERTS and whether -march=native is really used (saiga_buildconfig.h, decided by Saiga_compiler_flags.cmake).
json11::Json WorldBenchmark::buildInfo()
{
    std::string type = SAIGA_BUILD_CONFIG;
    if (type.empty())
    {
        // A single-config build without CMAKE_BUILD_TYPE
#ifdef NDEBUG
        type = "release (NDEBUG)";
#else
        type = "debug (no NDEBUG)";
#endif
    }
#ifdef SAIGA_ASSERTS
    bool asserts = true;
#else
    bool asserts = false;
#endif
#ifdef SAIGA_ARCHNATIVE_ACTIVE
    bool archnative = true;
#else
    bool archnative = false;
#endif
    return json11::Json::object{{"type", type}, {"asserts", asserts}, {"archnative", archnative}};
}

const std::vector<std::string>& WorldBenchmark::names()
{
    // S36b: night is the flyover path at 22:00 with game_world's lamps, night_town (round 1, finding 19) a low circle
    // inside a dense lamp grid at 22:00 (both reported, not in benchmarks.json)
    static const std::vector<std::string> n = {"flyover", "forest", "terraform", "night", "night_town"};
    return n;
}

const std::vector<std::string>& WorldBenchmark::gpuKeys()
{
    static const std::vector<std::string> k = {"upload + cull", "terrain + water", "trees", "grass", "sky"};
    return k;
}

bool WorldBenchmark::known(const std::string& name)
{
    auto& n = names();
    return std::find(n.begin(), n.end(), name) != n.end();
}

WorldBenchmark::Pose WorldBenchmark::pose(const std::string& name, const Ground& ground, int measuredFrame)
{
    SAIGA_ASSERT(known(name), "unknown benchmark");
    // The centre of the prototype's 8 km area (until S27, §17.7; S36: at the space's centre)
    const double c = kPrototypeAreaCentre;
    Pose p;
    if (name == "flyover" || name == "night")
    {
        const double pi    = 3.14159265358979323846;
        const double angle = 2.0 * pi * double(measuredFrame) / measuredFrames;
        const double x     = c + 3000.0 * std::cos(angle);
        const double z     = c + 3000.0 * std::sin(angle);
        p.eye              = vec3(float(x), float(ground(x, z) + 300.0), float(z));
        p.target           = vec3(float(c), float(ground(c, c)), float(c));
    }
    else if (name == "night_town")
    {
        // S36b round 1, finding 19: a circle of 150 m radius 30 m above the ground around the centre of game_world's
        // lamp grid (25 m spacing around the area's centre), looking at the centre: hundreds of lights in view, more
        // shadowed lamps than the atlas's tiles, the clusters full
        const double pi    = 3.14159265358979323846;
        const double angle = 2.0 * pi * double(measuredFrame) / measuredFrames;
        const double x     = c + 150.0 * std::cos(angle);
        const double z     = c + 150.0 * std::sin(angle);
        p.eye              = vec3(float(x), float(ground(x, z) + 30.0), float(z));
        p.target           = vec3(float(c), float(ground(c, c)), float(c));
    }
    else if (name == "terraform")
    {
        // S22: a static pose about 85 degrees down over the four chunk corners of the edit schedule (the brush areas
        // x, z in the area's centre - [282, -26]; the game's TerraformBenchmark); g = the ground at the centre - 128,
        // never edited
        const double e = c - 128;
        const double g = ground(e, e);
        p.eye          = vec3(float(e), float(g + 300.0), float(e - 28));
        p.target       = vec3(float(e), float(g), float(e));
    }
    else
    {
        const double t = double(measuredFrame) / double(measuredFrames - 1);
        const double x = c - 300.0 + 600.0 * t;
        p.eye          = vec3(float(x), float(ground(x, c) + 2.0), float(c));
        p.target       = p.eye + vec3(1, 0, 0);
    }
    return p;
}

WorldBenchmark::Stat WorldBenchmark::summarize(std::vector<double> values)
{
    Stat s;
    if (values.empty()) return s;
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    s.median = n % 2 == 1 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
    // Nearest rank: the smallest value with at least 95% of the values <= it
    size_t rank = size_t(std::ceil(0.95 * double(n)));
    s.p95       = values[std::max<size_t>(rank, 1) - 1];
    double sum  = 0;
    for (double v : values) sum += v;
    s.mean           = sum / double(n);
    s.p99            = values[std::max<size_t>(size_t(std::ceil(0.99 * double(n))), 1) - 1];
    s.max            = values.back();
    const size_t one = (n + 99) / 100;
    double worst     = 0;
    for (size_t i = n - one; i < n; ++i) worst += values[i];
    s.worst1Percent = worst / double(one);
    return s;
}

WorldBenchmark::WorldBenchmark(const std::string& name, Ground ground, const std::string& gpuName)
    : name(name), gpuName(gpuName), groundFn(std::move(ground))
{
    SAIGA_ASSERT(known(name), "unknown benchmark");
    SAIGA_ASSERT(groundFn, "WorldBenchmark: no ground");
}

WorldBenchmark::Pose WorldBenchmark::currentPose(uint64_t frame) const
{
    const int64_t i =
        std::min<int64_t>(std::max<int64_t>(int64_t(frame) - int64_t(firstMeasured), 0), measuredFrames - 1);
    return pose(name, groundFn, int(i));
}

bool WorldBenchmark::measuring(uint64_t frame) const
{
    return measured(frame);
}

void WorldBenchmark::addSample(const Sample& t)
{
    if (t.intervalMs >= 0 && t.frame > lastCpuFrame)
    {
        if (measured(t.frame))
        {
            // The terrain stream time is not part of the frame interval (S20)
            cpuMs.push_back(std::max(t.intervalMs - t.terrainStreamMs, 0.0));
            streamMs.push_back(t.terrainStreamMs);
            if (t.workMs >= 0) cpuWorkMs.push_back(t.workMs);
            if (t.frameWorkMs >= 0) frameWorkMs.push_back(t.frameWorkMs);
            if (t.waitPartsMs[0] >= 0) waitParts.push_back(t.waitPartsMs);
        }
        lastCpuFrame = t.frame;
    }
    if (!t.sectionMs.empty() && t.frame > lastGpuFrame)
    {
        if (measured(t.frame))
        {
            for (const auto& [key, ms] : t.sectionMs) gpuMs[key].push_back(ms);
            gpuFrames++;
        }
        lastGpuFrame = t.frame;
    }
}

bool WorldBenchmark::finished() const
{
    if (lastCpuFrame < lastMeasured) return false;
    return lastGpuFrame >= lastMeasured || lastCpuFrame >= lastMeasured + gpuGraceFrames;
}

int WorldBenchmark::countStalls(const std::vector<double>& intervalsMs)
{
    double median = summarize(intervalsMs).median;
    int stalls    = 0;
    for (double ms : intervalsMs)
        if (ms > 4 * median && ms > 50) stalls++;
    return stalls;
}

json11::Json WorldBenchmark::result() const
{
    auto stat = [](const std::vector<double>& v)
    {
        Stat s = summarize(v);
        return json11::Json::object{{"median", s.median}, {"p95", s.p95}};
    };
    auto detail = [](const std::vector<double>& v)
    {
        Stat s = summarize(v);
        return json11::Json::object{
            {"mean", s.mean}, {"median", s.median}, {"p95", s.p95}, {"p99", s.p99}, {"max", s.max}};
    };
    const Stat interval = summarize(cpuMs);
    // Long: > max(1.5 x the median, the median + 1 ms) (the 1 ms keeps sub-millisecond jitter at high frame rates out)
    int longFrames = 0;
    for (double ms : cpuMs) longFrames += ms > std::max(1.5 * interval.median, interval.median + 1.0) ? 1 : 0;
    // Frame pacing: the p99 change of the interval between neighbouring frames
    std::vector<double> change;
    for (size_t i = 1; i < cpuMs.size(); ++i) change.push_back(std::abs(cpuMs[i] - cpuMs[i - 1]));
    const double pacing = summarize(change).p99;
    // Alternating intervals (most changes above 0.1 ms flip their sign): uneven pacing rather than single long frames
    int flips = 0, changes = 0;
    for (size_t i = 2; i < cpuMs.size(); ++i)
    {
        const double a = cpuMs[i - 1] - cpuMs[i - 2], b = cpuMs[i] - cpuMs[i - 1];
        if (std::abs(a) < 0.1 || std::abs(b) < 0.1) continue;
        changes++;
        flips += (a > 0) != (b > 0) ? 1 : 0;
    }
    // Sustained alternation: most neighbouring changes are above 0.1 ms (isolated spikes leave flat changes), nearly
    // all of them flip their sign (random jitter flips about 2 of 3), and the typical (median) change is a quarter of
    // the frame (the p99 is set by single spikes)
    const bool alternating = changes >= 20 && changes * 2 >= int(cpuMs.size()) && flips > 0.85 * changes &&
                             summarize(change).median > 0.25 * interval.mean;
    // The CPU's waits per frame (means): the frame slot, the image acquire, the present
    std::array<double, 3> wait = {};
    for (const auto& w : waitParts)
        for (size_t k = 0; k < 3; ++k) wait[k] += w[k] / double(waitParts.size());
    // Without GPU samples (no timestamp support) gpu_ms is null, not zeros that look like a fast result
    json11::Json gpu;
    if (gpuFrames > 0)
    {
        json11::Json::object passes;
        for (const auto& [key, values] : gpuMs) passes[key] = stat(values);
        gpu = passes;
    }
    double streamMean = 0;
    for (double v : streamMs) streamMean += v / double(streamMs.size());
    return json11::Json::object{
        {"gpu", gpuName},
        {"quality", "default"},
        {"build", buildInfo()},
        // The workload: the prototype area's origin (S36: at the space's centre; the paths are around its centre)
        {"area_origin", kPrototypeAreaOrigin},
        {"frames", int(cpuMs.size())},
        {"gpu_frames", gpuFrames},
        {"width", width},
        {"height", height},
        {"cpu_ms", stat(cpuMs)},
        {"cpu_work_ms", stat(cpuWorkMs)},
        {"gpu_ms", gpu},
        {"stalled_frames", countStalls(cpuMs)},
        // The tail of the frame interval (stutter): mean, p99, max, the 1% low frame rate and the frames longer than
        // 1.5 x the median; the CPU work of the whole frame (cpu_work_ms is the recording only)
        {"frame_interval_ms", detail(cpuMs)},
        {"fps", json11::Json::object{{"mean", interval.mean > 0 ? 1000 / interval.mean : 0.0},
                                     {"low_1pct", interval.worst1Percent > 0 ? 1000 / interval.worst1Percent : 0.0}}},
        {"long_frames", longFrames},
        {"pacing_p99_ms", pacing},
        {"alternating", alternating},
        {"wait_ms",
         json11::Json::object{
             {"frame_slot", wait[0]}, {"acquire", wait[1]}, {"present", wait[2]}, {"terrain_stream", streamMean}}},
        {"frame_work_ms", detail(frameWorkMs)},
        // S20: the terrain stream time of the measured frames (excluded from the interval) and the terrain block
        {"terrain_stream_ms", detail(streamMs)},
        {"terrain", terrainInfo},
    };
}

std::string WorldBenchmark::summaryText(const json11::Json& r, const json11::Json& baseline)
{
    std::string out;
    char line[300];
    const auto& b = r["build"];
    std::snprintf(line, sizeof(line), "%s, %d x %d, %d frames, %s%s\n", r["gpu"].string_value().c_str(),
                  r["width"].int_value(), r["height"].int_value(), r["frames"].int_value(),
                  b["type"].string_value().c_str(), b["asserts"].bool_value() ? " with asserts" : "");
    out += line;
    const auto& fps  = r["fps"];
    const int frames = std::max(r["frames"].int_value(), 1);
    const int longs  = r["long_frames"].int_value();
    std::snprintf(line, sizeof(line),
                  "%.0f fps, 1%% low %.0f fps, max frame %.2f ms, long frames (> 1.5 x median) %d (%.0f%%), stalled "
                  "%d\n",
                  fps["mean"].number_value(), fps["low_1pct"].number_value(),
                  r["frame_interval_ms"]["max"].number_value(), longs, 100.0 * longs / frames,
                  r["stalled_frames"].int_value());
    out += line;
    if (r["alternating"].bool_value())
    {
        std::snprintf(line, sizeof(line),
                      "uneven frame pacing: the intervals alternate between short and long (p99 change between frames "
                      "%.2f ms); the long frames are this, not single stutters\n",
                      r["pacing_p99_ms"].number_value());
        out += line;
    }
    else if (longs * 10 > frames)
    {
        std::snprintf(line, sizeof(line), "frequent long frames (p99 change between frames %.2f ms)\n",
                      r["pacing_p99_ms"].number_value());
        out += line;
    }
    const auto& wait = r["wait_ms"];
    if (wait.is_object())
    {
        std::snprintf(
            line, sizeof(line), "CPU wait per frame (means): frame slot %.2f ms, acquire %.2f ms, present %.2f ms\n",
            wait["frame_slot"].number_value(), wait["acquire"].number_value(), wait["present"].number_value());
        out += line;
    }
    std::snprintf(line, sizeof(line), "%-16s %8s %8s %8s %8s\n", "per frame (ms)", "mean", "median", "p99", "max");
    out += line;
    auto row = [&](const char* name, const json11::Json& v)
    {
        if (!v.is_object()) return;
        std::snprintf(line, sizeof(line), "%-16s %8.2f %8.2f %8.2f %8.2f\n", name, v["mean"].number_value(),
                      v["median"].number_value(), v["p99"].number_value(), v["max"].number_value());
        out += line;
    };
    row("frame interval", r["frame_interval_ms"]);
    row("CPU busy", r["frame_work_ms"]);
    row("terrain stream", r["terrain_stream_ms"]);
    row("GPU busy", r["gpu_frame_ms"]["gpu"]);
    // What limits the frame rate (no vsync in a benchmark): the busy times against the mean interval (the frames' time)
    const double frameMs = r["frame_interval_ms"]["mean"].number_value();
    // No GPU times (no timestamps): no GPU utilization and no verdict
    const bool hasGpu   = r["gpu_frame_ms"]["gpu"]["mean"].is_number();
    const double gpuMs  = r["gpu_frame_ms"]["gpu"]["mean"].number_value();
    const double workMs = r["frame_work_ms"]["mean"].number_value();
    if (frameMs > 0 && !hasGpu)
    {
        std::snprintf(line, sizeof(line),
                      "no GPU times (timestamps not supported): CPU busy %.0f%% of the frame time\n",
                      100 * workMs / frameMs);
        out += line;
    }
    else if (frameMs > 0)
    {
        const double gpu = gpuMs / frameMs, cpu = workMs / frameMs;
        // Neither fills the frame: the largest wait (the frame slot, the image acquire, the present)
        const auto& w           = r["wait_ms"];
        const double waits[]    = {w["frame_slot"].number_value(), w["acquire"].number_value(),
                                   w["present"].number_value()};
        const char* waitNames[] = {"the frame slot (an earlier GPU frame)", "the swap chain image acquire",
                                   "the present"};
        const size_t waitMax    = size_t(std::max_element(waits, waits + 3) - waits);
        std::string waitText    = w.is_object() ? std::string(waitNames[waitMax]) + " (" +
                                                   std::to_string(int(std::lround(100 * waits[waitMax] / frameMs))) +
                                                   "% of the frame time)"
                                                : std::string("between frames");
        const char* busier      = gpu >= cpu ? "GPU" : "CPU";
        std::string verdict     = std::max(gpu, cpu) >= 0.9 ? std::string("limited by the ") + busier
                                  : std::max(gpu, cpu) >= 0.6
                                      ? std::string("mostly ") + busier + "-bound; the CPU waits mostly for " + waitText
                                  : w.is_object() && waits[waitMax] >= 0.05
                                      ? std::string("limited by waits for ") + waitText
                                      : std::string("limit undetermined");
        std::snprintf(line, sizeof(line), "%s: GPU busy %.0f%%, CPU busy %.0f%% of the frame time (means)\n",
                      verdict.c_str(), 100 * gpu, 100 * cpu);
        out += line;
    }
    // The largest GPU passes (not the sub-sections)
    std::vector<std::pair<double, std::string>> sections;
    for (const auto& [name, v] : r["gpu_frame_ms"].object_items())
        if (name != "gpu" && name.find('.') == std::string::npos)
            sections.push_back({v["median"].number_value(), name});
    std::sort(sections.rbegin(), sections.rend());
    if (!sections.empty())
    {
        out += "GPU passes (median ms):";
        for (size_t i = 0; i < sections.size() && i < 4; ++i)
        {
            std::snprintf(line, sizeof(line), " %s %.2f", sections[i].second.c_str(), sections[i].first);
            out += line;
        }
        out += "\n";
    }
    // The committed baseline of this GPU (data/tests/benchmarks.json): the change of the S03 numbers. A baseline of
    // another workload (the prototype area elsewhere: before S36 in the space's corner, no area_origin) is not compared
    if (baseline.is_object() && baseline["area_origin"].number_value() != r["area_origin"].number_value())
    {
        std::snprintf(line, sizeof(line),
                      "baseline of another workload (area origin %.0f, now %.0f): not compared, measure a new one\n",
                      baseline["area_origin"].number_value(), r["area_origin"].number_value());
        out += line;
    }
    else if (baseline.is_object())
    {
        // The warning first: a baseline of another build is no comparison
        const auto& bb = baseline["build"];
        if (bb.is_object() &&
            (bb["type"] != b["type"] || bb["asserts"] != b["asserts"] || bb["archnative"] != b["archnative"]))
        {
            std::snprintf(line, sizeof(line), "baseline not comparable (it is a %s build %s asserts):\n  ",
                          bb["type"].string_value().c_str(), bb["asserts"].bool_value() ? "with" : "without");
            out += line;
        }
        auto change = [](double now, double base) { return base > 0 ? 100.0 * (now - base) / base : 0.0; };
        std::snprintf(line, sizeof(line), "vs baseline: frame interval median %+.1f%%, S03 passes",
                      change(r["cpu_ms"]["median"].number_value(), baseline["cpu_ms"]["median"].number_value()));
        out += line;
        for (const auto& [key, v] : r["gpu_ms"].object_items())
        {
            const double base = baseline["gpu_ms"][key]["median"].number_value();
            if (base < 0.05) continue;  // below 50 us the change is noise
            std::snprintf(line, sizeof(line), " %s %+.1f%%", key.c_str(), change(v["median"].number_value(), base));
            out += line;
        }
        out += "\n";
    }
    return out;
}

bool WorldBenchmark::writeResult(const std::string& file, const json11::Json& json)
{
    std::ofstream out(file);
    if (!out) return false;
    out << json.dump() << "\n";
    // Regression: a failure of the final flush was reported as success (the stream was closed after the return)
    out.close();
    return bool(out);
}

}  // namespace Vulkan
}  // namespace Saiga
