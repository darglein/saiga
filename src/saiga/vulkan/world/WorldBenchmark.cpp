/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/world/WorldBenchmark.h"

#include "saiga/core/util/assert.h"
#include "saiga/saiga_buildconfig.h"

#include <algorithm>
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
    static const std::vector<std::string> n = {"flyover", "forest"};
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

WorldBenchmark::Pose WorldBenchmark::pose(const std::string& name, const HeightField& hf, int measuredFrame)
{
    SAIGA_ASSERT(known(name), "unknown benchmark");
    // The world center: (4096, 4096) for the default 8 km world
    float c = hf.worldSize() * 0.5f;
    Pose p;
    if (name == "flyover")
    {
        const double pi = 3.14159265358979323846;
        double angle    = 2.0 * pi * double(measuredFrame) / measuredFrames;
        float x         = c + 3000.0f * float(std::cos(angle));
        float z         = c + 3000.0f * float(std::sin(angle));
        p.eye           = vec3(x, hf.height(x, z) + 300.0f, z);
        p.target        = vec3(c, hf.height(c, c), c);
    }
    else
    {
        float t  = float(measuredFrame) / float(measuredFrames - 1);
        float x  = c - 300.0f + 600.0f * t;
        p.eye    = vec3(x, std::max(hf.height(x, c), hf.waterLevel()) + 2.0f, c);
        p.target = p.eye + vec3(1, 0, 0);
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
    return s;
}

WorldBenchmark::WorldBenchmark(const std::string& name, const HeightField& heightField, const std::string& gpuName)
    : name(name), gpuName(gpuName), hf(&heightField)
{
    SAIGA_ASSERT(known(name), "unknown benchmark");
}

WorldBenchmark::Pose WorldBenchmark::currentPose(uint64_t frame) const
{
    const int64_t i =
        std::min<int64_t>(std::max<int64_t>(int64_t(frame) - int64_t(firstMeasured), 0), measuredFrames - 1);
    return pose(name, *hf, int(i));
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
            cpuMs.push_back(t.intervalMs);
            if (t.workMs >= 0) cpuWorkMs.push_back(t.workMs);
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
    // Without GPU samples (no timestamp support) gpu_ms is null, not zeros that look like a fast result
    json11::Json gpu;
    if (gpuFrames > 0)
    {
        json11::Json::object passes;
        for (const auto& [key, values] : gpuMs) passes[key] = stat(values);
        gpu = passes;
    }
    return json11::Json::object{
        {"gpu", gpuName},
        {"quality", "default"},
        {"build", buildInfo()},
        {"frames", int(cpuMs.size())},
        {"gpu_frames", gpuFrames},
        {"width", width},
        {"height", height},
        {"cpu_ms", stat(cpuMs)},
        {"cpu_work_ms", stat(cpuWorkMs)},
        {"gpu_ms", gpu},
        {"stalled_frames", countStalls(cpuMs)},
    };
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
