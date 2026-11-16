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
// The first and the last measured frame number (frame numbers count from 1, WorldRenderer::frameNumber())
constexpr int firstMeasured = WorldBenchmark::warmupFrames + 1;
constexpr int lastMeasured  = WorldBenchmark::warmupFrames + WorldBenchmark::measuredFrames;
// GPU times arrive a few frames late; if the last measured frame is skipped by the readback, stop this much later
constexpr int gpuGraceFrames = 10;

bool measured(int frame)
{
    return frame >= firstMeasured && frame <= lastMeasured;
}

// The build configuration of saiga (saiga_buildconfig.h): the CMake build type, SAIGA_ASSERTS and whether
// -march=native is really used (decided by Saiga_compiler_flags.cmake).
json11::Json buildInfo()
{
    std::string type = SAIGA_BUILD_TYPE;
    if (type.empty())
    {
        // Multi-config generators have no CMAKE_BUILD_TYPE at configure time
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
}  // namespace

const std::vector<std::string>& WorldBenchmark::names()
{
    static const std::vector<std::string> n = {"flyover", "forest"};
    return n;
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

WorldBenchmark::WorldBenchmark(const std::string& name, WorldRenderer& world, const std::string& gpuName)
    : name(name), gpuName(gpuName), world(&world), state(std::make_shared<State>())
{
    SAIGA_ASSERT(known(name), "unknown benchmark");
    // The callback owns a reference to the state and never touches this object; it is removed when the benchmark is
    // finished or the benchmark object was destroyed (then the callback holds the last reference)
    auto s = state;
    world.frameCallbacks.push_back(
        [s](const WorldRenderer::FrameTimes& t)
        {
            if (s.use_count() == 1) return true;
            s->add(t);
            return s->finished();
        });
}

WorldBenchmark::Pose WorldBenchmark::currentPose() const
{
    int frame = world->frameNumber() + 1;
    int i     = std::min(std::max(frame - firstMeasured, 0), measuredFrames - 1);
    return pose(name, world->heightField(), i);
}

void WorldBenchmark::State::add(const WorldRenderer::FrameTimes& times)
{
    if (times.cpuMs >= 0 && times.cpuFrame > lastCpuFrame)
    {
        if (measured(times.cpuFrame))
        {
            cpuMs.push_back(times.cpuMs);
            if (times.cpuWorkMs >= 0) cpuWorkMs.push_back(times.cpuWorkMs);
        }
        lastCpuFrame = times.cpuFrame;
    }
    if (times.gpuMs >= 0 && times.gpuFrame > lastGpuFrame)
    {
        if (measured(times.gpuFrame))
            for (int p = 0; p < WorldRenderer::PASS_COUNT; ++p) gpuPassMs[p].push_back(times.gpuPassMs[p]);
        lastGpuFrame = times.gpuFrame;
    }
}

bool WorldBenchmark::State::finished() const
{
    if (lastCpuFrame < lastMeasured) return false;
    return lastGpuFrame >= lastMeasured || lastCpuFrame >= lastMeasured + gpuGraceFrames;
}

void WorldBenchmark::addSample(const WorldRenderer::FrameTimes& times)
{
    state->add(times);
}

bool WorldBenchmark::finished() const
{
    return state->finished();
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
    if (!state->gpuPassMs[0].empty())
    {
        json11::Json::object passes;
        for (int p = 0; p < WorldRenderer::PASS_COUNT; ++p)
            passes[WorldRenderer::gpuPassName(p)] = stat(state->gpuPassMs[p]);
        gpu = passes;
    }
    return json11::Json::object{
        {"gpu", gpuName},
        {"quality", "default"},
        {"build", buildInfo()},
        {"frames", int(state->cpuMs.size())},
        {"gpu_frames", int(state->gpuPassMs[0].size())},
        {"width", width},
        {"height", height},
        {"cpu_ms", stat(state->cpuMs)},
        {"cpu_work_ms", stat(state->cpuWorkMs)},
        {"gpu_ms", gpu},
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

bool WorldBenchmark::measuring() const
{
    return measured(world->frameNumber() + 1);
}

}  // namespace Vulkan
}  // namespace Saiga
