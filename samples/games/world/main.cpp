/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

/**
 * The world game (docs/engine_design.md §19): the large procedural world prototype (8 x 8 km terrain, 200k trees,
 * grass around the camera) on the engine app (EngineApp), rendered in a uniform gray. S44 builds the first gameplay
 * on it.
 *
 * Controls: WASD + mouse (left or middle button) to fly, shift = fast, space = up, F toggles the walk mode, G the GUI,
 * F1 the engine statistics, F12 a screenshot, ESC quits.
 *
 * MCP (http://127.0.0.1:8765/mcp, --mcp-port or SAIGA_MCP_PORT, 0 disables it): the engine's world and app tools and
 * the prototype's tools (screenshot, get_state, set_camera, move_camera, set_settings, terrain_query).
 *
 * Benchmarks (§17.7): game_world --benchmark <flyover|forest> flies a fixed camera path at 1920 x 1080 (validation,
 * vsync, GUI and MCP off, time 0, the default world: SAIGA_WORLD_TREES is ignored), writes
 * benchmark_<name>.json to the working directory and exits with 0, or with 1 if the benchmark did not finish, the
 * result could not be written, the rendered size changed or a frame stalled (an old result file is deleted at the
 * start). --headless runs without a window (agents, servers).
 */

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/engine/app/EngineApp.h"
#include "saiga/engine/app/EngineMain.h"
#include "saiga/saiga_buildconfig.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/world/WorldBenchmark.h"
#include "saiga/vulkan/world/WorldMcp.h"
#include "saiga/vulkan/world/WorldRenderer.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>

using namespace Saiga;
using namespace Saiga::Engine;

namespace
{
// The game's state outside the World: the prototype's settings and height field and its camera controls (render state,
// not world state)
struct WorldGame
{
    Vulkan::WorldRenderer world;
    std::unique_ptr<Vulkan::WorldBenchmark> bench;
    std::string benchmarkName;
    bool benchmarkDone = false, benchmarkOk = false, sizeChanged = false;
    bool walkMode   = false;
    float eyeHeight = 1.8f;
    bool fWasDown   = false;
    // The frame interval: from the previous frame's beforeRender to this one's (renderFrame to renderFrame)
    std::chrono::steady_clock::time_point lastBeforeRender;
    bool hasLastBeforeRender = false;
    double frameWork         = 0;  // the app's CPU work of the attempts of the frame of lastBeforeRenderFrame
    // The RenderScene frame of lastBeforeRender: a frame that was not submitted is attempted again with the same number
    // (its interval runs from the first attempt, so a stall stays in the result)
    uint64_t lastBeforeRenderFrame = 0;
    // The engine's steps of the measured frames, taken at the frame boundaries that close them
    int64_t stepSamplesAtStart = -1;
    // The step times of the measured range, collected per frame (the profile keeps only the last 600 steps: at 30 fps
    // the 600 measured frames have about 1200 steps)
    std::vector<double> stepTimes, extractTimes;
    uint64_t stepSamplesSeen = 0;
    bool stepTaken           = false;
    json11::Json::object stepStat, extractStat;
    // The app's gpu profile (the whole frame and RenderScene's sections) of the measured frames, matched by the
    // RenderScene frame number: the measured frames whose sample did not arrive yet, the values by section
    std::set<uint64_t> gpuFramesPending;
    std::map<std::string, std::vector<double>> gpuFrameMs;
    int gpuFramesWaited = 0;
};
WorldGame* g_game = nullptr;

std::string resultFile(const std::string& name)
{
    return "benchmark_" + name + ".json";
}

// The S03 keys of the benchmark's gpu_ms from RenderScene's sections of one frame (gpu/<section> of the app's sample).
// "upload + cull" is the sum of two sections that can overlap (a section begins at TOP_OF_PIPE and ends at
// BOTTOM_OF_PIPE, so cull's begin may precede transfer's end): it can exceed their combined elapsed time by at most the
// shorter section per frame (transfer, a median of about 0.005 ms in both benchmarks; not measured per frame)
std::map<std::string, double> s03Passes(const std::map<std::string, double>& section)
{
    auto get = [&](const char* n)
    {
        auto it = section.find(n);
        return it == section.end() ? 0.0 : it->second;
    };
    return {{"upload + cull", get("transfer") + get("cull")},
            {"terrain + water", get("depth.terrain") + get("scene.terrain")},
            {"trees", get("scene.instances")},
            {"grass", get("scene.grass")},
            {"sky", get("scene.sky")}};
}

void initWorld(EngineApp& app)
{
    WorldGame& g = *g_game;
    Vulkan::WorldParameters params;
    // A benchmark always uses the default world (its result is compared with the baseline)
    if (const char* trees = std::getenv("SAIGA_WORLD_TREES"); trees && g.benchmarkName.empty())
        params.treeCount = std::atoi(trees);
    g.world.init(params);
    g.world.attach(*app.scene());

    // Start on a hill: next to the highest point, looking over the world center
    auto& hf    = g.world.heightField();
    vec3 peak   = hf.highestPoint();
    vec3 center = vec3(hf.worldSize() * 0.5f, 0, hf.worldSize() * 0.5f);
    vec3 toPeak = vec3(peak(0) - center(0), 0, peak(2) - center(2));
    vec3 eye    = peak + (toPeak.norm() > 1 ? toPeak.normalized() * 150.0f : vec3(150, 0, 0));
    eye(1)      = std::max(hf.height(eye(0), eye(2)), peak(1) - 40) + 60;
    center(1)   = hf.height(center(0), center(2));
    app.setViewCamera(0, eye.cast<double>(), cameraLookRotation((center - eye).cast<double>()),
                      radians(double(g.world.settings.fovy)));

    if (!g.benchmarkName.empty())
    {
        std::string gpu = app.scene()->base().physicalDevice.getProperties().deviceName.data();
        g.bench         = std::make_unique<Vulkan::WorldBenchmark>(g.benchmarkName, g.world.heightField(), gpu);
        std::remove(resultFile(g.benchmarkName).c_str());
        std::cout << "Benchmark " << g.benchmarkName << " on " << gpu << std::endl;
    }
}

void beforeRender(EngineApp& app)
{
    WorldGame& g               = *g_game;
    Vulkan::RenderScene& scene = *app.scene();
    // The frame being recorded
    const uint64_t frame = scene.frames() + 1;
    g.world.update();
    // The app's free camera is this game's camera: the walk mode and the field of view of the settings adjust it as
    // tool writes (also while a tool holds it, S14)
    ViewCamera cam    = app.viewCamera();
    const double fovy = radians(double(g.world.settings.fovy));
    bool changed      = cam.fovy != fovy;
    cam.fovy          = fovy;
    if (g.walkMode)
    {
        const double y =
            double(g.world.heightField().height(float(cam.position(0)), float(cam.position(2)))) + double(g.eyeHeight);
        changed         = changed || y != cam.position(1);
        cam.position(1) = y;
    }
    if (changed) app.setViewCameraByTool(cam.space, cam.position, cam.rotation, cam.fovy, false);
    if (g.bench)
    {
        auto pose = g.bench->currentPose(frame);
        app.setViewCamera(0, pose.eye.cast<double>(), cameraLookRotation((pose.target - pose.eye).cast<double>()),
                          fovy);
        if (g.bench->measuring(frame))
        {
            ivec2 size = scene.size();
            if (g.bench->width != 0 && (size(0) != g.bench->width || size(1) != g.bench->height)) g.sizeChanged = true;
            g.bench->width  = size(0);
            g.bench->height = size(1);
        }
    }
    // The time of the wind (benchmarks: 0); the app sets the camera of the frame
    World& w = app.world();
    scene.setTime(g.bench ? 0.0 : w.time() + w.alpha() * w.dt());

    // The CPU times of the previous frame: its interval (until now) and its recording and submit
    const auto now = std::chrono::steady_clock::now();
    // The app's CPU work of the last loop belongs to the frame it attempted: summed over every attempt of a frame (a
    // frame that was not submitted is attempted again with the same number)
    {
        Profiler& fp = app.frameProfile();
        double work  = 0;
        if (g.hasLastBeforeRender && fp.enabled && fp.lastSample(fp.scope("work"), work)) g.frameWork += work;
    }
    if (!g.hasLastBeforeRender || frame != g.lastBeforeRenderFrame)
    {
        if (g.bench && g.hasLastBeforeRender)
        {
            Vulkan::WorldBenchmark::Sample sample;
            sample.frame       = frame - 1;
            sample.intervalMs  = std::chrono::duration<double, std::milli>(now - g.lastBeforeRender).count();
            sample.workMs      = scene.lastRecordingMs();
            sample.frameWorkMs = g.frameWork;
            sample.waitPartsMs = scene.lastWaitParts();
            g.bench->addSample(sample);
        }
        g.frameWork             = 0;
        g.lastBeforeRender      = now;
        g.lastBeforeRenderFrame = frame;
        g.hasLastBeforeRender   = true;
    }

    if (g.bench)
    {
        // Frame k is RenderScene's frame being recorded (W: the last warm-up frame, M1..Mn: the measured frames). In a
        // frame the steps run before beforeRender.
        const uint64_t lastWarm = Vulkan::WorldBenchmark::warmupFrames;
        const uint64_t lastMeas = lastWarm + Vulkan::WorldBenchmark::measuredFrames;
        Profiler& wp            = app.world().profiler();
        // Steps: from after W's steps (beforeRender of W) to after Mn's steps (beforeRender of Mn)
        if (frame == lastWarm)
        {
            g.stepSamplesAtStart = int64_t(wp.sampleCount());
            g.stepSamplesSeen    = wp.sampleCount();
        }
        if (g.stepSamplesAtStart >= 0 && !g.stepTaken && frame > lastWarm && frame <= lastMeas)
        {
            // This frame's steps: the newest samples of the step scope
            const uint64_t n                  = wp.sampleCount() - g.stepSamplesSeen;
            g.stepSamplesSeen                 = wp.sampleCount();
            const std::vector<double> history = wp.scopeHistory(wp.scope("step"));
            const size_t take                 = size_t(std::min<uint64_t>(n, history.size()));
            g.stepTimes.insert(g.stepTimes.end(), history.end() - std::ptrdiff_t(take), history.end());
            // The extract of those steps (S19b; its own scope inside phase 8)
            const std::vector<double> extract = wp.scopeHistory(wp.scope("step/phase:extract/extract"));
            const size_t takeExtract          = size_t(std::min<uint64_t>(n, extract.size()));
            g.extractTimes.insert(g.extractTimes.end(), extract.end() - std::ptrdiff_t(takeExtract), extract.end());
        }
        if (frame == lastMeas && g.stepSamplesAtStart >= 0 && !g.stepTaken)
        {
            const auto s  = Vulkan::WorldBenchmark::summarize(g.stepTimes);
            g.stepStat    = json11::Json::object{{"median", s.median}, {"p95", s.p95}};
            const auto x  = Vulkan::WorldBenchmark::summarize(g.extractTimes);
            g.extractStat = json11::Json::object{{"median", x.median}, {"p95", x.p95}};
            g.stepTaken   = true;
        }
        if (frame > lastWarm && frame <= lastMeas) g.gpuFramesPending.insert(frame);
        // The app's GPU samples of the measured frames (they arrive a few frames later); collected only after the
        // measured frames, outside of the timed recording: the app's log (kGpuLogSize = 600 + 64 frames) still holds
        // every measured frame then, the rest arrive during the drain
        std::vector<std::pair<std::string, double>> sample;
        const bool collect = frame > lastMeas;
        for (auto it = g.gpuFramesPending.begin(); collect && it != g.gpuFramesPending.end();)
        {
            if (!app.gpuFrameSample(*it, sample))
            {
                ++it;
                continue;
            }
            std::map<std::string, double> sections;
            for (const auto& v : sample)
            {
                // gpu and gpu/<section> (the sub-sections are flat names with a dot)
                const std::string& n = v.first;
                if (n == "gpu")
                    g.gpuFrameMs["gpu"].push_back(v.second);
                else if (n.rfind("gpu/", 0) == 0 && n.find('/', 4) == std::string::npos)
                {
                    g.gpuFrameMs[n.substr(4)].push_back(v.second);
                    sections[n.substr(4)] = v.second;
                }
            }
            Vulkan::WorldBenchmark::Sample gpuSample;
            gpuSample.frame     = *it;
            gpuSample.sectionMs = s03Passes(sections);
            g.bench->addSample(gpuSample);
            it = g.gpuFramesPending.erase(it);
        }
        if (frame > lastMeas) g.gpuFramesWaited++;
    }
    // The app's GPU samples of the measured frames, or 16 frames more (a frame without timestamps has none)
    const bool gpuFramesIn = g.gpuFramesPending.empty() || g.gpuFramesWaited > 16;
    if (g.bench && g.bench->finished() && gpuFramesIn && !g.benchmarkDone)
    {
        const std::string file = resultFile(g.benchmarkName);
        if (g.bench->width != 1920 || g.bench->height != 1080)
            std::cout << "Benchmark warning: rendered at " << g.bench->width << " x " << g.bench->height
                      << " instead of 1920 x 1080 (not comparable with the baseline)" << std::endl;
        if (g.sizeChanged) std::cout << "Benchmark error: the rendered size changed during the run" << std::endl;
        const bool complete = g.bench->complete();
        if (!complete) std::cout << "Benchmark error: CPU samples of measured frames are missing" << std::endl;
        json11::Json::object result = g.bench->result().object_items();
        // The engine's numbers of exactly the measured frames (S13): the steps and the command recording and submit
        // (without the waits for the frame slot and the image): cpu_work_ms, RenderScene's lastRecordingMs of the
        // submitted measured frames (the app profile's frame/render/recording has a 0 for an app loop without a frame)
        result["step"]      = g.stepStat;
        result["extract"]   = g.extractStat;
        result["recording"] = result["cpu_work_ms"];
        // The whole GPU frame and RenderScene's sections and sub-sections of the measured frames
        json11::Json::object gpuFrame;
        for (const auto& [section, values] : g.gpuFrameMs)
        {
            const auto s      = Vulkan::WorldBenchmark::summarize(values);
            gpuFrame[section] = json11::Json::object{{"median", s.median}, {"p95", s.p95}};
            // The whole GPU frame also with mean, p99 and max (the summary's GPU row)
            if (section == "gpu")
                gpuFrame[section] = json11::Json::object{
                    {"mean", s.mean}, {"median", s.median}, {"p95", s.p95}, {"p99", s.p99}, {"max", s.max}};
        }
        result["gpu_frame_ms"] = gpuFrame.empty() ? json11::Json() : json11::Json(gpuFrame);
        const int stalls       = result["stalled_frames"].int_value();
        if (stalls > 0)
            std::cout << "Benchmark error: " << stalls
                      << " stalled frames (e.g. the window was minimized or the swap chain recreated)" << std::endl;
        if (result["gpu_ms"].is_null())
            std::cout << "Benchmark warning: no GPU times (timestamps not supported)" << std::endl;
        const bool written = Vulkan::WorldBenchmark::writeResult(file, result);
        // A readable summary; the JSON (the baseline format) is in the file
        // The committed baseline of this GPU, if the source tree is known
        json11::Json baseline;
        if (std::string(SAIGA_PROJECT_SOURCE_DIR).size() > 0)
        {
            std::ifstream in(std::string(SAIGA_PROJECT_SOURCE_DIR) + "/data/tests/benchmarks.json");
            std::stringstream text;
            text << in.rdbuf();
            std::string parseError;
            const json11::Json all = json11::Json::parse(text.str(), parseError);
            baseline               = all[result["gpu"].string_value()][g.benchmarkName];
        }
        std::cout << "Benchmark " << g.benchmarkName << ": " << Vulkan::WorldBenchmark::summaryText(result, baseline)
                  << (written ? "Benchmark result: " + file : "Benchmark error: could not write " + file) << std::endl;
        g.benchmarkOk   = written && complete && !g.sizeChanged && stalls == 0;
        g.benchmarkDone = true;
        app.quit(g.benchmarkOk ? 0 : 1);
    }
}

void gui(EngineApp& app)
{
    WorldGame& g = *g_game;
    // F toggles the walk mode (when ImGui does not have the keyboard)
    const bool fDown = !ImGui::GetIO().WantCaptureKeyboard && ImGui::IsKeyDown(ImGuiKey_F);
    if (fDown && !g.fWasDown) g.walkMode = !g.walkMode;
    g.fWasDown = fDown;
    g.world.imgui();
    ImGui::Begin("World");
    ImGui::Checkbox("walk mode (F)", &g.walkMode);
    ImGui::SliderFloat("eye height", &g.eyeHeight, 0.5f, 50.0f);
    ImGui::SliderFloat("speed", &app.cameraSpeed(), 1, 200);
    ImGui::SliderFloat("speed fast", &app.cameraSpeedFast(), 10, 2000);
    const Vec3 p = app.viewCamera().position;
    ImGui::Text("camera %.1f %.1f %.1f (%.1f m above ground)", p(0), p(1), p(2),
                p(1) - double(g.world.heightField().height(float(p(0)), float(p(2)))));
    if (app.mcp().running()) ImGui::Text("MCP: http://127.0.0.1:%d/mcp", app.mcp().port());
    ImGui::End();
}
}  // namespace

int main(int argc, char* argv[])
{
    WorldGame game;
    g_game = &game;

    EngineAppSetup setup;
    setup.name               = "world";
    setup.render.msaaSamples = 4;
    setup.cameraSpeed        = 25;
    setup.cameraSpeedFast    = 250;
    setup.benchmarks         = Vulkan::WorldBenchmark::names();
    setup.mcpInstructions    = Vulkan::worldMcpInstructions();
    EngineApp* appPtr        = nullptr;
    setup.renderHooks.init   = [&](EngineApp& app)
    {
        appPtr             = &app;
        game.benchmarkName = app.options().benchmark;
        initWorld(app);
    };
    setup.renderHooks.shutdown     = [&](EngineApp&) { game.world.destroy(); };
    setup.renderHooks.beforeRender = [&](EngineApp& app, double) { beforeRender(app); };
    setup.renderHooks.gui          = [&]() { gui(*appPtr); };
    setup.gameTools                = [&](McpServer& server, EngineApp& app)
    {
        if (!app.scene()) return;
        Vulkan::WorldMcpContext context;
        context.world     = &game.world;
        context.getCamera = [&app]()
        {
            Vulkan::WorldMcpCamera c;
            c.position = app.viewCamera().position;
            c.rotation = app.viewCamera().rotation;
            c.fovy     = app.viewCamera().fovy;
            return c;
        };
        // A tool write without a hold (game_set_camera moves the camera this game renders from)
        context.setCamera = [&app](const Vulkan::WorldMcpCamera& c)
        { app.setViewCameraByTool(app.viewCamera().space, c.position, c.rotation, c.fovy, false); };
        context.walkMode  = &game.walkMode;
        context.eyeHeight = &game.eyeHeight;
        Vulkan::registerWorldMcpTools(server, context);
    };

    const int code = engineMain(argc, argv, setup, {},
                                [&](EngineApp& app, const AppOptions& options)
                                {
                                    if (options.benchmark.empty()) app.showStatistics(true);  // the frame times
                                });
    // A window closed before the end of the path
    if (code == 0 && !game.benchmarkName.empty() && !game.benchmarkDone)
    {
        std::cout << "Benchmark error: did not finish" << std::endl;
        return 1;
    }
    return code;
}
