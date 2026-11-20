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
 * vsync, GUI and MCP off, frozen time, the default world: SAIGA_WORLD_TREES is ignored), writes
 * benchmark_<name>.json to the working directory and exits with 0, or with 1 if the benchmark did not finish, the
 * result could not be written, the rendered size changed or a frame stalled (an old result file is deleted at the
 * start). --headless runs without a window (agents, servers).
 */

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/engine/app/EngineApp.h"
#include "saiga/engine/app/EngineMain.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/world/WorldBenchmark.h"
#include "saiga/vulkan/world/WorldMcp.h"
#include "saiga/vulkan/world/WorldRenderer.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <set>

using namespace Saiga;
using namespace Saiga::Engine;

namespace
{
// The game's state outside the World: the prototype renderer and its camera controls (render state, not world state)
struct WorldGame
{
    Vulkan::WorldRenderer world;
    std::unique_ptr<Vulkan::WorldBenchmark> bench;
    std::string benchmarkName;
    bool benchmarkDone = false, benchmarkOk = false, sizeChanged = false;
    bool walkMode   = false;
    float eyeHeight = 1.8f;
    bool fWasDown   = false;
    // GPU pass times of the newest frame of the world renderer (for the engine's gpu profile)
    // The world renderer's frame -> the RenderScene frame it was recorded in, and the world passes by RenderScene frame
    // (the app matches them to the frame's completion)
    std::map<int, uint64_t> sceneFrameOf;
    std::map<uint64_t, std::array<double, Vulkan::WorldRenderer::PASS_COUNT>> passesOf;
    // The engine's numbers of the measured frames (step, recording), taken at the frame boundaries that close them
    int64_t stepSamplesAtStart = -1, frameSamplesAtStart = -1;
    // The step times of the measured range, collected per frame (the profile keeps only the last 600 steps: at 30 fps
    // the 600 measured frames have about 1200 steps)
    std::vector<double> stepTimes;
    uint64_t stepSamplesSeen = 0;
    bool stepTaken = false, recordingTaken = false;
    json11::Json::object stepStat, recordingStat;
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

void initWorld(EngineApp& app)
{
    WorldGame& g = *g_game;
    Vulkan::WorldParameters params;
    // A benchmark always uses the default world (its result is compared with the baseline)
    if (const char* trees = std::getenv("SAIGA_WORLD_TREES"); trees && g.benchmarkName.empty())
        params.treeCount = std::atoi(trees);
    params.reversedZ = true;
    g.world.init(app.scene()->base(), app.scene()->formats(), params);
    g.world.frameCallbacks.push_back(
        [](const Vulkan::WorldRenderer::FrameTimes& t)
        {
            WorldGame& g = *g_game;
            if (t.gpuMs < 0) return false;
            auto it = g.sceneFrameOf.find(t.gpuFrame);
            if (it == g.sceneFrameOf.end()) return false;
            g.passesOf[it->second] = t.gpuPassMs;
            g.sceneFrameOf.erase(g.sceneFrameOf.begin(), std::next(it));
            while (g.passesOf.size() > 32) g.passesOf.erase(g.passesOf.begin());
            return false;
        });

    // Start on a hill: next to the highest point, looking over the world center
    PerspectiveCamera& camera = app.camera();
    auto& hf                  = g.world.heightField();
    vec3 peak                 = hf.highestPoint();
    vec3 center               = vec3(hf.worldSize() * 0.5f, 0, hf.worldSize() * 0.5f);
    vec3 toPeak               = vec3(peak(0) - center(0), 0, peak(2) - center(2));
    vec3 eye                  = peak + (toPeak.norm() > 1 ? toPeak.normalized() * 150.0f : vec3(150, 0, 0));
    eye(1)                    = std::max(hf.height(eye(0), eye(2)), peak(1) - 40) + 60;
    center(1)                 = hf.height(center(0), center(2));
    camera.setView(eye, center, vec3(0, 1, 0));

    if (!g.benchmarkName.empty())
    {
        g.world.settings.freezeTime = true;
        std::string gpu             = app.scene()->base().physicalDevice.getProperties().deviceName.data();
        g.bench                     = std::make_unique<Vulkan::WorldBenchmark>(g.benchmarkName, g.world, gpu);
        std::remove(resultFile(g.benchmarkName).c_str());
        std::cout << "Benchmark " << g.benchmarkName << " on " << gpu << std::endl;
    }
}

void transfer(EngineApp& app, vk::CommandBuffer cmd)
{
    WorldGame& g              = *g_game;
    PerspectiveCamera& camera = app.camera();
    if (g.walkMode)
    {
        vec3 p          = camera.getPosition();
        p(1)            = g.world.heightField().height(p(0), p(2)) + g.eyeHeight;
        camera.position = make_vec4(p, 1);
        camera.calculateModel();
        camera.updateFromModel();
    }
    if (g.bench)
    {
        auto pose = g.bench->currentPose();
        camera.setView(pose.eye, pose.target, vec3(0, 1, 0));
        if (g.bench->measuring())
        {
            ivec2 size = app.scene()->size();
            if (g.bench->width != 0 && (size(0) != g.bench->width || size(1) != g.bench->height)) g.sizeChanged = true;
            g.bench->width  = size(0);
            g.bench->height = size(1);
        }
    }
    const ivec2 size = app.scene()->size();
    // The scene pass of this frame: linear and pre-exposed with a tone map, display values in clay mode (S17)
    g.world.setOutputLinear(app.scene()->sceneWritesLinear(), app.scene()->preExposure());
    g.world.transfer(cmd, camera.view, camera.getPosition(), float(size(0)) / float(std::max(1, size(1))));
    // This frame's number in both renderers (the RenderScene frame being recorded is frames() + 1)
    g.sceneFrameOf[g.world.frameNumber()] = app.scene()->frames() + 1;
    while (g.sceneFrameOf.size() > 32) g.sceneFrameOf.erase(g.sceneFrameOf.begin());
    if (g.bench)
    {
        // Frame k is the world renderer's frame number after its transfer (W: the last warm-up frame, M1..Mn: the
        // measured frames). In a frame the steps run before transfer; its recording sample is added after transfer
        // and counted when the frame ends.
        const int frame    = g.world.frameNumber();
        const int lastWarm = Vulkan::WorldBenchmark::warmupFrames;
        const int lastMeas = lastWarm + Vulkan::WorldBenchmark::measuredFrames;
        auto stat    = [](const Profiler::Stats& s) { return json11::Json::object{{"median", s.p50}, {"p95", s.p95}}; };
        Profiler& wp = app.world().profiler();
        Profiler& fp = app.frameProfile();
        // Steps: from after W's steps (transfer of W) to after Mn's steps (transfer of Mn)
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
        }
        if (frame == lastMeas && g.stepSamplesAtStart >= 0 && !g.stepTaken)
        {
            const auto s = Vulkan::WorldBenchmark::summarize(g.stepTimes);
            g.stepStat   = json11::Json::object{{"median", s.median}, {"p95", s.p95}};
            g.stepTaken  = true;
        }
        // Recordings: the frames completed from the transfer of M1 (through W) to the transfer of Mn+1 (through Mn)
        if (frame > lastWarm && frame <= lastMeas) g.gpuFramesPending.insert(app.scene()->frames() + 1);
        // The app's GPU samples of the measured frames (they arrive a few frames later)
        std::vector<std::pair<std::string, double>> sample;
        for (auto it = g.gpuFramesPending.begin(); it != g.gpuFramesPending.end();)
        {
            if (!app.gpuFrameSample(*it, sample))
            {
                ++it;
                continue;
            }
            for (const auto& v : sample)
            {
                // gpu and gpu/<section> (not the game's passes below a section, they are in gpu_ms)
                const std::string& n = v.first;
                if (n == "gpu")
                    g.gpuFrameMs["gpu"].push_back(v.second);
                else if (n.rfind("gpu/", 0) == 0 && n.find('/', 4) == std::string::npos)
                    g.gpuFrameMs[n.substr(4)].push_back(v.second);
            }
            it = g.gpuFramesPending.erase(it);
        }
        if (frame > lastMeas) g.gpuFramesWaited++;
        if (frame == lastWarm + 1) g.frameSamplesAtStart = int64_t(fp.sampleCount());
        if (frame == lastMeas + 1 && g.frameSamplesAtStart >= 0 && !g.recordingTaken)
        {
            const int n      = int(std::max<int64_t>(1, int64_t(fp.sampleCount()) - g.frameSamplesAtStart));
            g.recordingStat  = stat(fp.scopeStats(fp.scope("frame/render/recording"), n));
            g.recordingTaken = true;
        }
    }
    // The app's GPU samples of the measured frames, or 16 frames more (a frame without timestamps has none)
    const bool gpuFramesIn = g.gpuFramesPending.empty() || g.gpuFramesWaited > 16;
    if (g.bench && g.bench->finished() && g.recordingTaken && gpuFramesIn && !g.benchmarkDone)
    {
        const std::string file = resultFile(g.benchmarkName);
        if (g.bench->width != 1920 || g.bench->height != 1080)
            std::cout << "Benchmark warning: rendered at " << g.bench->width << " x " << g.bench->height
                      << " instead of 1920 x 1080 (not comparable with the baseline)" << std::endl;
        if (g.sizeChanged) std::cout << "Benchmark error: the rendered size changed during the run" << std::endl;
        json11::Json::object result = g.bench->result().object_items();
        // The engine's numbers of exactly the measured frames (S13): the steps and the command recording and submit
        // (frame/render/recording, without the waits for the frame slot and the image)
        result["step"]      = g.stepStat;
        result["recording"] = g.recordingStat;
        // The whole GPU frame and RenderScene's sections (transfer, scene, resolve, post, imgui) of the measured frames
        json11::Json::object gpuFrame;
        for (const auto& [section, values] : g.gpuFrameMs)
        {
            const auto s      = Vulkan::WorldBenchmark::summarize(values);
            gpuFrame[section] = json11::Json::object{{"median", s.median}, {"p95", s.p95}};
        }
        result["gpu_frame_ms"] = gpuFrame.empty() ? json11::Json() : json11::Json(gpuFrame);
        const int stalls       = result["stalled_frames"].int_value();
        if (stalls > 0)
            std::cout << "Benchmark error: " << stalls
                      << " stalled frames (e.g. the window was minimized or the swap chain recreated)" << std::endl;
        if (result["gpu_ms"].is_null())
            std::cout << "Benchmark warning: no GPU times (timestamps not supported)" << std::endl;
        const bool written = Vulkan::WorldBenchmark::writeResult(file, result);
        std::cout << (written ? "Benchmark result: " + file + "\n" + json11::Json(result).dump()
                              : "Benchmark error: could not write " + file)
                  << std::endl;
        g.benchmarkOk   = written && !g.sizeChanged && stalls == 0;
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
    vec3 p = app.camera().getPosition();
    ImGui::Text("camera %.1f %.1f %.1f (%.1f m above ground)", p(0), p(1), p(2),
                p(1) - g.world.heightField().height(p(0), p(2)));
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
    for (int i = 0; i < Vulkan::WorldRenderer::PASS_COUNT; ++i)
        setup.renderHooks.passNames.push_back(Vulkan::WorldRenderer::gpuPassName(i));
    EngineApp* appPtr      = nullptr;
    setup.renderHooks.init = [&](EngineApp& app)
    {
        appPtr             = &app;
        game.benchmarkName = app.options().benchmark;
        initWorld(app);
    };
    setup.renderHooks.shutdown  = [&](EngineApp&) { game.world.destroy(); };
    setup.renderHooks.transfer  = [&](vk::CommandBuffer cmd) { transfer(*appPtr, cmd); };
    setup.renderHooks.render    = [&](vk::CommandBuffer cmd) { game.world.render(cmd); };
    setup.renderHooks.gui       = [&]() { gui(*appPtr); };
    setup.renderHooks.passTimes = [&](uint64_t frame, std::vector<std::pair<std::string, double>>& passes)
    {
        // Regression (CodeRabbit, S13): one latest-result flag gave the world passes of one frame to another frame's
        // completion, and a second completion in the same poll got zeros
        auto it = game.passesOf.find(frame);
        if (it == game.passesOf.end()) return false;
        for (int i = 0; i < Vulkan::WorldRenderer::PASS_COUNT; ++i)
            passes.push_back({Vulkan::WorldRenderer::gpuPassName(i), it->second[size_t(i)]});
        game.passesOf.erase(it);
        return true;
    };
    setup.gameTools = [&](McpServer& server, EngineApp& app)
    {
        if (!app.scene()) return;
        Vulkan::WorldMcpContext context;
        context.world     = &game.world;
        context.camera    = &app.camera();
        context.walkMode  = &game.walkMode;
        context.eyeHeight = &game.eyeHeight;
        Vulkan::registerWorldMcpTools(server, context);
    };

    const int code = engineMain(argc, argv, setup, {},
                                [&](EngineApp& app, const AppOptions& options)
                                {
                                    if (options.benchmark.empty()) app.showStatistics(false);
                                });
    // A window closed before the end of the path
    if (code == 0 && !game.benchmarkName.empty() && !game.benchmarkDone)
    {
        std::cout << "Benchmark error: did not finish" << std::endl;
        return 1;
    }
    return code;
}
