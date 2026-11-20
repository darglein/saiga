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

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>

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
    Vulkan::WorldRenderer::FrameTimes lastTimes;
    bool hasTimes = false;
    // The world profiler's sample count when the measured frames began (the benchmark's step statistic)
    int64_t stepSamplesAtStart = -1;
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
    g.world.init(app.scene()->forwardRenderer(), params);
    g.world.frameCallbacks.push_back(
        [](const Vulkan::WorldRenderer::FrameTimes& t)
        {
            g_game->lastTimes = t;
            g_game->hasTimes  = true;
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
    g.world.transfer(cmd, camera.view, camera.getPosition(), float(size(0)) / float(std::max(1, size(1))));
    if (g.bench && g.stepSamplesAtStart < 0 && g.bench->measuring())
        g.stepSamplesAtStart = int64_t(app.world().profiler().sampleCount());
    if (g.bench && g.bench->finished() && !g.benchmarkDone)
    {
        const std::string file = resultFile(g.benchmarkName);
        if (g.bench->width != 1920 || g.bench->height != 1080)
            std::cout << "Benchmark warning: rendered at " << g.bench->width << " x " << g.bench->height
                      << " instead of 1920 x 1080 (not comparable with the baseline)" << std::endl;
        if (g.sizeChanged) std::cout << "Benchmark error: the rendered size changed during the run" << std::endl;
        json11::Json::object result = g.bench->result().object_items();
        // The engine's numbers: the step and the frame recording (S13)
        auto stat    = [](const Profiler::Stats& s) { return json11::Json::object{{"median", s.p50}, {"p95", s.p95}}; };
        Profiler& wp = app.world().profiler();
        // The steps of the measured frames (not the last 600 steps: at hundreds of frames per second they reach back
        // into the warm-up)
        const int stepSamples =
            int(std::max<int64_t>(1, int64_t(wp.sampleCount()) - std::max<int64_t>(0, g.stepSamplesAtStart)));
        result["step"]      = stat(wp.scopeStats(wp.scope("step"), stepSamples));
        result["recording"] = stat(app.frameProfile().scopeStats(app.frameProfile().scope("frame/render"),
                                                                 Vulkan::WorldBenchmark::measuredFrames));
        const int stalls    = result["stalled_frames"].int_value();
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
    setup.mcpInstructions    = Vulkan::worldMcpInstructions(false);
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
    setup.renderHooks.passTimes = [&](std::vector<std::pair<std::string, double>>& passes)
    {
        if (!game.hasTimes) return false;
        game.hasTimes = false;
        // No completed GPU frame or no timestamps: no samples (zeros would lower the statistics)
        if (game.lastTimes.gpuMs < 0 || game.lastTimes.gpuPassMs.empty()) return false;
        for (int i = 0; i < Vulkan::WorldRenderer::PASS_COUNT; ++i)
            passes.push_back({Vulkan::WorldRenderer::gpuPassName(i), game.lastTimes.gpuPassMs[size_t(i)]});
        return true;
    };
    setup.gameTools = [&](McpServer& server, EngineApp& app)
    {
        if (!app.scene()) return;
        Vulkan::WorldMcpContext context;
        context.renderer           = &app.scene()->forwardRenderer();
        context.world              = &game.world;
        context.camera             = &app.camera();
        context.walkMode           = &game.walkMode;
        context.eyeHeight          = &game.eyeHeight;
        context.measurePerformance = false;
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
