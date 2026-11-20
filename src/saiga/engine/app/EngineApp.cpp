/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// EngineApp: the frame loop of the engine (docs/engine_design.md §8.3; S13).

#include "saiga/engine/app/EngineApp.h"

#include "saiga/core/framework/framework.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/keyboard.h"
#include "saiga/core/util/mouse.h"
#include "saiga/engine/app/AppTools.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/PlayerInput.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/tools/WorldTools.h"
#include "saiga/saiga_buildconfig.h"
#include "saiga/saiga_modules.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/window/StandaloneWindow.h"
#include "saiga/vulkan/window/Window.h"

#ifdef SAIGA_USE_GLFW
#    include "saiga/core/glfw/glfw_camera.h"
#endif

#include <fstream>
#include <stdexcept>
#include <thread>

namespace Saiga
{
namespace Engine
{
struct EngineApp::CameraHolder
{
#ifdef SAIGA_USE_GLFW
    Glfw_Camera<PerspectiveCamera> camera;
#else
    PerspectiveCamera camera;
#endif
};

namespace
{
SaigaParameters saigaParameters()
{
    // As initSaigaSample, without config.ini
    SaigaParameters p;
    p.shaderDirectory.push_back(SAIGA_PROJECT_SOURCE_DIR "/shader");
    const std::string data = SAIGA_PROJECT_SOURCE_DIR "/data";
    p.textureDirectory     = {data, data + "/textures"};
    p.modelDirectory       = {data, data + "/models"};
    p.fontDirectory        = {data, data + "/fonts"};
    p.dataDirectory        = {data};
    return p;
}

bool fileExists(const std::string& path)
{
    return !path.empty() && std::ifstream(path).good();
}
}  // namespace

EngineApp::EngineApp(const EngineAppSetup& setup, const AppOptions& options)
    : appSetup(setup),
      appOptions(options),
      cameraHolder(std::make_unique<CameraHolder>()),
      frameProfiler(600),
      gpuProfiler(600)
{
    if (!appOptions.benchmark.empty())
    {
        // A benchmark measures the world only: a fixed size, no vsync, validation, GUI or MCP (§17.7)
        appSetup.headless = false;
        appSetup.noRender = false;
        appSetup.width = 1920, appSetup.height = 1080;
        appSetup.vsync = false, appSetup.validation = false;
        appSetup.render.imgui = false;
        appSetup.mcpPort      = 0;
        // The automatic screenshot (SAIGA_SCREENSHOT) would end the run at frame 60
        if (!appOptions.screenshot.empty()) SAIGA_LOG("app", "benchmark: the screenshot is ignored");
        appOptions.screenshot.clear();
    }
    SAIGA_ASSERT(!appSetup.noRender || appSetup.headless, "EngineApp: noRender needs headless");
    initSaiga(saigaParameters());

    gameWorld = std::make_unique<World>(appSetup.world);

    if (!appSetup.noRender)
    {
        WindowParameters wp;
        wp.name            = appSetup.name;
        wp.backend         = appSetup.headless ? WindowParameters::Backend::Headless : WindowParameters::Backend::GLFW;
        wp.width           = appSetup.width;
        wp.height          = appSetup.height;
        wp.mode            = WindowParameters::Mode::Windowed;
        wp.vsync           = appSetup.vsync;
        wp.saigaParameters = saigaParameters();
        wp.imguiParameters.enable = appSetup.render.imgui;
        appWindow                 = Vulkan::CreateVulkanWindow(wp);
        if (!appWindow) throw std::runtime_error("EngineApp: this build has no window backend (GLFW) for a window");
        appWindow->markRunning();
        Vulkan::VulkanParameters vp;
        vp.enableValidationLayer = appSetup.validation;
        vp.vsync                 = appSetup.vsync;
        vp.enableImgui           = appSetup.render.imgui;
        vp.enableDefragmentation = false;
        renderScene              = std::make_unique<Vulkan::RenderScene>(*appWindow, vp, appSetup.render);
    }

    PerspectiveCamera& cam = cameraHolder->camera;
    cam.setProj(60.0f, float(appSetup.width) / float(appSetup.height), 0.1f, 50.0f, true);
    cam.setView(vec3(0, 2, 6), vec3(0, 0, 0), vec3(0, 1, 0));
    if (appWindow) appWindow->setCamera(&cam);

    inputMap.setActions(gameWorld->actionList());
    // The input bindings (fatal on errors, R-DATA-1)
    std::string inputFile = appSetup.inputFile;
    if (inputFile.empty() && !appSetup.world.dataDirectory.empty())
    {
        const std::string candidate = appSetup.world.dataDirectory + "/input.json";
        if (fileExists(candidate)) inputFile = candidate;
    }
    if (!inputFile.empty())
    {
        std::string error;
        if (!inputMap.load(inputFile, "input.user.json", gameWorld->actionList(), error))
            SAIGA_EXIT_ERROR("input bindings: " + error);
    }

    // MCP: the server and the tools always exist (tests call handleMessage); the server listens only with a port
    server                     = std::make_unique<McpServer>(appSetup.name, "1.0");
    server->toolTimeoutSeconds = appSetup.mcpToolTimeoutSeconds;
    WorldToolsOptions toolOptions;
    toolOptions.snapshotDirectory = appSetup.snapshotDirectory;
    worldApi                      = registerWorldTools(*server, *gameWorld, toolOptions);
    worldApi->addProfile("frame", &frameProfiler);
    worldApi->addProfile("gpu", &gpuProfiler);
    registerAppTools(*server, *this);
    if (appSetup.gameTools) appSetup.gameTools(*server, *this);
    if (appSetup.mcpPort > 0)
    {
        if (server->start(appSetup.mcpPort))
            SAIGA_LOG("app", "MCP server: http://127.0.0.1:%d/mcp", server->port());
        else
            SAIGA_LOG("app", "MCP server: could not listen on port %d", appSetup.mcpPort);
    }

    scopeFrame     = frameProfiler.scope("frame");
    scopePoll      = frameProfiler.scope("frame/poll");
    scopeAdvance   = frameProfiler.scope("frame/advance");
    scopeRender    = frameProfiler.scope("frame/render");
    counterFps     = frameProfiler.counter("fps");
    counterSteps   = frameProfiler.counter("steps_per_frame");
    counterDropped = frameProfiler.counter("dropped_steps");
    scopeGpuFrame  = gpuProfiler.scope("gpu");
    for (const char* name : Vulkan::RenderScene::gpuSectionNames())
        scopeGpuSections.push_back(gpuProfiler.scope(std::string("gpu/") + name));

    // The local player survives a load (its handle is kept); otherwise the first player of peer 0, or none
    gameWorld->addLoadListener(
        [this](const LoadInfo&)
        {
            World& w = *gameWorld;
            if (localPlayerEntity.valid() && w.alive(localPlayerEntity)) return;
            localPlayerEntity = Entity{};
            if (!w.registry().registered(Player::id)) return;
            w.viewAll<const Player>().each(
                [&](Entity e, const Player& p)
                {
                    if (!localPlayerEntity.valid() && p.peer == 0) localPlayerEntity = e;
                });
        });

    if (renderScene && appSetup.renderHooks.init)
    {
        appSetup.renderHooks.init(*this);
        hooksInitialized = true;
    }
}

EngineApp::~EngineApp()
{
    // MCP first (no handler may run on a half-destroyed app), then the GPU work, the game's GPU state, the frame, the
    // world, the window
    if (server) server->stop();
    if (renderScene) renderScene->waitIdle();
    if (hooksInitialized && appSetup.renderHooks.shutdown) appSetup.renderHooks.shutdown(*this);
    renderScene.reset();
    worldApi.reset();
    server.reset();
    gameWorld.reset();
    appWindow.reset();
}

PerspectiveCamera& EngineApp::camera()
{
    return cameraHolder->camera;
}

void EngineApp::quit(int code)
{
    quitRequested = true;
    exitStatus    = code;
}

void EngineApp::setVirtualInput(const VirtualInput& input, int frames)
{
    virtualInput  = input;
    virtualFrames = frames;
    hasVirtual    = true;
}

int EngineApp::run()
{
    const double stepDt = gameWorld->dt();
    while (!quitRequested)
    {
        const auto start = std::chrono::steady_clock::now();
        frame();
        if (appWindow && appWindow->shouldClose() && !appSetup.headless) break;
        // Headless (or no frame presented): paced to the step rate, MCP and the world keep going
        if (appSetup.headless || !renderingEnabled)
            std::this_thread::sleep_until(start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                                      std::chrono::duration<double>(stepDt)));
    }
    return exitStatus;
}

void EngineApp::runFrames(int count, double dt)
{
    for (int i = 0; i < count && !quitRequested; ++i) frame(dt);
}

void EngineApp::frame()
{
    const auto now = std::chrono::steady_clock::now();
    double dt      = firstFrame ? 0.0 : std::chrono::duration<double>(now - lastFrame).count();
    firstFrame     = false;
    lastFrame      = now;
    frame(dt);
}

void EngineApp::frame(double dt)
{
    framesRun++;
    {
        Profiler::Timer frameTimer(frameProfiler, scopeFrame);
        {
            Profiler::Timer t(frameProfiler, scopePoll);
            pollInput(dt);
            server->poll();
        }
        applyPlayerInput();
        {
            Profiler::Timer t(frameProfiler, scopeAdvance);
            const uint64_t before = gameWorld->stepIndex();
            gameWorld->advance(dt);
            frameProfiler.setCounter(counterSteps, double(gameWorld->stepIndex() - before));
            frameProfiler.setCounter(counterDropped, double(gameWorld->droppedSteps() - droppedBefore));
            droppedBefore = gameWorld->droppedSteps();
        }
        if (renderScene && renderingEnabled)
        {
            Profiler::Timer t(frameProfiler, scopeRender);
            renderFrame();
        }
    }
    frameProfiler.setCounter(counterFps, dt > 0 ? 1.0 / dt : 0.0);
    frameProfiler.endSample();
    if (renderScene) collectGpuTimes();
    if (appWindow && appWindow->shouldClose()) quitRequested = true;
}

void EngineApp::pollInput(double dt)
{
    if (appWindow) appWindow->update(float(dt));
}

void EngineApp::applyPlayerInput()
{
    // The devices (a window with GLFW input; a headless app has none)
    InputState state;
    bool captured = false;
    if (appWindow && !appSetup.headless)
    {
        for (int k = 0; k < int(state.keys.size()); ++k) state.keys[size_t(k)] = keyboard().getKeyState(k) != 0;
        for (int b = 0; b < int(state.mouseButtons.size()); ++b)
            state.mouseButtons[size_t(b)] = mouse.getKeyState(b) != 0;
        const Vec2 m(double(mouse.getX()), double(mouse.getY()));
        if (mouseKnown) state.mouseDelta = m - lastMouse;
        lastMouse  = m;
        mouseKnown = true;
        if (ImGui::GetCurrentContext()) captured = ImGui::GetIO().WantCaptureKeyboard;
        // The app's keys: ESC quits, G toggles ImGui, F1 the statistics window, F12 saves screenshot.png
        auto pressedKey = [&](int k) { return !captured && state.keys[size_t(k)] && !previousKeys[size_t(k)]; };
        if (pressedKey(256)) quit(0);
        if (pressedKey('G') && renderScene) renderScene->setImguiVisible(!renderScene->imguiVisible());
        if (pressedKey(290)) statisticsVisible = !statisticsVisible;
        if (pressedKey(301) && renderScene)
            renderScene->requestScreenshot(
                [](const TemplatedImage<ucvec4>& img)
                {
                    if (img.width > 0 && img.save("screenshot.png")) SAIGA_LOG("app", "saved screenshot.png");
                });
        previousKeys = state.keys;
    }
    if (hasVirtual)
    {
        inputMap.update(state, captured, virtualInput);
        // Pressed actions are edges of one frame
        virtualInput.pressed.clear();
        if (virtualFrames > 0 && --virtualFrames == 0)
        {
            hasVirtual   = false;
            virtualInput = VirtualInput();
        }
    }
    else
        inputMap.update(state, captured);

    World& w = *gameWorld;
    if (localPlayerEntity.valid() && !w.alive(localPlayerEntity)) localPlayerEntity = Entity{};
    PerspectiveCamera& cam = cameraHolder->camera;
    if (!localPlayerEntity.valid())
    {
        // The free camera (WASD and mouse) when no player is driven
#ifdef SAIGA_USE_GLFW
        if (appWindow && !appSetup.headless && !captured)
        {
            cameraHolder->camera.update(1.0f / 60.0f);
            cameraHolder->camera.interpolate(1.0f / 60.0f, 0);
        }
#endif
        return;
    }
    // Neutral device input does not overwrite what an agent set (set_component); a changed or active input does
    const bool active = !inputMap.neutral();
    if (w.has<PlayerInput>(localPlayerEntity) && active)
    {
        // The move axes relative to the camera's heading
        const vec3 f = cam.rot * vec3(0, 0, -1);
        Vec3 forward = Vec3(f.x(), 0, f.z());
        forward      = forward.norm() > 1e-9 ? Vec3(forward.normalized()) : Vec3(0, 0, -1);
        const Vec3 right(-forward.z(), 0, forward.x());
        const Vec2 m    = inputMap.move();
        PlayerInput& in = w.get<PlayerInput>(localPlayerEntity);
        in.move         = right * m.x() + forward * m.y();
        in.look         = Quat(cam.rot.w(), cam.rot.x(), cam.rot.y(), cam.rot.z());
        in.heldButtons  = inputMap.heldBits();
    }
    for (uint32_t id : inputMap.pressedActions())
        if (w.has<Player>(localPlayerEntity)) w.queueAction(PlayerAction{localPlayerEntity, id, Vec3::Zero()});
    if (appSetup.applyInput) appSetup.applyInput(inputMap, w, localPlayerEntity);
}

void EngineApp::renderFrame()
{
    const GameRenderHooks& h = appSetup.renderHooks;
    auto transfer            = [&](vk::CommandBuffer cmd)
    {
        if (h.transfer) h.transfer(cmd);
    };
    auto render = [&](vk::CommandBuffer cmd)
    {
        if (h.render) h.render(cmd);
    };
    auto gui = [&]()
    {
        if (h.gui) h.gui();
        if (statisticsVisible) statisticsWindow();
    };
    if (!appOptions.screenshot.empty() && framesRendered + 1 == uint64_t(appOptions.screenshotFrame))
    {
        const std::string file = appOptions.screenshot;
        renderScene->requestScreenshot(
            [this, file](const TemplatedImage<ucvec4>& img)
            {
                if (img.width > 0 && img.save(file))
                    SAIGA_LOG("app", "saved screenshot %s", file.c_str());
                else
                    SAIGA_LOG("app", "could not save the screenshot %s", file.c_str());
                quit(img.width > 0 ? 0 : 1);
            });
    }
    if (!renderScene->renderFrame(transfer, render, gui)) return;
    framesRendered++;
    for (size_t i = 0; i < frameCallbacks.size();)
    {
        // A callback may add callbacks
        auto cb = frameCallbacks[i];
        if (cb())
            frameCallbacks.erase(frameCallbacks.begin() + std::ptrdiff_t(i));
        else
            ++i;
    }
}

void EngineApp::statisticsWindow()
{
    // The prototype's "World Statistics" for the engine: frame and step time plots, the step profile per phase and
    // system, the GPU, the counters (docs/engine_design.md §17.1)
    ImGui::SetNextWindowSize(ImVec2(520, 560), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Engine statistics", &statisticsVisible))
    {
        ImGui::End();
        return;
    }
    auto plot = [](const char* label, const std::vector<double>& values)
    {
        std::vector<float> f(values.begin(), values.end());
        double maxV = 0;
        for (double v : values) maxV = std::max(maxV, v);
        ImGui::PlotLines(label, f.data(), int(f.size()), 0, nullptr, 0.0f, float(maxV * 1.1 + 1e-3), ImVec2(0, 60));
    };
    const Profiler::Stats fs = frameProfiler.scopeStats(scopeFrame, 120);
    ImGui::Text("frame %.2f ms (p95 %.2f), %.0f fps", fs.p50, fs.p95, fs.avg > 0 ? 1000.0 / fs.avg : 0.0);
    plot("frame ms", frameProfiler.scopeHistory(scopeFrame));
    Profiler& wp = gameWorld->profiler();
    plot("step ms", wp.scopeHistory(wp.scope("step")));
    const Profiler::Stats gs = gpuProfiler.scopeStats(scopeGpuFrame, 120);
    ImGui::Text("GPU %.3f ms (p95 %.3f)", gs.p50, gs.p95);
    const json11::Json report = wp.report(120);
    if (ImGui::BeginTable("scopes", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY, ImVec2(0, 260)))
    {
        ImGui::TableSetupColumn("scope");
        ImGui::TableSetupColumn("avg ms");
        ImGui::TableSetupColumn("p95 ms");
        ImGui::TableSetupColumn("share");
        ImGui::TableHeadersRow();
        for (const auto& sc : report["scopes"].array_items())
        {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(sc["scope"].string_value().c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", sc["avg"].number_value());
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", sc["p95"].number_value());
            ImGui::TableNextColumn();
            ImGui::Text("%.0f%%", sc["share_of_parent"].number_value() * 100);
        }
        ImGui::EndTable();
    }
    for (const auto& c : report["counters"].array_items())
        ImGui::Text("%s: %.0f", c["counter"].string_value().c_str(), c["avg"].number_value());
    ImGui::End();
}

void EngineApp::collectGpuTimes()
{
    Vulkan::FrameTimes times;
    while (renderScene->takeGpuFrameTimes(times))
    {
        gpuProfiler.add(scopeGpuFrame, times.gpuMs);
        for (size_t i = 0; i < times.gpuSectionMs.size() && i < scopeGpuSections.size(); ++i)
            gpuProfiler.add(scopeGpuSections[i], times.gpuSectionMs[i]);
        std::vector<std::pair<std::string, double>> passes;
        if (appSetup.renderHooks.passTimes && appSetup.renderHooks.passTimes(passes))
            for (const auto& p : passes) gpuProfiler.add(gpuProfiler.scope("gpu/game/" + p.first), p.second);
        gpuProfiler.endSample();
    }
}

}  // namespace Engine
}  // namespace Saiga
