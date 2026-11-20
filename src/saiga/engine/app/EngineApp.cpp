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
    {
        SaigaParameters p = saigaParameters();
        // Without a device no shader is loaded (a server installed without the source tree)
        p.ensureSaigaShadersExist = !appSetup.noRender;
        initSaiga(p);
    }

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
#ifdef SAIGA_USE_GLFW
    auto& controls             = cameraHolder->camera;
    controls.mode0             = CameraControlMode::ROTATE_FIRST_PERSON_FIX_UP_VECTOR;
    controls.mode1             = CameraControlMode::ROTATE_FIRST_PERSON_FIX_UP_VECTOR;
    controls.movementSpeed     = appSetup.cameraSpeed;
    controls.movementSpeedFast = appSetup.cameraSpeedFast;
#else
    cameraSpeedValue     = appSetup.cameraSpeed;
    cameraSpeedFastValue = appSetup.cameraSpeedFast;
#endif

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
        if (!inputMap.load(inputFile, "input." + appSetup.name + ".user.json", gameWorld->actionList(), error))
            SAIGA_EXIT_ERROR("input bindings: " + error);
    }
    else
        inputMap.setDefaults();

    // MCP: the server and the tools always exist (tests call handleMessage); the server listens only with a port
    // The game's instructions describe its renderer tools; without a device there are none
    const std::string instructions =
        appSetup.noRender || appSetup.mcpInstructions.empty()
            ? "Controls a running engine game (" + appSetup.name +
                  "). The world API tools (describe_world, list_entities, inspect, step, save_snapshot, ...) read and "
                  "change the game world; app_status describes the app and its input bindings, set_input drives the "
                  "local player, measure_performance measures steps" +
                  (appSetup.noRender ? std::string(" (this app does not render)")
                                     : std::string(", frames and the GPU")) +
                  "."
            : appSetup.mcpInstructions;
    server                     = std::make_unique<McpServer>(appSetup.name, "1.0", instructions);
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

    scopeFrame   = frameProfiler.scope("frame");
    scopePoll    = frameProfiler.scope("frame/poll");
    scopeAdvance = frameProfiler.scope("frame/advance");
    scopeRender  = frameProfiler.scope("frame/render");
    // The command recording and submit inside render (render also waits for a frame slot and the swap chain image)
    scopeRecording = frameProfiler.scope("frame/render/recording");
    counterFps     = frameProfiler.counter("fps");
    counterSteps   = frameProfiler.counter("steps_per_frame");
    counterDropped = frameProfiler.counter("dropped_steps");
    scopeGpuFrame  = gpuProfiler.scope("gpu");
    for (const char* name : Vulkan::RenderScene::gpuSectionNames())
        scopeGpuSections.push_back(gpuProfiler.scope(std::string("gpu/") + name));
    // Before the first sample: a scope created later would count the empty history as zero samples
    if (!appSetup.renderHooks.passNames.empty()) scopeGpuGame = gpuProfiler.scope("gpu/game");
    for (const std::string& name : appSetup.renderHooks.passNames)
        scopeGpuPasses.push_back(gpuProfiler.scope("gpu/game/" + name));

    // set_input expires after its steps also inside one world().step(n) or a multi-step advance: released right after
    // the last step it covers, before the next step can read it
    gameWorld->addStepListener(
        [this]()
        {
            if (!hasVirtual || virtualSteps <= 0) return true;
            const uint64_t now = gameWorld->stepIndex();
            if (now >= virtualStartStep && now - virtualStartStep < uint64_t(virtualSteps)) return true;
            hasVirtual   = false;
            virtualInput = VirtualInput();
            releaseInputNow();
            return true;
        });

    // The local player survives a load (its handle is kept); otherwise the first player of peer 0, or none
    gameWorld->addLoadListener(
        [this](const LoadInfo&)
        {
            World& w = *gameWorld;
            if (localPlayerEntity.valid() && w.alive(localPlayerEntity) && w.registry().registered(Player::id) &&
                w.has<Player>(localPlayerEntity))
                return;
            // Another player: the loaded state replaced the old one's input, no release is pending for the new one
            localPlayerEntity = Entity{};
            inputWasActive    = false;
            if (!w.registry().registered(Player::id)) return;
            w.viewAll<const Player>().each(
                [&](Entity e, const Player& p)
                {
                    if (!localPlayerEntity.valid() && p.peer == 0) localPlayerEntity = e;
                });
        });

    if (renderScene && appSetup.renderHooks.init)
    {
        // The constructor does not finish if init throws (no destructor): shutdown runs here for the GPU state init
        // created before it threw, while the device still exists
        hooksInitialized = true;
        try
        {
            appSetup.renderHooks.init(*this);
        }
        catch (...)
        {
            if (server) server->stop();
            renderScene->waitIdle();
            if (appSetup.renderHooks.shutdown) appSetup.renderHooks.shutdown(*this);
            throw;
        }
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

float& EngineApp::cameraSpeed()
{
#ifdef SAIGA_USE_GLFW
    return cameraHolder->camera.movementSpeed;
#else
    return cameraSpeedValue;
#endif
}

float& EngineApp::cameraSpeedFast()
{
#ifdef SAIGA_USE_GLFW
    return cameraHolder->camera.movementSpeedFast;
#else
    return cameraSpeedFastValue;
#endif
}

void EngineApp::quit(int code)
{
    quitRequested = true;
    exitStatus    = code;
}

void EngineApp::setVirtualInput(const VirtualInput& input, int steps)
{
    virtualInput     = input;
    virtualSteps     = steps;
    virtualStartStep = gameWorld->stepIndex();
    hasVirtual       = true;
    // Written to the local player at once, with the last device state: a step run before the next frame
    // (world().step(), the step tool) reads it; its pressed actions are queued once here
    if (!gameWorld->inStep())
    {
        updateInputMap(lastDeviceState, lastCaptured);
        writePlayerInput();
    }
}

void EngineApp::setLocalPlayer(Entity player)
{
    if (player == localPlayerEntity) return;
    // The previous player's active device input is released on it (the release of the next neutral frame would go to
    // the new player)
    World& w = *gameWorld;
    if (inputWasActive && localPlayerEntity.valid() && w.alive(localPlayerEntity))
    {
        if (w.has<PlayerInput>(localPlayerEntity))
        {
            PlayerInput& in = w.get<PlayerInput>(localPlayerEntity);
            in.move         = Vec3::Zero();
            in.heldButtons  = 0;
        }
        // The game's own controls (applyInput) of the previous player get a neutral update too
        if (appSetup.applyInput)
        {
            InputMap neutral = inputMap;
            neutral.update(InputState(), false);
            appSetup.applyInput(neutral, w, localPlayerEntity);
        }
    }
    inputWasActive    = false;
    localPlayerEntity = player;
}

int EngineApp::run()
{
    const double stepDt = gameWorld->dt();
    while (!quitRequested)
    {
        const auto start = std::chrono::steady_clock::now();
        frame();
        if (appWindow && appWindow->shouldClose() && !appSetup.headless) break;
        // Headless, or no frame presented (rendering disabled, a minimized window): paced to the step rate, MCP and the
        // world keep going
        if (appSetup.headless || !lastFrameRendered)
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
        lastFrameRendered = false;
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
    lastFrameDt = dt;
}

void EngineApp::applyPlayerInput()
{
    // The devices (a window with GLFW input; a headless app has none)
    InputState state;
    bool captured = false, mouseCaptured = false;
    if (appWindow && !appSetup.headless)
    {
        if (ImGui::GetCurrentContext())
        {
            captured      = ImGui::GetIO().WantCaptureKeyboard;
            mouseCaptured = ImGui::GetIO().WantCaptureMouse;
        }
        for (int k = 0; k < int(state.keys.size()); ++k) state.keys[size_t(k)] = keyboard().getKeyState(k) != 0;
        // Clicks and drags on ImGui windows do not reach the game
        for (int b = 0; b < int(state.mouseButtons.size()); ++b)
            state.mouseButtons[size_t(b)] = !mouseCaptured && mouse.getKeyState(b) != 0;
        const Vec2 m(double(mouse.getX()), double(mouse.getY()));
        if (mouseKnown && !mouseCaptured) state.mouseDelta = m - lastMouse;
        lastMouse  = m;
        mouseKnown = true;
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
    // Kept for input updates between frames (set_input, its expiry); a mouse move counts once
    lastDeviceState            = state;
    lastDeviceState.mouseDelta = Vec2::Zero();
    lastCaptured               = captured;
    updateInputMap(state, captured);

    World& w = *gameWorld;
    if (localPlayerEntity.valid() && !w.alive(localPlayerEntity)) localPlayerEntity = Entity{};
    if (!localPlayerEntity.valid())
    {
        // The free camera (WASD and mouse) when no player is driven
#ifdef SAIGA_USE_GLFW
        if (appWindow && !appSetup.headless)
        {
            // The frame's time: the speed does not depend on the frame rate
            const float dt = float(std::min(lastFrameDt, 0.1));
            if (mouseCaptured)
                cameraHolder->camera.resetMouse();
            else
                cameraHolder->camera.interpolate(dt, 0);
            if (!captured) cameraHolder->camera.update(dt);
        }
#endif
        return;
    }
    writePlayerInput();
}

void EngineApp::updateInputMap(const InputState& state, bool captured)
{
    // set_input lasts a number of world steps, counted from when it was set: steps of advance and steps run between
    // frames (world().step(), the step tool) alike; it is released before the next step could read it again (a load to
    // an earlier step ends it too)
    if (hasVirtual && virtualSteps > 0)
    {
        const uint64_t now = gameWorld->stepIndex();
        if (now < virtualStartStep || now - virtualStartStep >= uint64_t(virtualSteps))
        {
            hasVirtual   = false;
            virtualInput = VirtualInput();
        }
    }
    if (hasVirtual)
    {
        inputMap.update(state, captured, virtualInput);
        // Pressed actions are edges of one frame
        virtualInput.pressed.clear();
    }
    else
        inputMap.update(state, captured);
}

void EngineApp::writePlayerInput()
{
    World& w = *gameWorld;
    if (localPlayerEntity.valid() && !w.alive(localPlayerEntity)) localPlayerEntity = Entity{};
    if (!localPlayerEntity.valid()) return;
    PerspectiveCamera& cam = cameraHolder->camera;
    // Neutral device input does not overwrite what an agent set (set_component); an active input does, and its release
    // (the first neutral update after it) writes the neutral values once
    const bool active  = !inputMap.neutral();
    const bool release = !active && inputWasActive;
    inputWasActive     = active;
    if (w.has<PlayerInput>(localPlayerEntity) && (active || release))
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

void EngineApp::releaseInputNow()
{
    // The input map without the virtual input: the last device state again (no new edges, so a key held across the
    // expiry is not pressed again), written to the player (neutral if no device input is active)
    inputMap.update(lastDeviceState, lastCaptured);
    writePlayerInput();
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
    if (!appOptions.screenshot.empty() && !screenshotRequested &&
        framesRendered + 1 >= uint64_t(appOptions.screenshotFrame))
    {
        screenshotRequested    = true;
        const std::string file = appOptions.screenshot;
        renderScene->requestScreenshot(
            [this, file](const TemplatedImage<ucvec4>& img)
            {
                const bool saved = img.width > 0 && img.save(file);
                if (saved)
                    SAIGA_LOG("app", "saved screenshot %s", file.c_str());
                else
                    SAIGA_LOG("app", "could not save the screenshot %s", file.c_str());
                quit(saved ? 0 : 1);
            });
    }
    if (!renderScene->renderFrame(transfer, render, gui)) return;
    frameProfiler.add(scopeRecording, renderScene->lastRecordingMs());
    lastFrameRendered = true;
    framesRendered++;
    // Only the callbacks registered before this frame's dispatch: one a callback adds runs from the next frame on
    size_t count = frameCallbacks.size();
    for (size_t i = 0; i < count;)
    {
        // Called in place, keeping its state; moved out while it runs because it may add callbacks (the vector may
        // reallocate; new ones are appended, so index i stays its slot)
        auto cb = std::move(frameCallbacks[i]);
        bool done;
        try
        {
            done = cb();
        }
        catch (...)
        {
            // A throwing callback is removed (its slot must not stay empty)
            frameCallbacks.erase(frameCallbacks.begin() + std::ptrdiff_t(i));
            throw;
        }
        if (done)
        {
            frameCallbacks.erase(frameCallbacks.begin() + std::ptrdiff_t(i));
            count--;
        }
        else
            frameCallbacks[i++] = std::move(cb);
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
    while (renderScene->takeGpuFrameTimes(times)) pendingGpu.push_back(times);
    // Oldest first; a frame waits for the game's passes of the same frame (they arrive in a later frame's transfer)
    const bool gamePasses = appSetup.renderHooks.passTimes && !appSetup.renderHooks.passNames.empty();
    while (!pendingGpu.empty())
    {
        const Vulkan::FrameTimes& t = pendingGpu.front();
        std::vector<std::pair<std::string, double>> passes;
        if (gamePasses && !appSetup.renderHooks.passTimes(t.frame, passes))
        {
            // Not known yet; after 8 more frames never (no sample: zeros would lower the statistics)
            if (renderScene->frames() < t.frame + 8) break;
            pendingGpu.pop_front();
            continue;
        }
        gpuProfiler.add(scopeGpuFrame, t.gpuMs);
        for (size_t i = 0; i < t.gpuSectionMs.size() && i < scopeGpuSections.size(); ++i)
            gpuProfiler.add(scopeGpuSections[i], t.gpuSectionMs[i]);
        if (gamePasses)
        {
            // gpu/game: the sum of the game's passes (the parent of their scopes)
            double sum = 0;
            for (const auto& p : passes)
                for (size_t k = 0; k < appSetup.renderHooks.passNames.size(); ++k)
                    if (appSetup.renderHooks.passNames[k] == p.first)
                    {
                        gpuProfiler.add(scopeGpuPasses[k], p.second);
                        sum += p.second;
                    }
            if (scopeGpuGame >= 0) gpuProfiler.add(scopeGpuGame, sum);
        }
        gpuProfiler.endSample();
        pendingGpu.pop_front();
    }
}

}  // namespace Engine
}  // namespace Saiga
