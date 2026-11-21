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
#include "saiga/engine/net/LoopbackTransport.h"
#include "saiga/engine/sync/ExtractSystem.h"
#include "saiga/engine/sync/VulkanRenderSink.h"
#include "saiga/engine/tools/WorldTools.h"
#include "saiga/saiga_buildconfig.h"
#include "saiga/saiga_modules.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/window/StandaloneWindow.h"
#include "saiga/vulkan/window/Window.h"

#ifdef SAIGA_USE_GLFW
#    include "saiga/core/glfw/glfw_camera.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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

Quat cameraLookRotation(const Vec3& direction, const Vec3& fallbackForward)
{
    Vec3 f = direction;
    if (!(f.norm() > 1e-12) || !std::isfinite(f.norm())) f = fallbackForward;
    f.normalize();
    // Only (almost) straight up / down (S14, CodeRabbit: a clamp at |y| > 0.999 changed valid pitches, 88 degrees came
    // back as 87.4): the view needs an up vector that is not parallel, and the heading is kept with a tiny horizontal
    // part, from the fallback (or -z) when there is none
    constexpr double kMinHorizontal = 1e-3;  // about 89.94 degrees
    Vec3 h(f(0), 0, f(2));
    if (h.norm() < kMinHorizontal)
    {
        if (h.norm() < 1e-6) h = Vec3(fallbackForward(0), 0, fallbackForward(2));
        if (!(h.norm() >= 1e-6)) h = Vec3(0, 0, -1);
        h = h.normalized() * kMinHorizontal;
        f = Vec3(h(0), (f(1) < 0 ? -1.0 : 1.0) * std::sqrt(1.0 - kMinHorizontal * kMinHorizontal), h(2));
    }
    // The camera looks along -z: z = -f, x = up x z, y = z x x
    const Vec3 z = -f;
    const Vec3 x = Vec3(0, 1, 0).cross(z).normalized();
    const Vec3 y = z.cross(x);
    Mat3 m;
    for (int i = 0; i < 3; ++i)
    {
        m(i, 0) = x(i);
        m(i, 1) = y(i);
        m(i, 2) = z(i);
    }
    return quat_cast(m);
}

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

    if (appOptions.coop)
    {
        // --coop (S16b): a hidden host world (peer 0) and the client world (peer 1) the app renders, over a loopback
        // pair with the given delay; the client shares the host's data store (reload_data reloads it once)
        LoopbackOptions lo;
        lo.delayRounds = appOptions.coopDelay;
        lo.seed        = appSetup.world.seed;
        coopTransports = LoopbackTransport::createNetwork(2, lo);
        WorldSetup hs  = appSetup.world;
        hs.transport   = coopTransports[0].get();
        hs.role        = NetRole::Host;
        hs.localPeer   = 0;
        hostWorldPtr   = std::make_unique<World>(hs);
        WorldSetup cs  = appSetup.world;
        cs.transport   = coopTransports[1].get();
        cs.role        = NetRole::Client;
        cs.localPeer   = 1;
        cs.sharedData  = hostWorldPtr->sharedDataStore();
        gameWorld      = std::make_unique<World>(cs);
        SAIGA_LOG("app", "co-op: a host and a client world over loopback, delay %d steps", appOptions.coopDelay);
        // The join policy (R-MECH-1): the peers of the host step that connected / disconnected them (the lists are
        // replaced at the next receive, so the step listener sees each change once)
        hostWorldPtr->addStepListener(
            [this]()
            {
                World& h                          = *hostWorldPtr;
                const std::vector<uint8_t> joined = h.peersJoined(), left = h.peersLeft();
                if (appSetup.onPeerJoined)
                    for (uint8_t p : joined) appSetup.onPeerJoined(*this, h, p);
                if (appSetup.onPeerLeft)
                    for (uint8_t p : left) appSetup.onPeerLeft(*this, h, p);
                return true;
            });
    }
    else
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
        // The rendered world's sink (S19b): the World forwards its meshes and materials, the extract its entities. With
        // --coop only the client world renders (the host has no sink)
        sink = std::make_unique<VulkanRenderSink>(*renderScene);
        gameWorld->setRenderSink(sink.get());
    }

    // The view camera: from (0, 2, 6) towards the origin
    view.position = Vec3(0, 2, 6);
    view.rotation = cameraLookRotation(Vec3(0, -2, -6));
    syncCamera();
    PerspectiveCamera& cam = cameraHolder->camera;
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
                                     : std::string(" or frames and the GPU; screenshot shows the next frame, "
                                                   "set_camera / move_camera place the camera")) +
                  "; list_scenarios / run_scenario run the game's scenarios."
            : appSetup.mcpInstructions;
    server                     = std::make_unique<McpServer>(appSetup.name, "1.0", instructions);
    server->toolTimeoutSeconds = appSetup.mcpToolTimeoutSeconds;
    WorldToolsOptions toolOptions;
    toolOptions.snapshotDirectory = appSetup.snapshotDirectory;
    if (renderScene)
        toolOptions.measureFrames = [this](int n, std::chrono::steady_clock::time_point deadline,
                                           McpServer::Respond respond) { measureFrames(n, deadline, respond); };
    worldApi = registerWorldTools(*server, *gameWorld, toolOptions);
    // --coop: the tools see the client world; pause, step, time scale and reload act on both (S16b)
    if (hostWorldPtr) worldApi->setCoop(WorldApi::CoopHooks{hostWorldPtr.get(), [this](int n) { stepAll(n); }});
    worldApi->addProfile("frame", &frameProfiler);
    worldApi->addProfile("gpu", &gpuProfiler);
    registerAppTools(*server, *this);
    if (appSetup.gameTools) appSetup.gameTools(*server, *this);
    server->addPollListener(
        [this]()
        {
            checkMeasurements();
            return true;
        });
    if (appSetup.mcpPort > 0)
    {
        if (server->start(appSetup.mcpPort))
            SAIGA_LOG("app", "MCP server: http://127.0.0.1:%d/mcp", server->port());
        else
            SAIGA_LOG("app", "MCP server: could not listen on port %d", appSetup.mcpPort);
    }

    statisticsVisible = appOptions.showStatistics;
    scopeFrame        = frameProfiler.scope("frame");
    scopePoll         = frameProfiler.scope("frame/poll");
    scopeAdvance      = frameProfiler.scope("frame/advance");
    scopeRender       = frameProfiler.scope("frame/render");
    // The command recording and submit inside render (render also waits for a frame slot and the swap chain image)
    scopeRecording = frameProfiler.scope("frame/render/recording");
    // The waits inside render for the frame slot, the swap chain image and the present: with vsync the display's
    // pace, without it the GPU's (idle time, not CPU work)
    scopeWait = frameProfiler.scope("frame/render/wait");
    // The CPU work of the frame: frame without wait (what the CPU really spends; frame is the frame's wall time)
    scopeWork      = frameProfiler.scope("work");
    counterFps     = frameProfiler.counter("fps");
    counterSteps   = frameProfiler.counter("steps_per_frame");
    counterDropped = frameProfiler.counter("dropped_steps");
    // The gpu scopes with their names (a frame measurement rebuilds a profile of logged samples with the same scopes)
    auto gpuScope = [this](const std::string& name)
    {
        const int id = gpuProfiler.scope(name);
        if (int(gpuScopeNames.size()) <= id) gpuScopeNames.resize(size_t(id) + 1);
        gpuScopeNames[size_t(id)] = name;
        return id;
    };
    // Before the first sample: a scope created later would count the empty history as zero samples
    scopeGpuFrame = gpuScope("gpu");
    for (const char* name : Vulkan::RenderScene::gpuSectionNames())
        scopeGpuSections.push_back(gpuScope(std::string("gpu/") + name));

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

    // The local player survives a load (its handle is kept); otherwise the first player of the world's peer, or none
    gameWorld->addLoadListener(
        [this](const LoadInfo&)
        {
            World& w = *gameWorld;
            // Only run_scenario's own load keeps the "descends from a scenario" mark (before_scenario stays)
            worldFromScenario = scenarioLoadInProgress;
            // Device input is transient: a held state restored by the load would never be released by neutral input
            // (S14; the games reset their own controls the same way)
            auto neutralInput = [&](Entity e)
            {
                if (!e.valid() || !w.alive(e) || !w.registry().registered(PlayerInput::id) || !w.has<PlayerInput>(e))
                    return;
                PlayerInput& in = w.get<PlayerInput>(e);
                in.move         = Vec3::Zero();
                in.heldButtons  = 0;
            };
            if (localPlayerEntity.valid() && w.alive(localPlayerEntity) && w.registry().registered(Player::id) &&
                w.has<Player>(localPlayerEntity))
            {
                neutralInput(localPlayerEntity);
                inputWasActive = false;
                return;
            }
            // Another player: the loaded state replaced the old one's input, no release is pending for the new one
            localPlayerEntity = Entity{};
            inputWasActive    = false;
            if (!w.registry().registered(Player::id)) return;
            w.viewAll<const Player>().each(
                [&](Entity e, const Player& p)
                {
                    if (!localPlayerEntity.valid() && p.peer == w.localPeer()) localPlayerEntity = e;
                });
            neutralInput(localPlayerEntity);
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
    // The world's instances leave the scene with the sink (before the scene)
    if (gameWorld && sink) gameWorld->setRenderSink(nullptr);
    sink.reset();
    renderScene.reset();
    worldApi.reset();
    server.reset();
    // The client before the host, both before their transports
    gameWorld.reset();
    hostWorldPtr.reset();
    coopTransports.clear();
    appWindow.reset();
}

const PerspectiveCamera& EngineApp::camera() const
{
    return cameraHolder->camera;
}

void EngineApp::syncCamera()
{
    PerspectiveCamera& cam = cameraHolder->camera;
    float aspect           = float(appSetup.width) / float(appSetup.height);
    if (renderScene && renderScene->size()(0) > 0 && renderScene->size()(1) > 0)
        aspect = float(renderScene->size()(0)) / float(renderScene->size()(1));
    cam.setProj(float(view.fovy * 180.0 / 3.14159265358979323846), aspect, 0.1f, 50.0f, true);
    // The view matrix: the inverse of the camera's pose (float; the scene gets the double camera)
    const Mat3 r = view.rotation.matrix();
    Mat4 v       = Mat4::Identity();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) v(i, j) = r(j, i);
    const Vec3 t = -(r.transpose() * view.position);
    for (int i = 0; i < 3; ++i) v(i, 3) = t(i);
    cam.setView(v.cast<float>());
}

bool EngineApp::setViewCamera(SpaceId space, const Vec3& position, const Quat& rotation, double fovy, bool snap)
{
    if (cameraHeld) return false;
    view.space    = space;
    view.position = position;
    view.rotation = rotation.normalized();
    view.fovy     = fovy;
    view.snap     = snap;
    syncCamera();
    return true;
}

void EngineApp::setViewCameraByTool(SpaceId space, const Vec3& position, const Quat& rotation, double fovy, bool hold)
{
    if (hold) cameraHeld = true;
    view.space    = space;
    view.position = position;
    view.rotation = rotation.normalized();
    view.fovy     = fovy;
    view.snap     = true;
    syncCamera();
}

void EngineApp::releaseCamera()
{
    const bool wasHeld = cameraHeld;
    cameraHeld         = false;
    if (wasHeld && appSetup.onCameraReleased) appSetup.onCameraReleased(*this);
}

void EngineApp::freeCamera(double dt, bool mouseLook, bool keys)
{
#ifdef SAIGA_USE_GLFW
    auto& c = cameraHolder->camera;
    // The mouse look of the GLFW camera from the view camera's pose
    syncCamera();
    if (mouseLook)
    {
        const quat before = c.rot;
        c.interpolate(float(dt), 0);
        const quat r = c.rot;
        // Only a turn replaces the double rotation (a tool's exact orientation stays while the mouse rests)
        if (r.w() != before.w() || r.x() != before.x() || r.y() != before.y() || r.z() != before.z())
            view.rotation = Quat(double(r.w()), double(r.x()), double(r.y()), double(r.z())).normalized();
    }
    if (keys)
    {
        // WASD, space up, shift fast; integrated in double (far from the origin a float camera would step)
        auto down          = [](int k) { return keyboard().getKeyState(k) != 0; };
        const double speed = down(340) ? c.movementSpeedFast : c.movementSpeed;
        const double fwd = (down('W') ? 1 : 0) - (down('S') ? 1 : 0), right = (down('D') ? 1 : 0) - (down('A') ? 1 : 0);
        view.position += view.rotation * Vec3(right, 0, -fwd) * (dt * speed);
        if (down(' ')) view.position(1) += dt * speed;
    }
    syncCamera();
#else
    (void)dt;
    (void)mouseLook;
    (void)keys;
#endif
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
    if (hostWorldPtr && player.valid() && gameWorld->alive(player) && gameWorld->registry().registered(Player::id) &&
        gameWorld->has<Player>(player))
        SAIGA_ASSERT(gameWorld->read<Player>(player).peer == gameWorld->localPeer(),
                     "EngineApp::setLocalPlayer: --coop drives the client's own player (Player.peer " +
                         std::to_string(int(gameWorld->read<Player>(player).peer)) + ", the client is peer " +
                         std::to_string(int(gameWorld->localPeer())) + ")");
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

void EngineApp::stepAll(int n)
{
    if (!hostWorldPtr)
    {
        gameWorld->step(n);
        return;
    }
    for (int i = 0; i < n; ++i)
    {
        hostWorldPtr->step();
        gameWorld->step();
    }
}

uint64_t EngineApp::droppedTotal() const
{
    return hostWorldPtr ? hostWorldPtr->droppedSteps() : gameWorld->droppedSteps();
}

void EngineApp::advanceWorlds(double dt)
{
    if (!hostWorldPtr)
    {
        gameWorld->advance(dt);
        return;
    }
    // One accumulator (the host's) for both worlds; the client's alpha follows it
    const int n = hostWorldPtr->stepsForAdvance(dt);
    stepAll(n);
    gameWorld->setAccumulatedTime(hostWorldPtr->accumulatedTime());
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

void EngineApp::restartFrameClock()
{
    if (!firstFrame) lastFrame = std::chrono::steady_clock::now();
}

void EngineApp::frame(double dt)
{
    framesRun++;
    // Also reset when the frame throws (a later measurement from outside a frame must not skip a loop)
    struct InFrame
    {
        bool& flag;
        explicit InFrame(bool& f) : flag(f) { flag = true; }
        ~InFrame() { flag = false; }
    } inFrameGuard(inFrame);
    // The frame's wall time, and its work: without the render's waits for the frame slot, the swap chain image and
    // the present (vsync or the GPU; idle time, not CPU work)
    const bool timed      = frameProfiler.enabled;
    const auto frameStart = std::chrono::steady_clock::now();
    lastFrameWaitMs       = 0;
    const auto waitParts  = lastFrameWaitParts;
    lastFrameWaitParts    = {};
    // A profile reset (reset_profile in the last loop's poll) applies before new samples
    checkStatisticsReset();
    // The last frame's interval ends now. It enters the statistics with its work and steps if it counts (decided when
    // it was rendered: presented, after the warm-up), its GPU time (arriving later) at the same time. Without a counted
    // frame (rendering suspended, a minimized window, a rejected present) the clock advances: old frames age out and
    // the gap is no long frame
    runSeconds += dt;
    if (dt > 0 && pendingCounted)
    {
        frameStats.addFrame(dt * 1000, pendingWorkMs, waitParts, pendingSteps);
        if (!renderedFrameTimes.empty() && renderedFrameTimes.back().second < 0)
            renderedFrameTimes.back().second = frameStats.now();
    }
    else if (dt > 0 && runSeconds > kStatisticsWarmupSeconds)
        frameStats.advance(dt * 1000);
    pendingCounted = false;
    pendingSteps   = 0;
    {
        {
            Profiler::Timer t(frameProfiler, scopePoll);
            pollInput(dt);
            server->poll();
        }
        applyPlayerInput();
        {
            Profiler::Timer t(frameProfiler, scopeAdvance);
            const uint64_t before = gameWorld->stepIndex();
            advanceWorlds(dt);
            pendingSteps = int(gameWorld->stepIndex() - before);
            frameProfiler.setCounter(counterSteps, double(pendingSteps));
            frameProfiler.setCounter(counterDropped, double(droppedTotal() - droppedBefore));
            droppedBefore = droppedTotal();
        }
        lastFrameRendered = false;
        if (renderScene && renderingEnabled)
        {
            Profiler::Timer t(frameProfiler, scopeRender);
            renderFrame();
        }
    }
    // The work also without the profiler (the frame statistics do not depend on it)
    const double frameMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - frameStart).count();
    pendingWorkMs = std::max(frameMs - lastFrameWaitMs, 0.0);
    if (timed)
    {
        frameProfiler.add(scopeFrame, frameMs);
        frameProfiler.add(scopeWork, pendingWorkMs);
    }
    frameProfiler.setCounter(counterFps, dt > 0 ? 1.0 / dt : 0.0);
    frameProfiler.endSample();
    if (renderScene) collectGpuTimes();
    // At the end of the loop: a measurement's last loop is reported before the next poll runs other tool calls
    if (!measurements.empty()) checkMeasurements();
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
    if (localPlayerEntity.valid() && !w.alive(localPlayerEntity))
    {
        // Gone (also a client's player after a Resync: the replicas are destroyed and arrive again)
        localPlayerEntity = Entity{};
        inputWasActive    = false;
    }
    if (!localPlayerEntity.valid() && !w.isHost() && w.registry().registered(Player::id))
    {
        // A client (--coop): its player arrives 1 + delay steps after the host's join step
        w.viewAll<const Player>().each(
            [&](Entity e, const Player& p)
            {
                if (!localPlayerEntity.valid() && p.peer == w.localPeer()) localPlayerEntity = e;
            });
    }
    if (!localPlayerEntity.valid())
    {
        // The free camera (WASD and mouse) when no player is driven
#ifdef SAIGA_USE_GLFW
        if (appWindow && !appSetup.headless)
        {
            // The frame's time: the speed does not depend on the frame rate. Regression (S19b review): the keys stopped
            // while the mouse was over an ImGui window; only the mouse look pauses then
            // A camera held by a tool moves only through the tools (as with a local player)
            if (mouseCaptured || cameraHeld) cameraHolder->camera.resetMouse();
            if (!cameraHeld) freeCamera(std::min(lastFrameDt, 0.1), !mouseCaptured, !captured);
        }
#endif
        return;
    }
    // Regression (CodeRabbit, S13): mouse look ran only for the free camera, so with a local player the view (and
    // PlayerInput.look) never turned. Look input turns the camera every frame (§8.3); the keys move the player, not it.
#ifdef SAIGA_USE_GLFW
    if (appWindow && !appSetup.headless)
    {
        // A camera held by a tool moves only through the tools
        if (mouseCaptured || cameraHeld)
            cameraHolder->camera.resetMouse();
        else
            freeCamera(std::min(lastFrameDt, 0.1), true, false);
    }
#endif
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
    const PerspectiveCamera& cam = cameraHolder->camera;
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
    World& w                 = *gameWorld;
    // Edits between steps (a paused world, tools, loads) reach the sink as snaps; then the game's frame state (the
    // camera from interpolatedPose, cosmetic instances), then the frame's interpolation and camera (S19b)
    w.extract().catchUp();
    if (h.beforeRender) h.beforeRender(*this, w.alpha());
    sink->setInterpolation(float(w.alpha()));
    sink->setCamera(view.space, view.position, view.rotation, view.fovy);
    view.snap     = false;
    auto noRecord = [](vk::CommandBuffer) {};
    auto gui      = [&]()
    {
        if (h.gui) h.gui();
        if (statisticsVisible)
            statisticsWindow();
        else if (statisticsOverlayEnabled)
            statisticsOverlay();
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
    const bool rendered = renderScene->renderFrame(noRecord, noRecord, gui);
    // Also for a frame that was not submitted (e.g. an out-of-date swap chain after the frame slot wait): the wait
    // is not work
    lastFrameWaitMs    = renderScene->lastWaitMs();
    lastFrameWaitParts = renderScene->lastWaitParts();
    frameProfiler.add(scopeWait, lastFrameWaitMs);
    if (!rendered) return;
    frameProfiler.add(scopeRecording, renderScene->lastRecordingMs());
    // Whether the frame counts in the frame statistics, decided once for its CPU interval and its GPU time: presented
    // (not rejected by the presentation engine) and after the warm-up. Its time is set when its interval is added
    pendingCounted = renderScene->lastPresented() && runSeconds > kStatisticsWarmupSeconds;
    if (pendingCounted) renderedFrameTimes.push_back({renderScene->frames(), -1.0});
    if (renderedFrameTimes.size() > 64) renderedFrameTimes.pop_front();
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

namespace
{
constexpr ImU32 kCpuColor   = IM_COL32(90, 200, 255, 255);
constexpr ImU32 kGpuColor   = IM_COL32(255, 170, 60, 255);
constexpr ImU32 kFrameColor = IM_COL32(120, 230, 120, 255);
constexpr ImU32 kRefColor   = IM_COL32(210, 210, 210, 170);
constexpr ImU32 kWarnColor  = IM_COL32(255, 110, 90, 255);
constexpr ImU32 kWhite      = IM_COL32(255, 255, 255, 255);

void coloredText(ImU32 color, const char* text)
{
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
}

// Right-aligned text in the current table cell
void cellRight(const char* text)
{
    ImGui::TableNextColumn();
    const float w = ImGui::CalcTextSize(text).x;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetColumnWidth() - w));
    ImGui::TextUnformatted(text);
}

// Text with a dark shadow (readable on any bar color)
void shadowText(ImDrawList* dl, ImVec2 at, const char* text)
{
    dl->AddText(ImVec2(at.x + 1, at.y + 1), IM_COL32(0, 0, 0, 200), text);
    dl->AddText(at, kWhite, text);
}

// A bar of a share with its label at a fixed place (ImGui's ProgressBar moves the label with the fill)
void shareBar(float share, const char* label, ImU32 color, float width)
{
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const float h   = ImGui::GetFrameHeight();
    ImGui::Dummy(ImVec2(width, h));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p0, ImVec2(p0.x + width, p0.y + h), ImGui::GetColorU32(ImGuiCol_FrameBg));
    dl->AddRectFilled(p0, ImVec2(p0.x + width * std::clamp(share, 0.0f, 1.0f), p0.y + h),
                      (color & 0x00FFFFFF) | 0xA0000000);
    shadowText(dl, ImVec2(p0.x + 6, p0.y + (h - ImGui::GetTextLineHeight()) / 2), label);
}

struct GraphLine
{
    float value;
    ImU32 color;
    const char* label;
};

// The frame time graph of the statistics window over the frames of the window (the same frames as the numbers; until
// the window is full, the time since the statistics started): per pixel column the longest frame interval (a single
// long frame stays visible) on a scale that shows the refresh; below it a strip of the CPU busy time (area) and the
// GPU busy time (line) scaled to them, so their size shows also when they are tiny against the refresh. Reference
// lines with their labels in a gutter on the right; intervals above the scale are clipped and marked red at the top.
void frameGraph(const FrameStatistics& fs, double span, float frameScale, const std::vector<GraphLine>& lines,
                float busyScale, float budgetMs)
{
    const float gutter = 62;
    const float width  = std::max(ImGui::GetContentRegionAvail().x, 140.0f);
    const float frameH = 70, busyH = 42, gap = 4;
    const float lineH   = ImGui::GetTextLineHeight();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##frameGraph", ImVec2(width, frameH + gap + busyH + lineH));
    const bool hovered = ImGui::IsItemHovered();
    const float plotW  = width - gutter;
    ImDrawList* dl     = ImGui::GetWindowDrawList();
    const ImU32 dim    = ImGui::GetColorU32(ImGuiCol_TextDisabled);

    // Per pixel column: the longest interval and the largest CPU and GPU busy times (-1: no frame in the column)
    const int columns = std::max(int(plotW), 1);
    std::vector<float> frameMax(size_t(columns), -1), workMax(size_t(columns), -1), gpuMax(size_t(columns), -1);
    const double from = fs.now() - span;
    auto column       = [&](double t) { return std::clamp(int((t - from) / span * columns), 0, columns - 1); };
    for (const auto& f : fs.frameSamples())
    {
        if (f.time <= from) continue;
        const size_t c = size_t(column(f.time));
        frameMax[c]    = std::max(frameMax[c], float(f.interval));
        workMax[c]     = std::max(workMax[c], float(f.work));
    }
    for (const auto& g : fs.gpuSamples())
    {
        if (g.time <= from) continue;
        const size_t c = size_t(column(g.time));
        gpuMax[c]      = std::max(gpuMax[c], float(g.ms));
    }
    auto plot = [&](float top, float height, float scale, auto&& draw)
    {
        const ImVec2 p0(origin.x, top), p1(origin.x + plotW, top + height);
        dl->AddRectFilled(p0, p1, ImGui::GetColorU32(ImGuiCol_FrameBg));
        auto y = [=](double v) { return p1.y - float(std::min(v, double(scale)) / scale) * (height - 2) - 1; };
        draw(p0, p1, y);
        // The scale at the top left (with a background, over the data)
        char text[32];
        std::snprintf(text, sizeof(text), "%.2g ms", scale);
        const ImVec2 size = ImGui::CalcTextSize(text);
        dl->AddRectFilled(ImVec2(p0.x + 1, p0.y + 1), ImVec2(p0.x + size.x + 7, p0.y + size.y + 2),
                          ImGui::GetColorU32(ImGuiCol_WindowBg));
        dl->AddText(ImVec2(p0.x + 4, p0.y + 1), dim, text);
    };
    auto polyline = [&](const std::vector<float>& v, ImU32 color, auto&& y)
    {
        int last = -1;
        for (int c = 0; c < columns; ++c)
        {
            if (v[size_t(c)] < 0) continue;
            if (last >= 0)
                dl->AddLine(ImVec2(origin.x + float(last), y(v[size_t(last)])),
                            ImVec2(origin.x + float(c), y(v[size_t(c)])), color, 1.5f);
            last = c;
        }
    };
    // The frame intervals
    plot(origin.y, frameH, frameScale,
         [&](ImVec2 p0, ImVec2 p1, auto&& y)
         {
             polyline(frameMax, kFrameColor, y);
             for (int c = 0; c < columns; ++c)
                 if (frameMax[size_t(c)] > frameScale)
                     dl->AddRectFilled(ImVec2(p0.x + float(c) - 1.5f, p0.y), ImVec2(p0.x + float(c) + 1.5f, p0.y + 6),
                                       kWarnColor);
             for (const GraphLine& l : lines)
             {
                 if (l.value <= 0 || l.value > frameScale) continue;
                 const float ly = y(l.value);
                 dl->AddLine(ImVec2(p0.x, ly), ImVec2(p1.x, ly), l.color, 1.0f);
                 dl->AddText(ImVec2(p1.x + 4, ly - lineH / 2), l.color, l.label);
             }
         });
    // The busy times on their own scale
    const float busyTop = origin.y + frameH + gap;
    plot(busyTop, busyH, busyScale,
         [&](ImVec2 p0, ImVec2 p1, auto&& y)
         {
             for (int c = 0; c < columns; ++c)
                 if (workMax[size_t(c)] > 0)
                     dl->AddLine(ImVec2(p0.x + float(c) + 0.5f, p1.y),
                                 ImVec2(p0.x + float(c) + 0.5f, y(workMax[size_t(c)])),
                                 (kCpuColor & 0x00FFFFFF) | 0xA0000000);
             polyline(gpuMax, kGpuColor, y);
             if (budgetMs > 0 && budgetMs <= busyScale)
             {
                 const float ly = y(budgetMs);
                 dl->AddLine(ImVec2(p0.x, ly), ImVec2(p1.x, ly), kRefColor, 1.0f);
                 dl->AddText(ImVec2(p1.x + 4, ly - lineH / 2), kRefColor, "budget");
             }
             dl->AddText(ImVec2(p1.x + 4, p0.y + (busyH - lineH) / 2), dim, "busy");
         });
    // The time axis
    char text[32];
    std::snprintf(text, sizeof(text), "-%.0f s", span);
    dl->AddText(ImVec2(origin.x, busyTop + busyH), dim, text);
    dl->AddText(ImVec2(origin.x + plotW - ImGui::CalcTextSize("now").x, busyTop + busyH), dim, "now");
    if (hovered)
    {
        const int c = int(ImGui::GetIO().MousePos.x - origin.x);
        if (c >= 0 && c < columns && frameMax[size_t(c)] >= 0)
        {
            char gpu[32] = "-";
            if (gpuMax[size_t(c)] >= 0) std::snprintf(gpu, sizeof(gpu), "%.2f ms", gpuMax[size_t(c)]);
            ImGui::SetTooltip(
                "%.1f s ago\nframe interval %.2f ms (the longest in this pixel column)\nCPU busy %.2f ms\nGPU busy %s",
                span * (1.0 - (c + 0.5) / columns), frameMax[size_t(c)], workMax[size_t(c)], gpu);
        }
    }
}
}  // namespace

void EngineApp::checkStatisticsReset()
{
    // A profile reset (reset_profile) also starts the frame statistics anew
    if (frameProfiler.resetCount() == statisticsResets) return;
    statisticsResets = frameProfiler.resetCount();
    clearFrameStatistics();
}

void EngineApp::clearFrameStatistics()
{
    frameStats.clear();
    renderedFrameTimes.clear();
    sessionPeriodMs       = 0;
    shownStatistics.valid = false;
}

void EngineApp::refreshShownStatistics()
{
    checkStatisticsReset();
    ShownStatistics& s = shownStatistics;
    s.at               = std::chrono::steady_clock::now();
    s.valid            = true;
    // The display's refresh rate as reported (the window may have moved to another monitor)
    s.refreshHz = appWindow ? appWindow->refreshRateHz() : 0.0;
    // A display with another refresh rate (the window moved): its period does not apply to the old frames
    if (s.refreshHz != lastRefreshHz)
    {
        if (lastRefreshHz > 0 && s.refreshHz > 0) clearFrameStatistics();
        lastRefreshHz = s.refreshHz;
    }
    const double window = kStatisticsWindows[size_t(statisticsWindowIndex)];
    const bool vsync    = renderScene && renderScene->vsync();
    s.summary           = frameStats.summarize(window);
    if (vsync)
    {
        // The refresh period from the frames that made a refresh (FrameStatistics::refreshPeriod). The display rarely
        // changes: the shortest period of the session is kept, and used while the window has no frame that made a
        // refresh (every frame misses one: the window's estimate is a multiple of it)
        const double period = FrameStatistics::sessionRefreshPeriod(
            FrameStatistics::refreshPeriod(s.summary, s.refreshHz), s.refreshHz, sessionPeriodMs);
        s.periodMs = period;
        s.summary  = frameStats.summarize(window, period);
    }
    else
        s.periodMs = s.summary.interval.median;
    // The time the graph covers: the window, or the time since the statistics started until it is full
    s.graphSpan = std::max(std::min(window, frameStats.now() - frameStats.firstTime()), 1.0);
    // The steps the window's frames ran (none while the world is paused: no table of older steps)
    s.stepReport = s.summary.steps > 0 ? gameWorld->profiler().report(s.summary.steps) : json11::Json();
}

EngineApp::Verdict EngineApp::statisticsVerdict() const
{
    const ShownStatistics& s            = shownStatistics;
    const FrameStatistics::Summary& sum = s.summary;
    const bool vsync                    = renderScene && renderScene->vsync();
    const double period                 = s.periodMs;
    const double busy                   = std::max(sum.work.median, sum.gpu.median);
    const char* limit                   = sum.gpu.median >= sum.work.median ? "GPU" : "CPU";
    Verdict v;
    v.period = period;
    char text[160];
    // The largest of the CPU's waits (frame slot, image acquire, present): what a frame waits for when neither the
    // CPU nor the GPU fills it
    const char* waitNames[] = {"the frame slot (an earlier GPU frame)", "the swap chain image acquire", "the present"};
    const size_t waitMax = size_t(std::max_element(sum.waitParts.begin(), sum.waitParts.end()) - sum.waitParts.begin());
    if (!renderScene || !renderingEnabled || renderScene->headless())
        std::snprintf(text, sizeof(text), "Not presenting: paced to the step rate (%.0f steps/s)",
                      1.0 / gameWorld->dt());
    else if (sum.interval.samples == 0)
        std::snprintf(text, sizeof(text), "Measuring (the first second is left out)");
    else if (vsync && sum.missedRefreshes > 0 && busy >= 0.85 * period)
    {
        std::snprintf(text, sizeof(text), "Limited by the %s: frames miss refreshes (%.0f Hz display)", limit,
                      1000.0 / period);
        v.warn = true;
    }
    else if (vsync && busy >= 0.85 * period)
        std::snprintf(text, sizeof(text), "Limited by the display, the %s near its limit (%.0f%% of the refresh)",
                      limit, 100 * busy / period);
    else if (vsync)
    {
        std::snprintf(text, sizeof(text), "Limited by the display: vsync at %.0f Hz (%.2f ms per refresh)",
                      1000.0 / period, period);
        v.displayLimited = true;
    }
    else if (busy >= 0.85 * period)
        std::snprintf(text, sizeof(text), "Limited by the %s (no vsync)", limit);
    else if (sum.waitParts[waitMax] >= 0.05)
        std::snprintf(text, sizeof(text), "Limited by waits for %s (%.2f ms per frame; no vsync)", waitNames[waitMax],
                      sum.waitParts[waitMax]);
    else
        std::snprintf(text, sizeof(text), "Limit undetermined: neither the CPU, the GPU nor a wait fills the frame");
    v.text = text;
    return v;
}

void EngineApp::statisticsOverlay()
{
    // A compact line at the top of the screen while the statistics window is closed (F1 opens it)
    if (!shownStatistics.valid || std::chrono::steady_clock::now() - shownStatistics.at >=
                                      std::chrono::duration<double>(kStatisticsRefreshSeconds))
        refreshShownStatistics();
    const FrameStatistics::Summary& sum = shownStatistics.summary;
    const bool vsync                    = renderScene && renderScene->vsync();
    // With vsync the shown frames (what the player sees)
    const FrameStatistics::Series& frames = vsync && sum.shown.samples > 0 ? sum.shown : sum.interval;
    ImGui::SetNextWindowPos(ImVec2(8, 6), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.5f);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                                   ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs;
    if (ImGui::Begin("##statistics overlay", nullptr, flags))
    {
        char text[200];
        std::snprintf(text, sizeof(text), "%.0f fps", frames.mean > 0 ? 1000.0 / frames.mean : 0.0);
        coloredText(kFrameColor, text);
        ImGui::SameLine();
        if (vsync && shownStatistics.periodMs > 0)
        {
            std::snprintf(text, sizeof(text), "(vsync %.0f Hz)", 1000.0 / shownStatistics.periodMs);
            ImGui::TextDisabled("%s", text);
            ImGui::SameLine();
        }
        if (frames.lowValid)
            std::snprintf(text, sizeof(text), "1%% low %.0f fps  max frame %.1f ms", 1000.0 / frames.worst1Percent,
                          frames.max);
        else
            std::snprintf(text, sizeof(text), "max frame %.1f ms", frames.max);
        ImGui::TextUnformatted(text);
        if (vsync && sum.missedRefreshes > 0)
        {
            ImGui::SameLine();
            std::snprintf(text, sizeof(text), "missed %.1f/s",
                          sum.seconds > 0 ? sum.missedRefreshes / sum.seconds : 0.0);
            coloredText(kWarnColor, text);
        }
        ImGui::SameLine();
        std::snprintf(text, sizeof(text), "CPU busy %.2f ms", sum.work.median);
        coloredText(kCpuColor, text);
        ImGui::SameLine();
        std::snprintf(text, sizeof(text), "GPU busy %.2f ms", sum.gpu.median);
        coloredText(kGpuColor, text);
        ImGui::SameLine();
        ImGui::TextDisabled("F1");
    }
    ImGui::End();
}

void EngineApp::statisticsWindow()
{
    // The engine's frame statistics (docs/engine_design.md §17.1): what limits the frame rate, the CPU's and the GPU's
    // busy time against the frame, the stutter measures, and the world step profile. The numbers are statistics over
    // a time window (the first second after the start is left out), refreshed every kStatisticsRefreshSeconds (numbers
    // that change every frame cannot be read); the graph shows every frame of the window.
    const ImVec2 display = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos(ImVec2(std::max(display.x - 1020, 0.0f), 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(520, std::min(560.0f, std::max(display.y - 20, 300.0f))), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Engine statistics", &statisticsVisible))
    {
        ImGui::End();
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (!shownStatistics.valid || now - shownStatistics.at >= std::chrono::duration<double>(kStatisticsRefreshSeconds))
        refreshShownStatistics();
    const FrameStatistics::Summary& sum = shownStatistics.summary;
    const bool vsync                    = renderScene && renderScene->vsync();
    const Verdict verdict               = statisticsVerdict();
    const double period                 = verdict.period;
    char text[240];

    // ------------------------------------------------------------------------------------ What limits the frame rate
    coloredText(verdict.warn ? kWarnColor : kWhite, verdict.text.c_str());
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Display: with vsync the CPU and the GPU finish early and wait for the next refresh.\n"
            "GPU / CPU: its busy time fills the frame (with vsync: frames miss refreshes).");
    // The CPU's and the GPU's busy time (median) as shares of the frame: of the refresh with vsync
    const char* of = vsync ? "refresh" : "frame";
    auto busyBar   = [&](const char* name, double ms, ImU32 color, const char* tooltip)
    {
        coloredText(color, name);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tooltip);
        ImGui::SameLine(80);
        std::snprintf(text, sizeof(text), "%.2f ms  (%.0f%% of the %.2f ms %s)", ms,
                      period > 0 ? 100 * ms / period : 0.0, period, of);
        shareBar(period > 0 ? float(ms / period) : 0.0f, text, color, ImGui::GetContentRegionAvail().x);
    };
    busyBar("CPU busy", sum.work.median, kCpuColor,
            "The CPU time the frame works (median): input, world steps, extract, recording and submit.\n"
            "Without the waits for the GPU and the display.");
    busyBar("GPU busy", sum.gpu.median, kGpuColor, "The GPU time of the frame (median, timestamps)");
    if (verdict.displayLimited)
    {
        const double busy = std::max(sum.work.median, sum.gpu.median);
        std::snprintf(text, sizeof(text),
                      "headroom ~%.0fx: the %s needs %.2f ms = %.0f%% of the refresh (rough estimate)",
                      busy > 0 ? period / busy : 0.0, sum.gpu.median >= sum.work.median ? "GPU" : "CPU", busy,
                      period > 0 ? 100 * busy / period : 0.0);
        ImGui::TextDisabled("%s", text);
    }

    // ------------------------------------------------------------------------------------ Frame rate and stutter
    // With vsync the shown frames (each interval rounded to whole refreshes): what the player sees
    const FrameStatistics::Series& frames = vsync && sum.shown.samples > 0 ? sum.shown : sum.interval;
    ImGui::Spacing();
    std::snprintf(text, sizeof(text), "%.0f fps", frames.mean > 0 ? 1000.0 / frames.mean : 0.0);
    coloredText(kFrameColor, text);
    ImGui::SameLine();
    if (frames.lowValid)
        std::snprintf(text, sizeof(text), "1%% low %.0f fps   max frame %.1f ms", 1000.0 / frames.worst1Percent,
                      frames.max);
    else
        std::snprintf(text, sizeof(text), "1%% low -   max frame %.1f ms", frames.max);
    ImGui::TextUnformatted(text);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "1%% low: 1000 / the mean of the slowest 1%% of the frames (here %d of %d; from 100 frames on).\n"
            "%s",
            frames.worstCount, frames.samples,
            vsync ? "With vsync of the shown frames: each interval rounded to whole refreshes, as the display\n"
                    "shows it. The CPU-side intervals in the table below jitter around the refresh."
                  : "max frame: the longest frame interval.");
    ImGui::SameLine();
    if (!vsync && sum.alternating)
    {
        std::snprintf(text, sizeof(text), "uneven pacing: %.1f / %.1f ms", sum.shortMs, sum.longMs);
        coloredText(kWarnColor, text);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "The frame intervals alternate between short and long (p99 change between neighbouring\n"
                "frames %.2f ms): the motion is uneven although the mean and the median look fine.",
                sum.pacingP99);
    }
    else
    {
        const int stutters = vsync ? sum.missedRefreshes : sum.longFrames;
        const double rate  = sum.seconds > 0 ? stutters / sum.seconds : 0.0;
        std::snprintf(text, sizeof(text), "%s %.1f/s", vsync ? "missed refreshes" : "long frames", rate);
        coloredText(rate > 0.2 ? kWarnColor : ImGui::GetColorU32(ImGuiCol_Text), text);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "%d in the window. Stutter the player notices. With vsync: the refreshes the frames missed\n"
                "(an interval of 2 refreshes is one). Without vsync: frames longer than 1.5 x the median\n"
                "(and the median + 1 ms). Frame pacing, the p99 change between neighbouring frames: %.2f ms.",
                stutters, sum.pacingP99);
    }

    // ------------------------------------------------------------------------------------ The window and the table
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("window");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(70);
    const char* windows[] = {"3 s", "10 s", "30 s"};
    if (ImGui::Combo("##window", &statisticsWindowIndex, windows, 3)) shownStatistics.valid = false;
    ImGui::SameLine();
    ImGui::TextDisabled("%.1f of %.0f s, %d frames, updated every %.1f s", sum.seconds,
                        kStatisticsWindows[size_t(statisticsWindowIndex)], sum.interval.samples,
                        kStatisticsRefreshSeconds);
    ImGui::SameLine(ImGui::GetContentRegionMax().x - 50);
    if (ImGui::SmallButton("reset")) clearFrameStatistics();
    if (ImGui::BeginTable("frame times", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp))
    {
        ImGui::TableSetupColumn("per frame (ms)", ImGuiTableColumnFlags_WidthStretch, 1.8f);
        for (const char* c : {"mean", "median", "p99", "max"})
            ImGui::TableSetupColumn(c, ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
        ImGui::TableNextColumn();
        ImGui::TableHeader("per frame (ms)");
        for (const char* c : {"mean", "median", "p99", "max"}) cellRight(c);
        auto row = [&](const char* name, ImU32 color, const FrameStatistics::Series& st, const char* tooltip)
        {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            coloredText(color, name);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tooltip);
            for (double v : {st.mean, st.median, st.p99, st.max})
            {
                std::snprintf(text, sizeof(text), "%.2f", v);
                cellRight(text);
            }
        };
        if (vsync && sum.shown.samples > 0)
            row("shown frame", kFrameColor, sum.shown,
                "The frames as the display shows them: each for whole refreshes (a late frame and the\n"
                "catch-up frame after it, absorbed by the swap chain queue, miss no refresh). fps, 1% low and\n"
                "max above are of these.");
        row(vsync ? "frame interval (CPU)" : "frame interval", kFrameColor, sum.interval,
            "From one frame's start to the next one's on the CPU. With vsync it jitters around the\n"
            "refresh; the display shows every frame for whole refreshes (fps and 1% low above).");
        row("CPU busy", kCpuColor, sum.work, "The CPU time of the frame without the waits for the GPU and the display");
        std::snprintf(text, sizeof(text),
                      "The frame interval without the CPU busy time: idle time, not work. Its parts (means):\n"
                      "frame slot %.2f ms (the GPU still runs an earlier frame), image acquire %.2f ms,\n"
                      "present %.2f ms (with vsync the display paces the acquire or the present).",
                      sum.waitParts[0], sum.waitParts[1], sum.waitParts[2]);
        row("CPU wait", ImGui::GetColorU32(ImGuiCol_TextDisabled), sum.idle, text);
        row("GPU busy", kGpuColor, sum.gpu, "The GPU time of the frame (timestamps)");
        ImGui::EndTable();
    }

    // ------------------------------------------------------------------------------------ The graph (every frame)
    ImGui::TextDisabled("top:");
    ImGui::SameLine();
    coloredText(kFrameColor, "frame interval");
    ImGui::SameLine();
    ImGui::TextDisabled("  bottom (own scale):");
    ImGui::SameLine();
    coloredText(kCpuColor, "CPU busy");
    ImGui::SameLine();
    coloredText(kGpuColor, "GPU busy");
    const float frameScale       = float(std::max(2.5 * period, sum.interval.p99 * 1.2));
    std::vector<GraphLine> lines = {{float(period), kRefColor, vsync ? "refresh" : "median"}};
    if (vsync) lines.push_back({float(2 * period), kWarnColor, "missed"});
    const float busyScale = float(std::max({sum.work.p99, sum.gpu.p99, 0.05}) * 1.5);
    frameGraph(frameStats, shownStatistics.graphSpan, frameScale, lines, busyScale, float(period));

    // ------------------------------------------------------------------------------------ The simulation
    const json11::Json& report = shownStatistics.stepReport;
    if (ImGui::CollapsingHeader("Simulation (world steps)"))
    {
        ImGui::TextDisabled("per step at %.0f steps/s: a frame runs as many steps as are due", 1.0 / gameWorld->dt());
        if (report.is_null())
            ImGui::TextDisabled("no world steps in the window (paused)");
        else
        {
            // The step profile keeps the last historySize steps: a longer window shows only those
            const int shownSteps = report["samples"].int_value();
            ImGui::TextDisabled("%d steps (%.1f s)%s", shownSteps, shownSteps * gameWorld->dt(),
                                shownSteps < sum.steps ? ", the newest the step profile keeps" : "");
        }
        if (ImGui::BeginTable("scopes", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY, ImVec2(0, 220)))
        {
            ImGui::TableSetupColumn("scope", ImGuiTableColumnFlags_WidthStretch, 3.0f);
            ImGui::TableSetupColumn("mean ms", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableSetupColumn("p99 ms", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableSetupColumn("share", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableHeadersRow();
            for (const auto& sc : report["scopes"].array_items())
            {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(sc["scope"].string_value().c_str());
                std::snprintf(text, sizeof(text), "%.3f", sc["avg"].number_value());
                cellRight(text);
                std::snprintf(text, sizeof(text), "%.3f", sc["p99"].number_value());
                cellRight(text);
                std::snprintf(text, sizeof(text), "%.0f%%", sc["share_of_parent"].number_value() * 100);
                cellRight(text);
            }
            ImGui::EndTable();
        }
        for (const auto& c : report["counters"].array_items())
            ImGui::Text("%s: %.0f", c["counter"].string_value().c_str(), c["avg"].number_value());
    }

    // ------------------------------------------------------------------------------------ Rendering settings
    if (renderScene && ImGui::CollapsingHeader("Rendering"))
    {
        // The tone map and exposure of the frame (R-MECH-1; applied from the next frame)
        int toneMap                = int(renderScene->parameters().toneMap);
        const char* toneMapNames[] = {"identity (clay)", "ACES"};
        if (ImGui::Combo("tone map", &toneMap, toneMapNames, 2)) renderScene->setToneMap(Vulkan::ToneMap(toneMap));
        // The exposure acts only on tone-mapped output (the range of --exposure)
        float exposure = renderScene->parameters().exposure;
        ImGui::BeginDisabled(renderScene->parameters().toneMap == Vulkan::ToneMap::Identity);
        if (ImGui::SliderFloat("exposure", &exposure, 1.0f / 1024, 1024, "%.3f", ImGuiSliderFlags_Logarithmic))
            renderScene->setExposure(exposure);
        ImGui::EndDisabled();
    }
    ImGui::End();
}

void EngineApp::collectGpuTimes()
{
    Vulkan::FrameTimes t;
    while (renderScene->takeGpuFrameTimes(t))
    {
        GpuLogEntry entry;
        entry.frame = t.frame;
        entry.values.push_back({scopeGpuFrame, t.gpuMs});
        for (size_t i = 0; i < t.gpuSectionMs.size() && i < scopeGpuSections.size(); ++i)
            entry.values.push_back({scopeGpuSections[i], t.gpuSectionMs[i]});
        for (const auto& v : entry.values) gpuProfiler.add(v.first, v.second);
        // At the time of its frame; frames of the warm-up (no time) are left out
        for (const auto& [frame, time] : renderedFrameTimes)
            if (frame == t.frame && time >= 0) frameStats.addGpu(t.gpuMs, time);
        gpuProfiler.endSample();
        gpuLog.push_back(std::move(entry));
        if (gpuLog.size() > kGpuLogSize) gpuLog.pop_front();
    }
}

bool EngineApp::gpuFrameSample(uint64_t frame, std::vector<std::pair<std::string, double>>& values) const
{
    values.clear();
    for (const GpuLogEntry& e : gpuLog)
    {
        if (e.frame != frame) continue;
        for (const auto& v : e.values)
            if (v.first >= 0 && size_t(v.first) < gpuScopeNames.size())
                values.push_back({gpuScopeNames[size_t(v.first)], v.second});
        return true;
    }
    return false;
}

// measure_performance(frames: n): the frame loops S..S+n-1 after the call (S: the next loop, or the current one when
// the call came from a poll outside a frame), the steps they ran and the GPU samples of the frames they rendered
struct EngineApp::FrameMeasurement
{
    int frames = 0;
    std::chrono::steady_clock::time_point deadline;
    McpServer::Respond respond;
    uint64_t firstLoop = 0;  // frame profiler sample index of the first measured loop
    bool started = false, ended = false;
    uint64_t worldStart = 0, sceneStart = 0, sceneEnd = 0, endLoop = 0, stepsRan = 0;
    uint64_t lastFrames = 0, lastLoops = 0, loopsWithoutRender = 0;
    json11::Json frameReport, stepReport;
    uint64_t worldResets = 0, frameResets = 0, gpuResets = 0, loads = 0, scenarioRuns = 0;
};

void EngineApp::measureFrames(int frames, std::chrono::steady_clock::time_point deadline, McpServer::Respond respond)
{
    auto m          = std::make_shared<FrameMeasurement>();
    m->frames       = frames;
    m->deadline     = deadline;
    m->respond      = std::move(respond);
    m->firstLoop    = frameProfiler.sampleCount() + (inFrame ? 1 : 0);
    m->worldResets  = gameWorld->profiler().resetCount();
    m->frameResets  = frameProfiler.resetCount();
    m->gpuResets    = gpuProfiler.resetCount();
    m->loads        = gameWorld->loadCount();
    m->scenarioRuns = scenarioRuns;
    measurements.push_back(m);
    checkMeasurements();
}

void EngineApp::checkMeasurements()
{
    const auto now = std::chrono::steady_clock::now();
    for (size_t i = 0; i < measurements.size();)
    {
        FrameMeasurement& m = *measurements[i];
        bool done           = false;
        if (gameWorld->profiler().resetCount() != m.worldResets || frameProfiler.resetCount() != m.frameResets ||
            gpuProfiler.resetCount() != m.gpuResets || gameWorld->loadCount() != m.loads ||
            scenarioRuns != m.scenarioRuns)
        {
            m.respond(McpServer::ToolResult::Error(
                gameWorld->loadCount() != m.loads ? "measure_performance: the world was loaded during the measurement"
                : scenarioRuns != m.scenarioRuns  ? "measure_performance: run_scenario ran during the measurement"
                                                  : "measure_performance: a profile was reset during the measurement"));
            done = true;
        }
        else
        {
            const uint64_t loops = frameProfiler.sampleCount();
            if (!m.started && loops >= m.firstLoop)
            {
                // At the first measured loop's poll: its steps run after it
                m.started    = true;
                m.worldStart = gameWorld->profiler().sampleCount();
                m.sceneStart = renderScene ? renderScene->frames() : 0;
            }
            if (m.started && !m.ended && loops >= m.firstLoop + uint64_t(m.frames))
            {
                m.ended       = true;
                m.endLoop     = loops;
                m.frameReport = profileWindow(frameProfiler, uint64_t(m.frames));
                m.stepsRan    = gameWorld->profiler().sampleCount() - m.worldStart;
                m.stepReport =
                    profileWindow(gameWorld->profiler(),
                                  std::min<uint64_t>(m.stepsRan, uint64_t(gameWorld->profiler().historySize())));
                m.sceneEnd = renderScene ? renderScene->frames() : 0;
            }
            if (m.ended)
            {
                // The GPU samples of the range arrive frames later: until a later frame's sample is logged, the scene
                // rendered 8 frames (the app's drop bound for unmatched frames) plus the frames in flight past the
                // range, 8 loops passed without a rendered frame, or the range is empty
                const uint64_t frames = renderScene ? renderScene->frames() : 0;
                if (frames != m.lastFrames) m.loopsWithoutRender = 0;
                if (loops != m.lastLoops && frames == m.lastFrames) m.loopsWithoutRender++;
                m.lastFrames       = frames;
                m.lastLoops        = loops;
                const bool later   = !gpuLog.empty() && gpuLog.back().frame > m.sceneEnd;
                const bool gpuDone = m.sceneEnd == m.sceneStart || later ||
                                     frames >= m.sceneEnd + 8 + Vulkan::VulkanRenderer::framesInFlight ||
                                     m.loopsWithoutRender >= 8;
                if (gpuDone || now >= m.deadline)
                {
                    // The deadline ended the wait for the GPU samples: some of the range may be missing
                    answerMeasurement(m, !gpuDone);
                    done = true;
                }
            }
            else if (now >= m.deadline)
            {
                // Stopped early: the loops measured so far
                const uint64_t measured = m.started ? loops - m.firstLoop : 0;
                m.frameReport           = profileWindow(frameProfiler, measured);
                m.stepsRan              = m.started ? gameWorld->profiler().sampleCount() - m.worldStart : 0;
                m.stepReport =
                    profileWindow(gameWorld->profiler(),
                                  std::min<uint64_t>(m.stepsRan, uint64_t(gameWorld->profiler().historySize())));
                m.sceneEnd = renderScene ? renderScene->frames() : 0;
                if (!m.started) m.sceneStart = m.sceneEnd;
                answerMeasurement(m, true);
                done = true;
            }
        }
        if (done)
            measurements.erase(measurements.begin() + std::ptrdiff_t(i));
        else
            ++i;
    }
}

void EngineApp::answerMeasurement(FrameMeasurement& m, bool stoppedEarly)
{
    using json11::Json;
    // The GPU samples of the frames rendered in the range, as a profile with the app's gpu scopes
    Profiler gpu(int(gpuLog.size()) + 1);
    std::vector<int> ids(gpuScopeNames.size(), -1);
    for (size_t k = 0; k < gpuScopeNames.size(); ++k)
        if (!gpuScopeNames[k].empty()) ids[k] = gpu.scope(gpuScopeNames[k]);
    uint64_t first = 0, last = 0, samples = 0;
    for (const GpuLogEntry& e : gpuLog)
    {
        if (e.frame <= m.sceneStart || e.frame > m.sceneEnd) continue;
        for (const auto& v : e.values)
            if (v.first >= 0 && size_t(v.first) < ids.size() && ids[size_t(v.first)] >= 0)
                gpu.add(ids[size_t(v.first)], v.second);
        gpu.endSample();
        if (samples == 0) first = e.frame;
        last = e.frame;
        samples++;
    }
    Json::object gpuReport = profileWindow(gpu, samples).object_items();
    if (samples > 0) gpuReport["frames"] = Json::array{double(first), double(last)};
    Json::object steps = m.stepReport.object_items();
    steps["ran"]       = double(m.stepsRan);
    if (m.stepsRan > uint64_t(gameWorld->profiler().historySize())) steps["truncated"] = true;
    Json::object o{{"frame", m.frameReport}, {"steps", steps}, {"gpu", gpuReport}, {"requested_frames", m.frames}};
    if (stoppedEarly) o["stopped_early"] = true;
    m.respond(McpServer::ToolResult::Text(exactToolJson(Json(o))));
}

}  // namespace Engine
}  // namespace Saiga
