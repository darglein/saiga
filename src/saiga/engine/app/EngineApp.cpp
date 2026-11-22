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
#include "saiga/engine/terrain/TerrainSystem.h"
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

    setupDebugUi();
    scopeFrame   = frameProfiler.scope("frame");
    scopePoll    = frameProfiler.scope("frame/poll");
    scopeAdvance = frameProfiler.scope("frame/advance");
    scopeRender  = frameProfiler.scope("frame/render");
    // The command recording and submit inside render (render also waits for a frame slot and the swap chain image)
    scopeRecording = frameProfiler.scope("frame/render/recording");
    // The waits inside render for the frame slot, the swap chain image and the present: with vsync the display's
    // pace, without it the GPU's (idle time, not CPU work)
    scopeWait = frameProfiler.scope("frame/render/wait");
    // The terrain stream time of a complete point (S20): not work either
    scopeStream = frameProfiler.scope("frame/render/terrain_stream");
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
    // The last frame's terrain stream time pauses the simulation clock (no catch-up steps after a complete point)
    dt         = std::max(0.0, dt - pendingStreamMs / 1000.0);
    firstFrame = false;
    lastFrame  = now;
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
    // The last frame's terrain stream time: the time base covers it, the interval does not contain it (a complete
    // point is no long frame)
    if (pendingStreamMs > 0 && runSeconds > kStatisticsWarmupSeconds) frameStats.advance(pendingStreamMs);
    pendingStreamMs  = 0;
    frameStreamMs    = 0;
    streamIdleWaitMs = 0;
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
            // The debug UI's and the games' GUI actions that change the world, between steps (queueBetweenSteps)
            std::vector<std::function<void()>> actions;
            actions.swap(betweenSteps);
            for (auto& a : actions) a();
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
    // Neither the render's waits nor the terrain stream time of a complete point are CPU work
    pendingWorkMs = std::max(frameMs - lastFrameWaitMs - frameStreamMs, 0.0);
    // The benchmark mode's GPU wait before the stream belongs to the pause as well (it exists only because of it)
    pendingStreamMs = frameStreamMs + streamIdleWaitMs;
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
        // The app's keys: ESC quits, G toggles ImGui, F1..F10 and ` select the debug UI's tabs (DebugUi::handleKeys;
        // the input map does not see F1..F10), F12 saves screenshot.png
        auto pressedKey = [&](int k) { return !captured && state.keys[size_t(k)] && !previousKeys[size_t(k)]; };
        if (pressedKey(256)) quit(0);
        if (pressedKey('G') && renderScene) renderScene->setImguiVisible(!renderScene->imguiVisible());
        InputState previous;
        previous.keys                       = previousKeys;
        const std::array<bool, 512> rawKeys = state.keys;
        state                               = ui->handleKeys(state, previous, captured);
        if (pressedKey(301) && renderScene)
            renderScene->requestScreenshot(
                [](const TemplatedImage<ucvec4>& img)
                {
                    if (img.width > 0 && img.save("screenshot.png")) SAIGA_LOG("app", "saved screenshot.png");
                });
        previousKeys = rawKeys;
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
    if (!appOptions.screenshot.empty() && !screenshotRequested &&
        framesRendered + 1 >= uint64_t(appOptions.screenshotFrame))
    {
        // SAIGA_SCREENSHOT / --screenshot: a complete point; the capture waits for the terrain uploads
        screenshotRequested    = true;
        const std::string file = appOptions.screenshot;
        requestCapture(
            [this, file](const TemplatedImage<ucvec4>& img)
            {
                const bool saved = img.width > 0 && img.save(file);
                if (saved)
                    SAIGA_LOG("app", "saved screenshot %s", file.c_str());
                else
                    SAIGA_LOG("app", "could not save the screenshot %s", file.c_str());
                quit(saved ? 0 : 1);
            },
            true);
    }
    // The terrain of the view camera (S20): after beforeRender (the frame's camera), before the sink's camera
    streamTerrain();
    frameProfiler.add(scopeStream, streamMs);
    sink->setInterpolation(float(w.alpha()));
    sink->setCamera(view.space, view.position, view.rotation, view.fovy);
    view.snap     = false;
    auto noRecord = [](vk::CommandBuffer) {};
    auto gui      = [&]()
    {
        // The game's own GUI in every tab, then the debug UI (bar and panel) over it
        if (h.gui) h.gui();
        ui->draw();
    };
    const bool rendered = renderScene->renderFrame(noRecord, noRecord, gui);
    // Also for a frame that was not submitted (e.g. an out-of-date swap chain after the frame slot wait): the wait
    // is not work
    lastFrameWaitMs    = renderScene->lastWaitMs() + streamIdleWaitMs;
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

void EngineApp::requestCapture(CaptureCallback callback, bool fatal)
{
    pendingCaptures.push_back(PendingCapture{std::move(callback), fatal});
}

void EngineApp::streamTerrain()
{
    World& w               = *gameWorld;
    TerrainSystem& terrain = w.terrain();
    streamIdleWaitMs       = 0;
    // Benchmark mode: a frame that has to stream first lets the GPU finish the frames in flight (a wait, not work).
    // Otherwise the GPU would run ahead during the stream time, and the benchmark's frame interval without the stream
    // time would hide the GPU's time (S20). The wait pauses the simulation clock and the statistics' interval like the
    // stream time, but is not part of lastTerrainStreamMs (the benchmark keeps it in its interval)
    if (completeEachFrame && renderScene && !terrain.viewComplete(view.space, view.position))
    {
        const auto idle = std::chrono::steady_clock::now();
        renderScene->waitIdle();
        streamIdleWaitMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - idle).count();
    }
    const auto start  = std::chrono::steady_clock::now();
    bool fatalCapture = false, mcpCapture = false;
    for (const PendingCapture& c : pendingCaptures) (c.fatal ? fatalCapture : mcpCapture) = true;
    // An MCP capture completes the view only while one of its requests waits (else its readback is abandoned)
    const auto mcpDeadline = pendingScreenshotDeadline();
    mcpCapture             = mcpCapture && mcpDeadline.has_value();
    const bool fatal       = framesRendered == 0 || completeEachFrame || fatalCapture;
    // An MCP capture whose view did not complete this frame (the requests that wait longer must not get this frame)
    bool mcpIncomplete = false;
    if (fatal || mcpCapture)
    {
        double timeout = 120;
        if (!fatal) timeout = std::max(0.0, std::chrono::duration<double>(*mcpDeadline - start).count() - 0.05);
        const bool complete = terrain.completeView(view.space, view.position, timeout);
        if (!complete)
        {
            if (fatal)
                SAIGA_EXIT_ERROR("EngineApp: the terrain of the view was not complete within " +
                                 std::to_string(timeout) + " s (TerrainSystem::completeView)");
            // The screenshot requests whose deadline passed (or passes before the next frame) answer with an error; the
            // others keep waiting: the capture is not dispatched in this frame
            mcpIncomplete = true;
            if (screenshotExpire) screenshotExpire(std::chrono::steady_clock::now() + std::chrono::milliseconds(100));
        }
        streamMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    else
    {
        terrain.updateView(view.space, view.position);
        streamMs = 0;
    }
    frameStreamMs = streamMs;
    // The captures take the first frame before which no terrain upload is pending (at most 16 frames later)
    if (!pendingCaptures.empty() && !mcpIncomplete)
    {
        const bool ready = terrain.terrainUploadsPending() == 0;
        if (ready || captureWaitFrames >= 16 || !(fatal || mcpCapture))
        {
            if (!ready && (fatal || mcpCapture))
                SAIGA_LOG("app", "capture: terrain uploads still pending after %d frames", captureWaitFrames);
            for (PendingCapture& c : pendingCaptures) renderScene->requestScreenshot(std::move(c.callback));
            pendingCaptures.clear();
            captureWaitFrames = 0;
        }
        else
            captureWaitFrames++;
    }
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
