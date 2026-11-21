/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/camera/camera.h"
#include "saiga/core/image/templatedImage.h"
#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/engine/api/WorldApi.h"
#include "saiga/engine/app/DebugUi.h"
#include "saiga/engine/app/EngineAppSetup.h"
#include "saiga/engine/app/FrameStatistics.h"
#include "saiga/engine/app/InputMap.h"
#include "saiga/engine/app/McpCallLog.h"
#include "saiga/engine/core/Profiler.h"
#include "saiga/export.h"

#include <array>
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
class VulkanWindow;
class RenderScene;
}  // namespace Vulkan

namespace Engine
{
class LoopbackTransport;
class RenderSink;
class VulkanRenderSink;

// The camera of the next rendered frame (EngineApp::setViewCamera; S19b): double, fovy in radians, looking along -z
// of rotation (y up)
struct ViewCamera
{
    SpaceId space = 0;
    Vec3 position = Vec3(0, 2, 6);
    Quat rotation = Quat::Identity();
    double fovy   = 60.0 * 3.14159265358979323846 / 180.0;
    // The writer asked for no smoothing (a teleport; read by later consumers such as the camera rig of S35)
    bool snap = false;
};

// The rotation of a camera at the origin looking along dir (y up; dir near +-y keeps a heading from fallback, then
// -z); double (the app camera, the camera tools)
SAIGA_ENGINE_API Quat cameraLookRotation(const Vec3& dir, const Vec3& fallbackForward = Vec3(0, 0, -1));

/**
 * The engine's app (docs/engine_design.md §8.3; S13): the window (GLFW, or none when headless), the RenderScene frame,
 * the World, the MCP server and the input map, and the frame loop: poll events and input, mcp.poll(), the frame time,
 * the camera, the input to the local player, world.advance(), render. Headless it renders into offscreen images (or,
 * with noRender, not at all) and paces the frames to the step rate. When no frame can be presented (a minimized window)
 * the loop keeps polling MCP and advancing the world. S14: the camera hold of the camera tools, the frame clock restart
 * after blocking tool calls, the frame measurement of measure_performance(frames), the input reset on loads and the
 * scenario-load mark of run_scenario. S16b: --coop runs a hidden host world and the rendered client world over a
 * loopback pair, stepped interleaved from one accumulator; the join policy of EngineAppSetup::onPeerJoined spawns the
 * client's player, which becomes the local player when it arrives.
 */
class SAIGA_ENGINE_API EngineApp
{
   public:
    explicit EngineApp(const EngineAppSetup& setup, const AppOptions& options = AppOptions());
    ~EngineApp();
    EngineApp(const EngineApp&)            = delete;
    EngineApp& operator=(const EngineApp&) = delete;

    // Runs frames until quit (or the window closes); returns the exit code
    int run();
    // count frames with a fixed frame time (tests, benchmarks; deterministic)
    void runFrames(int count, double dt);
    // One iteration of the loop with a measured frame time
    void frame();
    void quit(int exitCode = 0);
    bool quitting() const { return quitRequested; }
    int exitCode() const { return exitStatus; }

    // The rendered world (input, MCP); with --coop the client world
    World& world() { return *gameWorld; }
    // --coop (S16b, AppOptions::coop): the hidden host world (nullptr without --coop) and the world the game spawns
    // its start state in (the host with --coop, else world())
    World* hostWorld() { return hostWorldPtr.get(); }
    World& startWorld() { return hostWorldPtr ? *hostWorldPtr : *gameWorld; }
    bool coop() const { return hostWorldPtr != nullptr; }
    // n steps of every world: with --coop the host one step, then the client one step, n times (so the client never
    // gets several host steps in one round); else world().step(n). The --demo prewarm, the step tool while paused
    void stepAll(int n);
    McpServer& mcp() { return *server; }
    WorldApi& api() { return *worldApi; }
    // The view camera as a float PerspectiveCamera (derived from viewCamera(): the input map's heading, old GUI code)
    const PerspectiveCamera& camera() const;
    // The single camera (S19b): the camera of the next rendered frame, used as is (no interpolation by the app; games
    // set it every frame from ExtractSystem::interpolatedPose in beforeRender). While a tool holds the camera the call
    // is ignored and returns false (games need no branch)
    bool setViewCamera(SpaceId space, const Vec3& position, const Quat& rotation, double fovy, bool snap = false);
    // A tool's write (set_camera / move_camera hold the camera; game_set_camera and the walk mode do not): not blocked
    void setViewCameraByTool(SpaceId space, const Vec3& position, const Quat& rotation, double fovy, bool hold);
    const ViewCamera& viewCamera() const { return view; }
    // The World's render sink over the scene (nullptr with noRender)
    VulkanRenderSink* vulkanSink() { return sink.get(); }
    // The free camera's speeds (initially EngineAppSetup::cameraSpeed/cameraSpeedFast)
    float& cameraSpeed();
    float& cameraSpeedFast();
    // nullptr with noRender
    Vulkan::RenderScene* scene() { return renderScene.get(); }
    Vulkan::VulkanWindow* window() { return appWindow.get(); }
    InputMap& input() { return inputMap; }
    const EngineAppSetup& setup() const { return appSetup; }
    const AppOptions& options() const { return appOptions; }

    // Called after each rendered frame; return true to remove it
    void onFrameRendered(std::function<bool()> callback) { frameCallbacks.push_back(std::move(callback)); }
    size_t frameCallbackCount() const { return frameCallbacks.size(); }
    // false: steps and MCP continue, no frames are rendered
    void setRenderingEnabled(bool enabled) { renderingEnabled = enabled; }
    bool isRenderingEnabled() const { return renderingEnabled; }
    // The entity the input writes (PlayerInput, queued actions, applyInput); none: the input moves the free camera
    // A pending release of the device input goes to the previous player (not the new one). With --coop it must be a
    // player of the client's peer (asserted); while there is none the app takes the entity with Player.peer ==
    // localPeer() when it arrives (also again after a Resync)
    void setLocalPlayer(Entity player);
    Entity localPlayer() const { return localPlayerEntity; }
    // The virtual device for the next steps (0: until replaced); a paused world keeps it until steps run
    void setVirtualInput(const VirtualInput& input, int steps);
    // The developer UI (S19c): the bar with the tabs, the panel, the console. Exists in every mode (also without ImGui
    // or a device), so games register tabs and sections unconditionally (e.g. in renderHooks.init)
    DebugUi& debugUi() { return *ui; }
    // Runs fn between steps: right after the next McpServer::poll of the frame loop, before the input and the steps.
    // For GUI buttons that change the world (the GUI runs while a frame is recorded, where the world must not step)
    void queueBetweenSteps(std::function<void()> fn) { betweenSteps.push_back(std::move(fn)); }

    // Camera hold (S14): set_camera / move_camera take the camera from the game; a game that drives the camera itself
    // skips its update while held, renders from camera() and resets its own camera on the release edge
    bool cameraHeldByTool() const { return cameraHeld; }
    void holdCamera() { cameraHeld = true; }
    // The release edge calls EngineAppSetup::onCameraReleased (the game resets its camera)
    void releaseCamera();
    // After a tool call that blocked the loop (run_scenario): the next frame's dt does not contain the blocked time
    void restartFrameClock();
    // run_scenario's load: the live world then descends from a scenario (before_scenario is kept) until another load
    void setScenarioLoadInProgress(bool inProgress) { scenarioLoadInProgress = inProgress; }
    bool liveWorldFromScenario() const { return worldFromScenario; }
    // Counted by run_scenario (a frame measurement running meanwhile answers with an error)
    void countScenarioRun() { scenarioRuns++; }
    // measure_performance(frames: n) (WorldToolsOptions::measureFrames): the next n frame loops
    void measureFrames(int frames, std::chrono::steady_clock::time_point deadline, McpServer::Respond respond);

    // The app's profiles (get_profile: frame, gpu): one sample per frame loop / per completed GPU frame
    Profiler& frameProfile() { return frameProfiler; }
    // The statistics window's frame statistics (presented frames after the warm-up)
    const FrameStatistics& frameStatistics() const { return frameStats; }
    Profiler& gpuProfile() { return gpuProfiler; }
    // The gpu profile's sample of RenderScene frame `frame` (one of the last 600 completed frames) by scope name (gpu,
    // gpu/<section>, ...); false if it is not logged (not completed yet, or too old)
    bool gpuFrameSample(uint64_t frame, std::vector<std::pair<std::string, double>>& values) const;
    uint64_t renderedFrames() const { return framesRendered; }
    uint64_t frameCount() const { return framesRun; }

    // ---- terrain (S20, docs/engine_design.md §8.3, §12.3) ---------------------------------------------------------
    // Every rendered frame streams the terrain of the view camera (TerrainSystem::updateView after beforeRender). The
    // complete points call TerrainSystem::completeView instead: the first rendered frame, the frames of a capture
    // (requestCapture: SAIGA_SCREENSHOT / --screenshot, the MCP screenshot), and every frame while
    // setCompleteViewEachFrame(true) (the benchmark mode, E9). A timeout is fatal except for MCP screenshots (their
    // deadline bounds the call; the expired ones are answered with an error)
    void setCompleteViewEachFrame(bool on) { completeEachFrame = on; }
    bool completeViewEachFrame() const { return completeEachFrame; }
    // The terrain stream time of the last rendered frame (its completeView calls; 0 for a frame with updateView): not
    // CPU work, not part of the frame interval in the statistics, and it pauses the simulation clock (frame()
    // subtracts it from the next measured interval)
    double lastTerrainStreamMs() const { return streamMs; }
    using CaptureCallback = std::function<void(const TemplatedImage<ucvec4>&)>;
    // A capture of a rendered frame once the terrain of the view is complete: each frame until then completes the
    // view again (with the then current view camera) and the capture takes the first frame before which no terrain
    // upload is pending (at most 16 frames later, then with a warning). fatal: a completeView timeout stops the program
    // (SAIGA_SCREENSHOT); otherwise the deadline of the MCP screenshot requests bounds it
    void requestCapture(CaptureCallback callback, bool fatal = false);
    // AppTools (S20): the earliest deadline of the unanswered screenshot requests (none: no limit from them) and the
    // answer of the requests whose deadline passed (or passes before the given time)
    void setScreenshotDeadlineHooks(std::function<std::optional<std::chrono::steady_clock::time_point>()> earliest,
                                    std::function<void(std::chrono::steady_clock::time_point)> expire)
    {
        screenshotEarliest = std::move(earliest);
        screenshotExpire   = std::move(expire);
    }
    std::optional<std::chrono::steady_clock::time_point> pendingScreenshotDeadline() const
    {
        return screenshotEarliest ? screenshotEarliest() : std::nullopt;
    }

   private:
    void frame(double dt);
    void pollInput(double dt);
    void applyPlayerInput();
    // The input map from a device state and the virtual input (set_input; its expiry and pressed actions)
    void updateInputMap(const InputState& state, bool captured);
    // The input map to the local player (PlayerInput, pressed actions, applyInput)
    void writePlayerInput();
    void renderFrame();
    // Writes the neutral input to the local player now (an expired set_input between steps)
    void releaseInputNow();
    // The frame's steps: World::advance, or with --coop the host's stepsForAdvance run by stepAll
    void advanceWorlds(double dt);
    uint64_t droppedTotal() const;
    void collectGpuTimes();
    struct FrameMeasurement;
    // Checks the running frame measurements (from a poll listener and when one starts)
    void checkMeasurements();
    void answerMeasurement(FrameMeasurement& m, bool stoppedEarly);
    // The debug UI's content (EngineAppDebugUi.cpp)
    void setupDebugUi();
    void ensureShownStatistics();
    void performanceTab();
    void perfSummary();
    void perfTable();
    void perfGraph(float height);
    void perfSimulation();
    void perfGpu();
    void engineTab();
    void barSummary(float width);
    // Pauses or resumes the world between steps (the Pause key, the buttons)
    void togglePause();

    EngineAppSetup appSetup;
    AppOptions appOptions;
    std::unique_ptr<Vulkan::VulkanWindow> appWindow;
    std::unique_ptr<Vulkan::RenderScene> renderScene;
    std::unique_ptr<VulkanRenderSink> sink;
    ViewCamera view;
    // The float camera of camera() (and the GLFW mouse look) from the view camera
    void syncCamera();
    // The free camera (no local player): mouse look (not while ImGui has the mouse) and the keys (not while it has the
    // keyboard), the motion integrated in double
    void freeCamera(double dt, bool mouseLook, bool keys);
    // --coop: the loopback pair (peer 0 the host, peer 1 the client) outlives both worlds; the client world (gameWorld)
    // is destroyed before the host
    std::vector<std::unique_ptr<LoopbackTransport>> coopTransports;
    std::unique_ptr<World> hostWorldPtr;
    std::unique_ptr<World> gameWorld;
    std::unique_ptr<McpServer> server;
    std::shared_ptr<WorldApi> worldApi;
    struct CameraHolder;
    std::unique_ptr<CameraHolder> cameraHolder;
    InputMap inputMap;
    VirtualInput virtualInput;
    int virtualSteps          = 0;
    uint64_t virtualStartStep = 0;  // the world's step index when set_input was called
    bool hasVirtual           = false;
    Entity localPlayerEntity;
    std::vector<std::function<bool()>> frameCallbacks;
    bool renderingEnabled = true;
    std::unique_ptr<McpCallLog> callLog;
    std::unique_ptr<DebugUi> ui;
    std::vector<std::function<void()>> betweenSteps;
    // The statistics window's numbers: over one of kStatisticsWindows (seconds; the combo), recomputed every
    // kStatisticsRefreshSeconds (numbers that change every frame cannot be read)
    static constexpr double kStatisticsWindows[3]     = {3.0, 10.0, 30.0};
    static constexpr double kStatisticsRefreshSeconds = 0.5;
    int statisticsWindowIndex                         = 1;
    FrameStatistics frameStats;
    double pendingWorkMs      = -1;  // the CPU work of the last frame, added with its interval at the next frame
    uint64_t statisticsResets = 0;   // frameProfiler.resetCount() the frame statistics belong to
    static constexpr double kStatisticsWarmupSeconds = 1.0;
    double runSeconds                                = 0;        // the sum of the frame intervals since the start
    std::deque<std::pair<uint64_t, double>> renderedFrameTimes;  // RenderScene frame -> frameStats time
    struct ShownStatistics
    {
        double refreshHz = 0;  // the display's (0: unknown)
        double periodMs  = 0;  // the refresh period (vsync, FrameStatistics::refreshPeriod) or the median interval
        double graphSpan = 1;  // the time the graph covers (s)
        FrameStatistics::Summary summary;
        json11::Json stepReport;    // the step profile over the steps of that time
        double stepMs     = 0;      // the median step time (the bar)
        bool droppedSteps = false;  // steps were dropped in the last 120 frames (the bar)
        std::chrono::steady_clock::time_point at;
        bool valid = false;
    } shownStatistics;
    void refreshShownStatistics();
    // The last frame counts in the frame statistics (presented, after the warm-up) and the steps it ran
    bool pendingCounted  = false;
    int pendingSteps     = 0;
    double lastRefreshHz = 0;  // the display's refresh rate of the last statistics refresh
    void checkStatisticsReset();
    void clearFrameStatistics();
    double sessionPeriodMs = 0;  // the shortest refresh period measured in this session (vsync)
    struct Verdict
    {
        std::string text;
        double period       = 0;  // the refresh period (vsync) or the median interval (ms)
        bool displayLimited = false, warn = false;
        std::string tag;  // short: VSYNC-BOUND, GPU-BOUND, CPU-BOUND, WAIT-BOUND, MEASURING, ...
    };
    // What limits the frame rate (the statistics window's headline)
    Verdict statisticsVerdict() const;
    Verdict shownVerdict;  // of the shown statistics (refreshShownStatistics)
    // A stutter rate (missed refreshes or long frames per second) above this is drawn as a warning
    static constexpr double kStutterWarnPerSecond = 0.2;
    bool showEngineGpuSections() const;
    bool quitRequested    = false;
    int exitStatus        = 0;
    bool hooksInitialized = false;
    std::chrono::steady_clock::time_point lastFrame;
    bool firstFrame        = true;
    bool lastFrameRendered = false;
    bool inputWasActive    = false;  // the last frame wrote non-neutral input to the local player
    InputState lastDeviceState;      // the devices of the last frame (without its mouse move)
    bool lastCaptured  = false;
    double lastFrameDt = 0;
    // Without GLFW (no camera controls); unconditional, so the class layout does not depend on the macro
    [[maybe_unused]] float cameraSpeedValue = 10, cameraSpeedFastValue = 100;
    bool screenshotRequested = false;
    uint64_t framesRendered = 0, framesRun = 0;
    Vec2 lastMouse = Vec2::Zero();
    std::array<bool, 512> previousKeys{};
    bool mouseKnown = false;
    Profiler frameProfiler, gpuProfiler;
    int scopeFrame = -1, scopePoll = -1, scopeAdvance = -1, scopeRender = -1, scopeRecording = -1, scopeGpuFrame = -1;
    int scopeWait = -1, scopeWork = -1, scopeStream = -1;
    // The render waits of the current frame (frame/render/wait), subtracted from the frame for work
    double lastFrameWaitMs                   = 0;
    std::array<double, 3> lastFrameWaitParts = {};  // its parts: frame slot, acquire, present
    int counterFps = -1, counterSteps = -1, counterDropped = -1;
    std::vector<int> scopeGpuSections;
    uint64_t droppedBefore = 0;
    // The last GPU samples with their RenderScene frame (measure_performance(frames)): the 600 frames of the longest
    // measurement and those logged while it waits for the samples of its last frames (at most 8 + frames in flight)
    static constexpr size_t kGpuLogSize = 600 + 64;
    struct GpuLogEntry
    {
        uint64_t frame = 0;
        std::vector<std::pair<int, double>> values;  // gpu profiler scope, ms
    };
    std::deque<GpuLogEntry> gpuLog;
    std::vector<std::string> gpuScopeNames;  // by gpu profiler scope id
    std::vector<std::shared_ptr<FrameMeasurement>> measurements;
    bool inFrame = false, cameraHeld = false, scenarioLoadInProgress = false, worldFromScenario = false;
    uint64_t scenarioRuns = 0;
    // Terrain (S20)
    bool completeEachFrame = false;
    double streamMs        = 0;  // the last rendered frame's terrain stream time
    double frameStreamMs   = 0;  // the current frame's (0 without a rendered frame)
    // Benchmark mode: the wait for the GPU before the current frame's stream (a wait, not work; pauses the clock)
    double streamIdleWaitMs = 0;
    double pendingStreamMs  = 0;  // that time, taken out of the next frame's interval (statistics, simulation clock)
    struct PendingCapture
    {
        CaptureCallback callback;
        bool fatal = false;
    };
    std::vector<PendingCapture> pendingCaptures;
    int captureWaitFrames = 0;
    std::function<std::optional<std::chrono::steady_clock::time_point>()> screenshotEarliest;
    std::function<void(std::chrono::steady_clock::time_point)> screenshotExpire;
    // The terrain of the frame: updateView, or completeView at a complete point; hands ready captures to the scene
    void streamTerrain();
};

}  // namespace Engine
}  // namespace Saiga
