/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/camera/camera.h"
#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/engine/api/WorldApi.h"
#include "saiga/engine/app/EngineAppSetup.h"
#include "saiga/engine/app/InputMap.h"
#include "saiga/engine/core/Profiler.h"
#include "saiga/export.h"

#include <array>
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
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
/**
 * The engine's app (docs/engine_design.md §8.3; S13): the window (GLFW, or none when headless), the RenderScene frame,
 * the World, the MCP server and the input map, and the frame loop: poll events and input, mcp.poll(), the frame time,
 * the camera, the input to the local player, world.advance(), render. Headless it renders into offscreen images (or,
 * with noRender, not at all) and paces the frames to the step rate. When no frame can be presented (a minimized window)
 * the loop keeps polling MCP and advancing the world. S14: the camera hold of the camera tools, the frame clock restart
 * after blocking tool calls, the frame measurement of measure_performance(frames), the input reset on loads and the
 * scenario-load mark of run_scenario.
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

    World& world() { return *gameWorld; }
    McpServer& mcp() { return *server; }
    WorldApi& api() { return *worldApi; }
    PerspectiveCamera& camera();
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
    // A pending release of the device input goes to the previous player (not the new one)
    void setLocalPlayer(Entity player);
    Entity localPlayer() const { return localPlayerEntity; }
    // The virtual device for the next steps (0: until replaced); a paused world keeps it until steps run
    void setVirtualInput(const VirtualInput& input, int steps);
    void showStatistics(bool show) { statisticsVisible = show; }

    // Camera hold (S14): set_camera / move_camera take the camera from the game; a game that drives the camera itself
    // skips its update while held, renders from camera() and resets its own camera on the release edge
    bool cameraHeldByTool() const { return cameraHeld; }
    void holdCamera() { cameraHeld = true; }
    void releaseCamera() { cameraHeld = false; }
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
    Profiler& gpuProfile() { return gpuProfiler; }
    uint64_t renderedFrames() const { return framesRendered; }
    uint64_t frameCount() const { return framesRun; }

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
    void collectGpuTimes();
    struct FrameMeasurement;
    // Checks the running frame measurements (from a poll listener and when one starts)
    void checkMeasurements();
    void answerMeasurement(FrameMeasurement& m, bool stoppedEarly);
    void statisticsWindow();

    EngineAppSetup appSetup;
    AppOptions appOptions;
    std::unique_ptr<Vulkan::VulkanWindow> appWindow;
    std::unique_ptr<Vulkan::RenderScene> renderScene;
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
    bool renderingEnabled = true, statisticsVisible = false;
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
    int counterFps = -1, counterSteps = -1, counterDropped = -1;
    std::vector<int> scopeGpuSections, scopeGpuPasses;
    int scopeGpuGame       = -1;
    uint64_t droppedBefore = 0;
    std::deque<Vulkan::FrameTimes> pendingGpu;  // completed frames waiting for the game's passes of the same frame
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
};

}  // namespace Engine
}  // namespace Saiga
