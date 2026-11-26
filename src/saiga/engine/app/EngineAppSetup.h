/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/ecs/Entity.h"
#include "saiga/engine/world/World.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"
#include "saiga/vulkan/svulkan.h"

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace Saiga
{
class McpServer;

namespace Engine
{
class EngineApp;
class InputMap;
class ScenarioList;
struct AppOptions;
struct SpaceLoading;

// The game's render hooks (docs/engine_design.md §8.3; S19b): the entities render through the World's render sink
// (RenderInstance, the extract); the game keeps device-bound state of its own (the prototype's tree sets, level
// geometry as instance sets, cosmetic dynamic instances) and sets the view camera
struct GameRenderHooks
{
    // After the device and the render sink exist (EngineApp::vulkanSink(): the scene for the game's own content)
    std::function<void(EngineApp&)> init;
    // After the device is idle (destroy the game's device-bound state)
    std::function<void(EngineApp&)> shutdown;
    // Every rendered frame after the extract's catch-up and before recording (not with --no-render): the camera
    // (EngineApp::setViewCamera, from ExtractSystem::interpolatedPose), cosmetic instances, the prototype's settings
    // and time; alpha: the world's interpolation factor of the frame
    std::function<void(EngineApp&, double alpha)> beforeRender;
    std::function<void()> gui;
};

struct EngineAppSetup
{
    std::string name = "game";  // the window title and the data directory data/<name>
    WorldSetup world;
    bool headless = false;  // no window: offscreen images, no device input (agents, servers, tests)
    bool noRender = false;  // no Vulkan device at all (with headless): the world steps, MCP answers
    int width = 1600, height = 900;
    bool vsync = true, validation = false;
    Vulkan::RenderSceneParameters render;  // --tonemap, --exposure set toneMap and exposure
    // The presentation settings of the environment (S36; exposure, ambient, clouds, atmosphere): the initial value of
    // EngineApp::environmentSettings (filled by the game or from readEnvironmentFile)
    Vulkan::EnvironmentSettings environment;
    void (*scenarios)(ScenarioList&) = nullptr;  // --scenarios, list_scenarios, run_scenario
    // --scenarios also runs the host + client variants (ScenarioOptions::networkVariants, §17.3); on for the tech
    // demos from S16b
    bool networkVariants = false;
    // --coop (S16b): the join policy (R-MECH-1), called by the app from a step listener of the hidden host world for
    // each peer of that step's peersJoined() / peersLeft() (e.g. spawn the peer's player, remove it)
    std::function<void(EngineApp&, World& host, uint8_t peer)> onPeerJoined, onPeerLeft;
    // Called after the command line was parsed, before the app exists (the game may adjust the options, e.g. clamp
    // the co-op delay for its --demo); return false and an error to reject them
    std::function<bool(AppOptions&, std::string& error)> checkOptions;
    // Called at the end of parseCommandLine, when world.dataDirectory is final (--data, else data/<name>): the game
    // reads its data files that set up the app (S36: the environment file of games/common/Environment, which --data
    // must select; CodeRabbit #83). The command line's own settings (--tonemap, --exposure) are already in the setup
    std::function<void(EngineAppSetup&)> onDataDirectory;
    std::function<void(McpServer&, EngineApp&)> gameTools;  // game_* tools
    // Writes the game's own control components from the input map (e.g. flight sticks); with a local player
    std::function<void(const InputMap&, World&, Entity localPlayer)> applyInput;
    GameRenderHooks renderHooks;
    // The camera hold of a tool ended (EngineApp::releaseCamera): the game resets its own camera (e.g. a chase camera's
    // reset(), so it does not glide back)
    std::function<void(EngineApp&)> onCameraReleased;
    int mcpPort                  = 8765;  // 0: the server is not started (the tools exist for handleMessage)
    double mcpToolTimeoutSeconds = 120;
    // McpServer::answerMarginFraction (tests with short timeouts raise it)
    double mcpAnswerMarginFraction = 0.1;
    // The frame statistics count presented frames from this much run time on (the sum of the frame intervals; tests
    // shorten it instead of waiting a second of real time)
    double statisticsWarmupSeconds = 1.0;
    std::string inputFile;          // default: <data directory>/input.json if it exists
    std::string snapshotDirectory;  // export_snapshot / import_snapshot (empty: refused)
    // S25: the save directory (--save <dir>; empty: none), opened after the game's start (an existing directory
    // replaces the world through the load path, a new one adopts it) and committed at quit; the commit interval in
    // world time (0: only at quit and commit_save) and the file syncs (the app test turns them off)
    std::string saveDirectory;
    double saveCommitIntervalSeconds = 300;
    bool saveSync                    = true;
    std::string mcpInstructions;  // the MCP server's instructions (initialize); a generic text if empty or noRender
    // The free camera's speeds (m/s; fast: with shift)
    float cameraSpeed = 10, cameraSpeedFast = 100;
    // The names --benchmark accepts (checked before the window opens; empty: the game has no benchmarks)
    std::vector<std::string> benchmarks;
    // The debug UI (S19c, DebugUi): off frees F1..F10 for the game and draws nothing (its state is not kept between
    // runs)
    bool debugUi = true;
    // The loading screen of a space switch (S21, §12.4; game policy, R-MECH-1): a load starts when the view camera's
    // space changes to one whose World::spaceReadiness within loadingRadius (m) is below 1, and ends at readiness 1
    // without pending terrain uploads, at a complete point, or after loadingMaxFrames frames (with a log line).
    // showLoadingScreen false: tracked and reported (EngineApp::spaceLoading, app_status), nothing drawn.
    // loadingScreen: drawn inside the ImGui frame while a load runs (null: the engine's drawLoadingScreen)
    double loadingRadius      = 256;
    uint64_t loadingMaxFrames = 600;
    bool showLoadingScreen    = true;
    std::function<void(EngineApp&, const SpaceLoading&)> loadingScreen;
};

// What the command line selects besides the setup
struct AppOptions
{
    std::string benchmark, screenshot, scenarioFilter;
    int screenshotFrame = 60;
    bool runScenarios   = false;
    bool showStatistics = false;  // --stats: the debug UI starts on its Performance tab (F2)
    int debugTab        = 0;      // SAIGA_DEBUG_TAB=<n>: the debug UI starts on tab n (1: F1, ...; 0: the saved tab)
    // --coop [delay] (S16b): a hidden host world and the rendered client world in one process over a loopback pair
    // with this delay in steps (0..30, default 2)
    bool coop     = false;
    int coopDelay = 2;
    std::vector<std::string> gameFlags;  // the declared game flags that were given (e.g. --demo)
    // --time-of-day h, --day-length s (S36): WorldTime's hours and rate (86400 / s), applied after the game's start;
    // a world without WorldTime (EnvironmentSetup::enabled) stops with an error
    std::optional<double> timeOfDay, dayLength;
    bool has(const std::string& flag) const
    {
        for (const std::string& f : gameFlags)
            if (f == flag) return true;
        return false;
    }
};

}  // namespace Engine
}  // namespace Saiga
