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
    Vulkan::RenderSceneParameters render;        // --tonemap, --exposure set toneMap and exposure
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
    std::function<void(McpServer&, EngineApp&)> gameTools;  // game_* tools
    // Writes the game's own control components from the input map (e.g. flight sticks); with a local player
    std::function<void(const InputMap&, World&, Entity localPlayer)> applyInput;
    GameRenderHooks renderHooks;
    // The camera hold of a tool ended (EngineApp::releaseCamera): the game resets its own camera (e.g. a chase camera's
    // reset(), so it does not glide back)
    std::function<void(EngineApp&)> onCameraReleased;
    int mcpPort                  = 8765;  // 0: the server is not started (the tools exist for handleMessage)
    double mcpToolTimeoutSeconds = 120;
    std::string inputFile;          // default: <data directory>/input.json if it exists
    std::string snapshotDirectory;  // export_snapshot / import_snapshot (empty: refused)
    std::string mcpInstructions;    // the MCP server's instructions (initialize); a generic text if empty or noRender
    // The free camera's speeds (m/s; fast: with shift)
    float cameraSpeed = 10, cameraSpeedFast = 100;
    // The names --benchmark accepts (checked before the window opens; empty: the game has no benchmarks)
    std::vector<std::string> benchmarks;
    // The debug UI (S19c, DebugUi): off frees F1..F10 for the game and draws nothing; persistence: its state in
    // debug_ui.json of the working directory (never when headless, with a screenshot option or a benchmark)
    bool debugUi = true, debugUiPersistence = true;
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
    bool has(const std::string& flag) const
    {
        for (const std::string& f : gameFlags)
            if (f == flag) return true;
        return false;
    }
};

}  // namespace Engine
}  // namespace Saiga
