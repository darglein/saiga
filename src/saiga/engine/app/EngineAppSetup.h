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

// Until S19 the game draws with the prototype renderers: the app calls these (docs/engine_design.md §8.3)
struct GameRenderHooks
{
    std::function<void(EngineApp&)> init;      // after the device exists (create the game's GPU state)
    std::function<void(EngineApp&)> shutdown;  // after the device is idle (destroy it)
    std::function<void(vk::CommandBuffer)> transfer, render;
    std::function<void()> gui;
    // The game's GPU passes of the RenderScene frame `frame` (milliseconds by pass name; the frame recorded while
    // RenderScene::frames() is frame - 1), false while they are not known (a completed frame waits up to 8 frames for
    // them, then it gets no gpu sample). They join the gpu profile as gpu/game/<name> (only the names of passNames,
    // registered before the first sample)
    std::vector<std::string> passNames;
    std::function<bool(uint64_t frame, std::vector<std::pair<std::string, double>>& passMs)> passTimes;
};

struct EngineAppSetup
{
    std::string name = "game";  // the window title and the data directory data/<name>
    WorldSetup world;
    bool headless = false;  // no window: offscreen images, no device input (agents, servers, tests)
    bool noRender = false;  // no Vulkan device at all (with headless): the world steps, MCP answers
    int width = 1600, height = 900;
    bool vsync = true, validation = false;
    Vulkan::RenderSceneParameters render;
    void (*scenarios)(ScenarioList&) = nullptr;             // --scenarios, run_scenario (S14)
    std::function<void(McpServer&, EngineApp&)> gameTools;  // game_* tools
    // Writes the game's own control components from the input map (e.g. flight sticks); with a local player
    std::function<void(const InputMap&, World&, Entity localPlayer)> applyInput;
    GameRenderHooks renderHooks;
    int mcpPort                  = 8765;  // 0: the server is not started (the tools exist for handleMessage)
    double mcpToolTimeoutSeconds = 120;
    std::string inputFile;          // default: <data directory>/input.json if it exists
    std::string snapshotDirectory;  // export_snapshot / import_snapshot (empty: refused)
    std::string mcpInstructions;    // the MCP server's instructions (initialize); a generic text if empty or noRender
    // The free camera's speeds (m/s; fast: with shift)
    float cameraSpeed = 10, cameraSpeedFast = 100;
    // The names --benchmark accepts (checked before the window opens; empty: the game has no benchmarks)
    std::vector<std::string> benchmarks;
};

// What the command line selects besides the setup
struct AppOptions
{
    std::string benchmark, screenshot, scenarioFilter;
    int screenshotFrame = 60;
    bool runScenarios   = false;
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
