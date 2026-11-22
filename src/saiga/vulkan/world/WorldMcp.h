/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/camera/camera.h"
#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/vulkan/world/WorldSettings.h"

#include <functional>
#include <utility>

namespace Saiga
{
namespace Vulkan
{
/**
 * What the MCP tools of the world control. The pointers must outlive the server.
 */
// The app's view camera as the tools see it (double; fovy in radians)
struct WorldMcpCamera
{
    Vec3 position = Vec3::Zero();
    Quat rotation = Quat::Identity();
    double fovy   = 1;
};

struct WorldMcpContext
{
    // The prototype's settings (PrototypeScene::settings) and render statistics (trees, terrain)
    WorldSettings* settings = nullptr;
    std::function<json11::Json()> stats;
    // The terrain height (m) under (x, z) (World::terrainHeight of the view space); the world's size (m), sea level
    // (m), its highest point (computed once by the game) and min / max height (the pyramid's, or the game's area stats
    // before the first view)
    std::function<double(double x, double z)> ground;
    double worldSize  = 0;
    double seaLevel   = 0;
    Vec3 highestPoint = Vec3::Zero();
    std::function<std::pair<double, double>()> heightRange;
    // The app's view camera (S19b: EngineApp::viewCamera / setViewCameraByTool without a hold, set by game_world;
    // saiga_vulkan has no engine header): game_set_camera moves it, game_set_settings' fovy sets its field of view
    std::function<WorldMcpCamera()> getCamera;
    std::function<void(const WorldMcpCamera&)> setCamera;
    // Optional: walk mode (the camera stays at eyeHeight above the terrain), see game_world
    bool* walkMode   = nullptr;
    float* eyeHeight = nullptr;
};

/**
 * Registers the game tools of the world prototype (docs/engine_design.md §16.2; S14): game_get_state,
 * game_set_camera, game_set_settings, game_terrain_query. The engine app provides screenshot, set_camera,
 * move_camera and measure_performance. The handlers run in McpServer::poll().
 */
SAIGA_VULKAN_API void registerWorldMcpTools(McpServer& server, const WorldMcpContext& context);

// The instructions for the agent (McpServer constructor)
SAIGA_VULKAN_API std::string worldMcpInstructions();

}  // namespace Vulkan
}  // namespace Saiga
