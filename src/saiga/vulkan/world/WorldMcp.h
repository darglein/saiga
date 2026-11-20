/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/camera/camera.h"
#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/vulkan/world/WorldRenderer.h"

#include <functional>

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
    WorldRenderer* world = nullptr;
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
