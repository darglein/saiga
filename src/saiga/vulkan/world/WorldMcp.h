/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/camera/camera.h"
#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/vulkan/VulkanForwardRenderer.h"
#include "saiga/vulkan/world/WorldRenderer.h"

namespace Saiga
{
namespace Vulkan
{
/**
 * What the MCP tools of the world control. The pointers must outlive the server.
 */
struct WorldMcpContext
{
    VulkanForwardRenderer* renderer = nullptr;
    WorldRenderer* world            = nullptr;
    Camera* camera                  = nullptr;
    // Optional: walk mode (the camera stays at eyeHeight above the terrain), see game_world
    bool* walkMode   = nullptr;
    float* eyeHeight = nullptr;
    // false: no measure_performance (the engine app registers its own, which also reports frames and the GPU)
    bool measurePerformance = true;
};

/**
 * Registers the tools to look at and control the world (screenshot, get_state, set_camera, move_camera,
 * set_settings, terrain_query, measure_performance). The handlers run in McpServer::poll(). Set
 * McpServer::toolTimeoutSeconds before: measure_performance stops early to answer within it.
 */
SAIGA_VULKAN_API void registerWorldMcpTools(McpServer& server, const WorldMcpContext& context);

// The instructions for the agent (McpServer constructor)
// measurePerformance: mention the renderer's measure_performance (false: the engine app's, see registerWorldTools)
SAIGA_VULKAN_API std::string worldMcpInstructions(bool measurePerformance = true);

}  // namespace Vulkan
}  // namespace Saiga
