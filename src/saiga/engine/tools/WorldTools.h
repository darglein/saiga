/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/export.h"

#include <memory>

namespace Saiga
{
class McpServer;

namespace Engine
{
class World;
class WorldApi;

struct WorldToolsOptions
{
    // false: only the reading tools are registered (observing agents)
    bool allowEdits = true;
};

/**
 * Registers the engine's world tools (docs/engine_design.md §16.2, R-MCP-1; S10) on an MCP server: one tool per
 * WorldApi operation (describe_world, world_status, list_entities, inspect, set_component, add_component,
 * remove_component, set_transform, spawn, destroy, get_global, set_global, emit_event, queue_action, pause, resume,
 * step, time_scale, changes_since, get_profile, reset_profile, logs, apply) and measure_performance. The handlers run
 * in McpServer::poll(), which the app calls between steps (the world must not be in a step then). Register before
 * McpServer::start. Returns the API the tools share: the caller keeps it while the tools are used and releases it
 * before the world goes away (the tools then answer with an error).
 */
SAIGA_ENGINE_API std::shared_ptr<WorldApi> registerWorldTools(McpServer& server, World& world,
                                                              const WorldToolsOptions& options = {});

}  // namespace Engine
}  // namespace Saiga
