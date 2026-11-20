/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/json11.hpp"
#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/export.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace Saiga
{
namespace Engine
{
class World;
class WorldApi;
class Profiler;

struct WorldToolsOptions
{
    // false: only the reading tools are registered (observing agents)
    bool allowEdits = true;
    // The directory of export_snapshot / import_snapshot (empty: those tools are refused)
    std::string snapshotDirectory;
    // measure_performance(frames: n): set by the app that renders (S14); answers later through respond, before the
    // deadline. Without it the frames form is an error.
    std::function<void(int frames, std::chrono::steady_clock::time_point deadline, McpServer::Respond respond)>
        measureFrames;
};

// When a tool handler must answer: the caller's deadline minus a tenth of the tool timeout (formatting and returning
// the result)
SAIGA_ENGINE_API std::chrono::steady_clock::time_point answerDeadline(const McpServer& server);
// The profile of the last n samples (n = 0: an empty window, not the whole history)
SAIGA_ENGINE_API json11::Json profileWindow(const Profiler& profiler, uint64_t samples);
// Compact JSON with exact numbers (the tools' result format)
SAIGA_ENGINE_API std::string exactToolJson(const json11::Json& j);

/**
 * Registers the engine's world tools (docs/engine_design.md §16.2, R-MCP-1; S10) on an MCP server: one tool per
 * WorldApi operation (describe_world, world_status, list_entities, inspect, set_component, add_component,
 * remove_component, set_transform, spawn, destroy, get_global, set_global, emit_event, queue_action, pause, resume,
 * step, time_scale, changes_since, get_profile, reset_profile, logs, save_snapshot, load_snapshot, list_snapshots,
 * delete_snapshot, diff_snapshot, export_snapshot, import_snapshot, get_data, reload_data, apply) and
 * measure_performance. The handlers run
 * in McpServer::poll(), which the app calls between steps (the world must not be in a step then). Register before
 * McpServer::start. Returns the API the tools share: the caller keeps it while the tools are used and releases it
 * before the world goes away (the tools then answer with an error).
 */
SAIGA_ENGINE_API std::shared_ptr<WorldApi> registerWorldTools(McpServer& server, World& world,
                                                              const WorldToolsOptions& options = {});

}  // namespace Engine
}  // namespace Saiga
