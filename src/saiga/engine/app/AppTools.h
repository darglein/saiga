/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/export.h"

namespace Saiga
{
class McpServer;

namespace Engine
{
class EngineApp;

/**
 * The app's MCP tools (docs/engine_design.md §16.2): quit (ends the loop after the frame), app_status (headless,
 * rendering, size, fps, frames, steps, local player, the input bindings, the camera and its hold) and set_input (a
 * virtual input device for the next world steps) (S13); screenshot, set_camera and move_camera (the camera hold),
 * list_scenarios and run_scenario (S14); set_ui_tab (the debug UI's tabs, S19c). measure_performance(frames) is a world
 * tool with the app's hook.
 */
SAIGA_ENGINE_API void registerAppTools(McpServer& server, EngineApp& app);

}  // namespace Engine
}  // namespace Saiga
