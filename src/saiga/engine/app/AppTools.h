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
 * The app's MCP tools (docs/engine_design.md §16.2; S13): quit (ends the loop after the frame), app_status (headless,
 * rendering, size, fps, frames, steps, local player, the input bindings) and set_input (a virtual input device: axes,
 * held buttons and pressed actions of the local player for the next frames). The other app tools (screenshot, camera,
 * run_scenario) come in S14.
 */
SAIGA_ENGINE_API void registerAppTools(McpServer& server, EngineApp& app);

}  // namespace Engine
}  // namespace Saiga
