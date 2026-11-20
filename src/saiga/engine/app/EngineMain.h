/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/app/EngineAppSetup.h"
#include "saiga/export.h"

#include <functional>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
/**
 * The command line of an engine game (docs/engine_design.md §8.3; S13): --headless, --no-render, --benchmark <name>,
 * --screenshot <file> (after --screenshot-frame N, default 60, then quit; SAIGA_SCREENSHOT is the same),
 * --width/--height N, --validation, --no-vsync, --scenarios [filter], --mcp-port N (also SAIGA_MCP_PORT; 0: off),
 * --data <dir> (default: data/<setup.name> of the source tree). Declared game flags (e.g. --demo) are passed on in
 * options.gameFlags; anything else is an error. No config.ini.
 */
SAIGA_ENGINE_API bool parseCommandLine(int argc, const char* const* argv, const std::vector<std::string>& gameFlags,
                                       EngineAppSetup& setup, AppOptions& options, std::string& error);

// Runs the game's scenarios (EngineAppSetup::scenarios) in all three variants, headless and without a device: one line
// per scenario and variant, the reports of the failures, the count. 0 if at least one ran and all passed, else 1
SAIGA_ENGINE_API int runGameScenarios(const EngineAppSetup& setup, const std::string& filter);

// main of an engine game: parses the command line, then runs the scenarios, or creates the app, calls start (the game
// spawns its world, sets the local player; may be empty) and runs it. Returns the exit code.
SAIGA_ENGINE_API int engineMain(int argc, const char* const* argv, EngineAppSetup setup,
                                const std::vector<std::string>& gameFlags,
                                const std::function<void(EngineApp&, const AppOptions&)>& start);

}  // namespace Engine
}  // namespace Saiga
