/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/json11.hpp"
#include "saiga/export.h"
#include "saiga/vulkan/scene/LightingTypes.h"

#include <string>
#include <vector>

namespace Saiga
{
class McpServer;

namespace Engine
{
class EngineApp;
class World;

/**
 * The lighting's MCP tools (docs/engine_design.md §16.2, §10.7; S36), registered after the app tools; each is also a
 * console command (DebugConsole), e.g. set_time_of_day hours=18.5:
 * - get_lighting(sections?): time, weather, sun, moon, the shadow light, the tone map, exposure, ambient, clouds, the
 *   settings and the lighting parameters (answered at once; later parts add keys). Changes nothing.
 * - set_time_of_day(hours?, day?, rate?), set_weather(space?, ...): world state through setWorldTime / setWeather
 *   (validated; in --coop applied to the hidden host, on a pure client refused).
 * - set_lighting(...): the EnvironmentSettings, the LightingParameters and the tone map, validated as a whole and
 *   applied all or nothing from the next frame; returns values and previous (an undo payload).
 * Unknown arguments and out-of-range values are errors naming the argument; without a device get_lighting and
 * set_lighting answer "this app renders no frames (no device)".
 */
SAIGA_ENGINE_API void registerLightingTools(McpServer& server, EngineApp& app);

// The settings and parameters against set_lighting's ranges (every key, also the ones a later part adds) and the
// exposure curve (not empty, finite, strictly ascending elevations); false and an error naming every invalid key (the
// set_lighting key's name). Used by the environment file (tryReadEnvironmentFile)
SAIGA_ENGINE_API bool validateLightingValues(const Vulkan::EnvironmentSettings& settings,
                                             const Vulkan::LightingParameters& parameters, std::string& error);

// The world the lighting tools and the Lighting tab write: the hidden host with --coop, else the app's world
SAIGA_ENGINE_API World& lightingWriteWorld(EngineApp& app);

// get_lighting's answer (sections: the top-level keys to keep; empty: all); false and an error for an unknown section
SAIGA_ENGINE_API bool lightingJson(EngineApp& app, const std::vector<std::string>& sections, json11::Json& out,
                                   std::string& error);

// app_status's lighting block: {time_of_day, ev100, tone_map, debug_view}; null without a device
SAIGA_ENGINE_API json11::Json lightingStatusJson(EngineApp& app);

}  // namespace Engine
}  // namespace Saiga
