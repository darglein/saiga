/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/json11.hpp"
#include "saiga/engine/lighting/LightingKeys.h"
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
 * - set_lighting_debug(view?, overlays?, mapping?) (S36a1b): the debug view, the overlays and the tab's preview
 * mapping.
 * - capture_lighting(image, ...) (S36a1b): a preview image (luminance, brdf_lut) with the numbers of its source values,
 *   or the numbers alone (stats_only); deferred to the frame that computes it.
 * - probe_lighting(pixel? | position?) (S36a1b): the 7-frame probe sequence of RenderScene::requestProbe at a pixel.
 * Unknown arguments and out-of-range values are errors naming the argument; without a device get_lighting and
 * set_lighting answer "this app renders no frames (no device)".
 */
SAIGA_ENGINE_API void registerLightingTools(McpServer& server, EngineApp& app);

// validateLightingValues: saiga/engine/lighting/LightingKeys.h (included)

// The lighting memory after a change of the parameters (round 2, finding 2): the memory now (LightingStatistics) with
// the cascades, the horizon maps, the shadow atlas and the cluster buffers replaced by their sizes for the parameters;
// largestBinding: the largest single storage binding of the parameters (a frame slot's cluster light indices)
struct LightingMemoryEstimate
{
    uint64_t total          = 0;
    uint64_t largestBinding = 0;
};
SAIGA_ENGINE_API LightingMemoryEstimate lightingMemoryAfter(const Vulkan::LightingStatistics& now,
                                                            const Vulkan::LightingParameters& after);
// The lighting memory limit of set_lighting and the Lighting tab
constexpr uint64_t kLightingMemoryLimit = uint64_t(1) << 30;

// The whole check of a lighting state (round 2, findings 2, 9; set_lighting and every write of the Lighting tab):
// validateLightingValues, then the memory (lightingMemoryAfter within kLightingMemoryLimit, the largest binding within
// storageLimit, the device's maxStorageBufferRange); false and an error naming the keys. memory: the estimate
SAIGA_ENGINE_API bool validateLighting(const Vulkan::LightingStatistics& now, uint64_t storageLimit,
                                       const Vulkan::EnvironmentSettings& settings,
                                       const Vulkan::LightingParameters& parameters, std::string& error,
                                       uint64_t* memory = nullptr);

// The lighting's GPU sections (get_lighting.gpu and the tab's GPU section; round 2, finding 8: one list): the 120-frame
// mean of each, lighting_ms (the sum against the §17.10 budget of 2.5 ms; the parent section "shadow" is reported but
// not summed, it encloses shadow.cascades, shadow.horizon and shadow.atlas) and debug_ms (the debug lines and previews)
struct LightingGpuTimes
{
    struct Section
    {
        std::string name;
        bool debug = false, summed = true;
        double meanMs = 0, lastMs = 0;
        bool hasLast = false;
    };
    std::vector<Section> sections;
    double lightingMs = 0, debugMs = 0;
    // The summed lighting time of each kept frame of the GPU profile, oldest first (the tab's summary graph)
    std::vector<double> lightingHistory;
    // The mean of the largest 1% of lightingHistory (at least one frame): the long frames that the mean hides
    double lightingWorst1PercentMs = 0;
    static constexpr double kBudgetMs = 2.5;
    static constexpr int kMeanFrames  = 120;
};
SAIGA_ENGINE_API LightingGpuTimes lightingGpuTimes(EngineApp& app);

// The world the lighting tools and the Lighting tab write: the hidden host with --coop, else the app's world
SAIGA_ENGINE_API World& lightingWriteWorld(EngineApp& app);

// get_lighting's answer (sections: the top-level keys to keep; empty: all); false and an error for an unknown section
SAIGA_ENGINE_API bool lightingJson(EngineApp& app, const std::vector<std::string>& sections, json11::Json& out,
                                   std::string& error);

// The debug view and overlays of the arguments (lighting_view: a LightingDebugView name, overlays: a list of overlay
// names; both optional, missing: None / no overlays); false and an error naming the argument and the known names
SAIGA_ENGINE_API bool parseLightingDebug(const json11::Json& view, const json11::Json& overlays,
                                         Vulkan::LightingDebugState& out, std::string& error);
SAIGA_ENGINE_API json11::Json lightingDebugJson(const Vulkan::LightingDebugState& state);

// app_status's lighting block: {time_of_day, ev100, tone_map, debug_view}; null without a device
SAIGA_ENGINE_API json11::Json lightingStatusJson(EngineApp& app);

}  // namespace Engine
}  // namespace Saiga
