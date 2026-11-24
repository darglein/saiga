/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/json11.hpp"
#include "saiga/export.h"
#include "saiga/vulkan/scene/LightingTypes.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"

#include <functional>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
// set_lighting's keys (S36; round 3, nit 11: in engine/lighting/, so the environment file's loader does not depend on
// the app's MCP tools): each reads its value from a LightingState and writes a validated value into a scratch copy. The
// table is shared by set_lighting / get_lighting, the Lighting tab (validateLighting) and the environment file.
struct LightingState
{
    Vulkan::EnvironmentSettings settings;
    Vulkan::LightingParameters parameters;
    Vulkan::ToneMap toneMap = Vulkan::ToneMap::Identity;
};

struct LightingKey
{
    const char* name;
    bool parameter;  // a LightingParameters field (get_lighting.parameters), else a setting (get_lighting.settings)
    std::function<json11::Json(const LightingState&)> get;
    std::function<bool(const json11::Json&, LightingState&, std::string&)> set;
    // The schema's description (the range of a number key)
    std::string description = {};
};

// Every key (also the ones a later part adds), in the tool's order
SAIGA_ENGINE_API const std::vector<LightingKey>& lightingKeys();

// The settings and parameters against set_lighting's ranges (every key), its cross-key rules (cluster_near <
// cluster_far, atlas_tile divides atlas_resolution) and the exposure curve (not empty, finite, strictly ascending
// elevations); false and an error naming every invalid key (the set_lighting key's name). Used by the environment file
// (tryReadEnvironmentFile) and validateLighting
SAIGA_ENGINE_API bool validateLightingValues(const Vulkan::EnvironmentSettings& settings,
                                             const Vulkan::LightingParameters& parameters, std::string& error);

// "[lo, hi]" with the trailing zeros removed
SAIGA_ENGINE_API std::string lightingRangeText(double lo, double hi);
// A finite number in [lo, hi] (the error names the argument and the range)
SAIGA_ENGINE_API bool lightingNumber(const json11::Json& j, const std::string& name, double lo, double hi, double& out,
                                     std::string& error);
// The JSON names of the tone map and the exposure mode
SAIGA_ENGINE_API const char* toneMapName(Vulkan::ToneMap t);
SAIGA_ENGINE_API const char* exposureModeName(Vulkan::ExposureMode m);

}  // namespace Engine
}  // namespace Saiga
