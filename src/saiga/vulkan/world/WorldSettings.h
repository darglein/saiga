/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/json11.hpp"
#include "saiga/export.h"
#include "saiga/vulkan/scene/PrototypeWorld.h"

#include <string>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
// The shading mode of the world. Clay: uniform gray with simple lighting, to judge the geometry (also the debug mode of
// the engine's renderer later, see docs/engine_design.md §10.5).
enum class WorldShading
{
    Clay
};

/**
 * The game settings of the world prototype that can be changed at run time (GUI and MCP's game_set_settings, see
 * toJson / fromJson; R-MECH-1). S19a: they map to RenderScene's PrototypeSettings (prototype()) and to the tree meshes'
 * LOD distances and bounds padding (PrototypeForest::apply); fovy is the game camera's.
 */
struct SAIGA_VULKAN_API WorldSettings
{
    WorldShading shading = WorldShading::Clay;

    bool terrain = true, trees = true, grass = true, water = true, sky = true;
    // The depth pre-pass of the opaque content (terrain, water, opaque instances; S19a): off for the prototype, whose
    // terrain already draws first (the pre-pass measured slower: docs/stages/S19a_prototype_passes.plan.md, Results)
    bool depthPrePass = false;
    // 0: off, 1: tint by level of detail
    int debugMode    = 0;
    bool flatShading = true;

    float fovy = 60.0f;

    bool fog           = true;
    float fogDensity   = 0.00015f;
    float fogFalloff   = 0.0012f;
    float sunAzimuth   = 35.0f;
    float sunElevation = 38.0f;
    float sunIntensity = 1.0f;
    float ambient      = 0.25f;
    float albedo       = 0.55f;
    float wind         = 1.0f;

    float terrainLodRange = 300.0f;
    float morphStart      = 0.7f;

    float treeLod1        = 70.0f;
    float treeLod2        = 300.0f;
    float treeMaxDistance = 30000.0f;

    float grassRadius   = 60.0f;
    float grassCellSize = 0.3f;
    float grassDensity  = 0.85f;
    float alphaCutoff   = 0.45f;

    json11::Json toJson() const;
    // Applies the members present in the json object and clamps all values (clamp()). Returns the names of unknown
    // keys and of keys with an invalid value (wrong JSON type, unknown shading mode); those are not applied.
    std::vector<std::string> fromJson(const json11::Json& json);
    // Limits all values to the ranges of the GUI.
    void clamp();
    // RenderScene's settings of the prototype content (passes, light, fog, wind, terrain and grass LOD)
    PrototypeSettings prototype() const;
};

}  // namespace Vulkan
}  // namespace Saiga
