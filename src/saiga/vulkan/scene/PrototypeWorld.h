/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"
#include "saiga/export.h"

#include <memory>

namespace Saiga
{
namespace Vulkan
{
struct SceneLighting;

/**
 * The settings of the prototype content of RenderScene (S19a, docs/engine_design.md §10.1; S20: the terrain, grass,
 * water and sky of the new path, the environment of setPrototypeEnvironment until S36): the game's WorldSettings map to
 * it (R-MECH-1: the toggles, distances and light stay game settings). The tree LOD distances and the wind's bounds
 * padding are mesh data (RenderScene::setMeshLodDistances, PrototypeForest).
 */
struct PrototypeSettings
{
    // The passes (racing turns the grass off); terrain = false keeps the water plane
    bool terrain = true, grass = true, sky = true, water = true;
    // The depth pre-pass of the opaque content (terrain, water, opaque instances; §10.5 pass 2). Also read without the
    // environment (the opaque instances). Off by default: the terrain already draws first, so the pre-pass only removes
    // terrain self-overdraw and costs a second CDLOD vertex pass (measured slower in S19a); the mechanism stays for S37
    bool depthPrePass = false;
    bool fog          = true;
    float fogDensity  = 0.00015f;
    float fogFalloff  = 0.0012f;
    // Degrees
    float sunAzimuth   = 35.0f;
    float sunElevation = 38.0f;
    float sunIntensity = 1.0f;
    float ambient      = 0.25f;
    float albedo       = 0.55f;
    // The wind strength of the trees' and the grass's shaders (the trees' bounds padding is mesh data)
    float wind = 1.0f;
    // CDLOD: the range of level 0 (m) and the morph start as a fraction of a level's range
    float terrainLodRange = 300.0f;
    float morphStart      = 0.7f;
    float grassRadius     = 60.0f;
    float grassCellSize   = 0.3f;
    float grassDensity    = 0.85f;
    // The foliage's alpha cutoff (trees and grass)
    float alphaCutoff = 0.45f;
    // 0: off, 1: tint by LOD (terrain levels, the trees' LOD bits)
    int debugMode    = 0;
    bool flatShading = true;
    // The skirts of the S20 terrain nodes (TerrainPass); off only for debugging and the crack test
    bool terrainSkirts = true;
};

// The lighting block of the instance draws from the prototype settings (sun, hemisphere ambient, fog with its base
// height, wind phases of the time, cutoff, tint): RenderScene's environment (S20)
SAIGA_VULKAN_API void prototypeLighting(const PrototypeSettings& settings, double time, float fogBase,
                                        SceneLighting& out);

// The color word of a PrototypeFoliage vertex (S19a): the foliage texture layer in bits 0-7 (a negative layer: opaque
// parts, 0xFF), the AO in bits 8-15 and the wind weight in bits 16-23 (unorm8; both asserted in [0, 1])
SAIGA_VULKAN_API uint32_t packPrototypeFoliageColor(int layer, float ao, float wind);

// The wind phase fmod(time * frequency, 2 pi) in double (§10.1: no absolute time in a shader)
SAIGA_VULKAN_API float prototypeWindPhase(double time, double frequency);

}  // namespace Vulkan
}  // namespace Saiga
