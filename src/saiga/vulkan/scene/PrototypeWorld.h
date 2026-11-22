/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"
#include "saiga/export.h"
#include "saiga/vulkan/scene/LightingTypes.h"

#include <memory>

namespace Saiga
{
namespace Vulkan
{
struct SceneLighting;

// The prototype's surface colours of the physical light path (S36, D21; linear albedo; S43 replaces them with
// materials): the terrain blends ground -> rock by slope (smoothstep on n.y around rockSlope) and ground -> shore in
// the 3 m above the sea level; the water plane, the grass, the foliage cards (leaves) and the opaque tree parts (bark).
// Clay mode ignores them (the gray albedo of the studio environment)
struct PrototypeAlbedo
{
    vec3 ground     = vec3(0.17f, 0.25f, 0.09f);
    vec3 rock       = vec3(0.36f, 0.34f, 0.31f);
    vec3 shore      = vec3(0.50f, 0.46f, 0.36f);
    vec3 water      = vec3(0.03f, 0.06f, 0.07f);
    vec3 grass      = vec3(0.14f, 0.27f, 0.07f);
    vec3 leaves     = vec3(0.08f, 0.17f, 0.05f);
    vec3 bark       = vec3(0.25f, 0.19f, 0.14f);
    float rockSlope = 0.75f;
};

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
    // The surface colours of the physical light path (S36, D21; not saved, the Lighting tab edits them)
    PrototypeAlbedo colors;
};

// The fixed lighting of an enclosed camera space (S21, until S36): the PrototypeSettings defaults (sun azimuth 35,
// elevation 38, intensity 1, ambient 0.25, albedo 0.55) without fog; independent of the game's
// setPrototypeEnvironment (its sun and fog belong to the outdoors)
inline const PrototypeSettings kEnclosedLighting = []()
{
    PrototypeSettings s;
    s.fog = false;
    return s;
}();

// The lighting block of the instance draws from the prototype settings (sun, hemisphere ambient, fog with its base
// height, wind phases of the time, cutoff, tint): RenderScene's environment (S20)
SAIGA_VULKAN_API void prototypeLighting(const PrototypeSettings& settings, double time, float fogBase,
                                        SceneLighting& out);

// The studio environment of clay mode from the prototype settings (S36, A3: S19a's sun, hemisphere ambient, albedo and
// fog reproduced; until the a1c migration moves them out of PrototypeSettings)
SAIGA_VULKAN_API StudioEnvironment studioFromPrototype(const PrototypeSettings& settings);

// The color word of a PrototypeFoliage vertex (S19a): the foliage texture layer in bits 0-7 (a negative layer: opaque
// parts, 0xFF), the AO in bits 8-15 and the wind weight in bits 16-23 (unorm8; both asserted in [0, 1])
SAIGA_VULKAN_API uint32_t packPrototypeFoliageColor(int layer, float ao, float wind);

// The wind phase fmod(time * frequency, 2 pi) in double (§10.1: no absolute time in a shader)
SAIGA_VULKAN_API float prototypeWindPhase(double time, double frequency);

}  // namespace Vulkan
}  // namespace Saiga
