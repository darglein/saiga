/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"

namespace Saiga
{
namespace Vulkan
{
/**
 * The lighting block of the scene draws (std140, shader/vulkan/scene/lighting/scene_block.glsl; S19a, v2 in S36a1b):
 * binding 0 of set 1 (SceneLightingSet), read by the instance, terrain, water, grass and sky pipelines.
 *
 * The first seven vec4 are the legacy S19a light model of the default pipeline variant (LightingParameters::pbr =
 * false; removed in the a1c flip): the defaults are S18's fixed light (sun normalize(0.4, 1, 0.7) with 0.8, ambient
 * 0.35, no hemisphere term, fog off), RenderScene::setPrototypeEnvironment takes the prototype's sun, ambient, fog and
 * wind. The rest is S36's (the LIGHTING_PBR variant, shadeSurface of lighting.glsl): the sun and moon in lux, the
 * exposure, the studio environment of clay mode, the active fog, the ambient floor, the enclosed space, the cloud
 * layer, the stars, the prototype albedo colours (D21) and the debug view. Camera-relative or float render data only.
 */
struct SceneLighting
{
    // ---- legacy (S19a) ----
    // xyz: towards the sun (normalized), w: intensity
    vec4 sunDir = vec4(0.3113996f, 0.7784989f, 0.5449493f, 0.8f);
    // x: ambient, y: 1 hemisphere ambient (0.6 + 0.4 n.y, the prototype's) else constant, z: albedo (foliage clay), w:
    // the foliage's alpha cutoff (also read by the LIGHTING_PBR variant)
    vec4 ambient = vec4(0.35f, 0, 0.55f, 0.45f);
    // x: density, y: height falloff, z: enabled (0/1), w: the fog's base height (the water level)
    vec4 fog      = vec4(0, 0, 0, 0);
    vec4 fogColor = vec4(0.42f, 0.42f, 0.42f, 1);
    // xyz: the camera position (float, the space's coordinates: the foliage's wind phase and the fog height), w: the
    // MSAA samples (both variants)
    vec4 camera = vec4(0, 0, 0, 1);
    // xyz: the tree wind phases fmod(time * (1.1, 0.9, 4.0), 2 pi) (computed in double), w: the wind strength (both)
    vec4 wind = vec4(0, 0, 0, 0);
    // x: debug mode (1: tint by LOD; both variants)
    vec4 debug = vec4(0, 0, 0, 0);

    // ---- S36 (LIGHTING_PBR) ----
    // xyz: towards the sun, w: its illuminance (lux at the top of the atmosphere; a1c: the transmittance LUT per pixel)
    vec4 sun = vec4(0, 0.70710678f, 0.70710678f, 120000);
    // xyz: towards the moon, w: its illuminance (lux, the phase's lit fraction applied)
    vec4 moon = vec4(0, -0.70710678f, -0.70710678f, 0);
    // xyz: towards the shadow light (A5), w: 0 none, 1 the sun, 2 the moon
    vec4 shadowLight = vec4(0, 1, 0, 0);
    // x: the effective exposure of the frame, y: the pre-exposure, z: EV100, w: 1 the scene writes linear values
    vec4 exposure = vec4(1, 1, 15, 0);
    // x: 0 studio (clay, or no environment), 1 physical (an environment and a tone map); y: 1 an enclosed space; z: the
    // debug view (LightingDebugView); w: the overlay flags
    uvec4 mode = uvec4(0, 0, 0, 0);
    // The studio environment (clay mode, A3): xyz towards the studio sun (normalized), w: its intensity
    vec4 studioSun = vec4(0.3113996f, 0.7784989f, 0.5449493f, 0.8f);
    // x: ambient, y: 1 hemisphere ambient, z: the clay albedo of the prototype's surfaces, w: unused
    vec4 studioShading = vec4(0.35f, 0, 0.55f, 0);
    // The active fog (studio or physical): x density, y height falloff, z enabled, w: the camera height above the fog
    // base (m, computed in double)
    vec4 fogParams   = vec4(0, 0, 0, 0);
    vec4 fogColorLin = vec4(0.42f, 0.42f, 0.42f, 1);
    // xyz: the ambient floor (lux-equivalent), w: the ambient scale
    vec4 minAmbient = vec4(0.002f, 0.002f, 0.002f, 1);
    // xyz: an enclosed space's ambient (lux-equivalent), w: unused
    vec4 enclosedAmbient = vec4(0, 0, 0, 0);
    // The cloud layer (a1c): x altitude, y radius, z cloudiness, w the shadow fade distance (m)
    vec4 clouds = vec4(2000, 40000, 0.3f, 8000);
    // xy: the wrapped cloud offset (m), zw: the wind (m/s)
    vec4 cloudMotion = vec4(0, 0, 3, 1);
    // The stars' rotation (rows of a mat3 in xyz)
    vec4 starRotation[3] = {vec4(1, 0, 0, 0), vec4(0, 1, 0, 0), vec4(0, 0, 1, 0)};
    // The prototype albedo colours (D21, linear; the physical path only): rgb, w of albedoGround: the rock slope (n.y)
    vec4 albedoGround = vec4(0.26f, 0.30f, 0.16f, 0.75f);
    vec4 albedoRock   = vec4(0.36f, 0.34f, 0.31f, 0);
    vec4 albedoShore  = vec4(0.50f, 0.46f, 0.36f, 0);
    vec4 albedoWater  = vec4(0.03f, 0.06f, 0.07f, 0);
    vec4 albedoGrass  = vec4(0.20f, 0.30f, 0.10f, 0);
    vec4 albedoLeaves = vec4(0.11f, 0.20f, 0.07f, 0);
    vec4 albedoBark   = vec4(0.25f, 0.19f, 0.14f, 0);
    // EnvironmentState::phases
    vec4 phases[2] = {vec4(0, 0, 0, 0), vec4(0, 0, 0, 0)};
    // x: the camera height above the sea level (m, double; the shore band), y: 1 a sea level exists
    vec4 sea = vec4(0, 0, 0, 0);
    // The debug views (A8): x: the Exposure view's centre (EV100), y: its range (EV), zw: unused
    vec4 debugParams = vec4(0, 8, 0, 0);
};
static_assert(sizeof(SceneLighting) == 34 * 16, "SceneLighting of shader/vulkan/scene/lighting/scene_block.glsl");

}  // namespace Vulkan
}  // namespace Saiga
