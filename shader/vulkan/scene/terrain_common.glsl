/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// TerrainFrame (S20): the per-frame uniforms of the terrain, water, grass and sky passes of RenderScene, see
// Saiga::Vulkan::TerrainFrameUniforms (src/saiga/vulkan/scene/TerrainPass.h). Everything is relative to the camera or
// to the frame anchor: the anchor is the camera's xz rounded down to 64 m (integer metres) and refHeight the camera
// height rounded down to 64 m (height units); cameraRel = camera - (anchor.x, refHeight * kHeightUnit, anchor.z),
// computed in double on the CPU.
layout(set = 0, binding = 0) uniform TerrainFrameUBO
{
    // Camera relative: projection * view without the translation
    mat4 viewProjRel;
    mat4 invViewProjRel;
    // xyz: cameraRel, w: unused
    vec4 cameraRel;
    // x, y: the anchor's x and z (m), z: refHeight (height units), w: the sea level (height units)
    ivec4 anchor;
    // x: samples per tile side, y: streamed levels, z: pyramid samples per side, w: base spacing (m)
    ivec4 grid;
    // x: a sea level exists, y: the root level, z: debug mode (1: tint by LOD), w: unused
    ivec4 flags;
    // Per streamed level: x the page-table offset, y the tiles per side
    ivec4 levels[8];
    // xyz: towards the sun, w: intensity
    vec4 sunDir;
    // x: albedo, y: ambient, z: flat shading, w: 0 display values, > 0 linear values times w (the pre-exposure)
    vec4 shading;
    // x: density, y: height falloff, z: enabled, w: the camera height above the fog base (m)
    vec4 fog;
    vec4 fogColor;
    vec4 skyColor;
    // x: the range of level 0 (m), y: the morph start ratio
    vec4 lod;
    // x: grass radius, y: grass cell size, z: grass density, w: MSAA samples
    vec4 grass;
    // x, y: the grass wind phases fmod(time * (1.7, 1.3), 2 pi), z: wind strength, w: alpha cutoff
    vec4 grassPhases;
    vec4 viewport;
}
frame;

const float kHeightUnit = 1.0 / 1024.0;
