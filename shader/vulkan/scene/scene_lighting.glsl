/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The lighting block of RenderScene's instance draws (S19a, Saiga::Vulkan::SceneLighting in SceneCulling.h; replaced
// by S36's lighting): S18's fixed light by default, the prototype's sun, ambient, fog and wind with
// RenderScene::setPrototypeEnvironment. Set 1 of the draw pipelines (binding 1: the foliage texture of the PrototypeFoliage
// class).
layout(std140, set = 1, binding = 0) uniform SceneLightingBlock
{
    // xyz: towards the sun (normalized), w: intensity
    vec4 sunDir;
    // x: ambient, y: 1 hemisphere ambient (0.6 + 0.4 n.y) else constant, z: albedo (foliage clay), w: alpha cutoff
    vec4 ambient;
    // x: density, y: height falloff, z: enabled, w: base height (water level)
    vec4 fog;
    vec4 fogColor;
    // xyz: the camera position (float), w: MSAA samples
    vec4 camera;
    // xyz: the tree wind phases (1.1, 0.9, 4.0 rad/s, modulo 2 pi), w: the wind strength
    vec4 wind;
    // x: 1 tint by LOD
    vec4 debug;
}
lighting;
