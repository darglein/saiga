/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The uniforms of the world passes, see Saiga::Vulkan::WorldUniforms (src/saiga/vulkan/world/WorldCommon.h).
layout(set = 0, binding = 0) uniform WorldUBO
{
    // Camera relative: projection * view without the translation. The world passes transform positions relative to
    // the camera (position - cameraPos, computed from small parts), so float precision does not depend on the
    // distance to the origin.
    mat4 viewProjRel;
    mat4 invViewProjRel;
    // xyz: camera position (world), w: time (s)
    vec4 cameraPos;
    // xyz: position used for the level of detail and culling
    vec4 lodCameraPos;
    // xyz: towards the sun, w: intensity
    vec4 sunDir;
    // x: albedo, y: ambient, z: flat shading, w: 0 display values (encoded in the shader), > 0 linear values times w
    // (the pre-exposure), see finalColor in shading.glsl
    vec4 shading;
    // x: density, y: height falloff, z: enabled, w: water level
    vec4 fog;
    vec4 fogColor;
    vec4 skyColor;
    // left, right, bottom, top, near, (dummy)
    vec4 frustum[6];
    // x: height map size, y: texel spacing, z: range of terrain level 0, w: morph start ratio
    vec4 terrain;
    // x: highest terrain level, y: debug mode, z: wind, w: alpha cutoff
    vec4 terrain2;
    // x: tree lod 1 distance, y: tree lod 2 distance, z: max tree distance, w: tree count
    vec4 trees;
    // x: grass radius, y: grass cell size, z: grass density, w: MSAA samples
    vec4 grass;
    vec4 viewport;
}
world;
