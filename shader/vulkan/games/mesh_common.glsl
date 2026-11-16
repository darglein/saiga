/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The uniforms and push constants of the game meshes, see Saiga::Game::MeshRenderer (games/common/MeshRenderer.h).
layout(set = 0, binding = 0) uniform GameUBO
{
    // Projection * view without the translation (camera relative)
    mat4 viewProjRel;
    // xyz: camera position (world)
    vec4 cameraPos;
    // xyz: towards the sun, w: intensity
    vec4 sunDir;
    // x: ambient, z: unused, w: sRGB encode in the shader
    vec4 shading;
    // x: density, y: height falloff, z: enabled, w: fog base height (water level)
    vec4 fog;
    vec4 fogColor;
}
game;

layout(push_constant) uniform PushConstants
{
    // Model matrix with the translation relative to the camera
    mat4 modelRel;
    // Columns of the normal matrix
    vec4 normal0;
    vec4 normal1;
    vec4 normal2;
    // rgb: color multiplier
    vec4 tint;
}
pc;
