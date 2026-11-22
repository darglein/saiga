#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// The shadow atlas's depth (S36b, B4; finding 14 of the plan's review 1): the linear depth distance(light, p) / range
// into the D16 atlas (a step of range / 65536 at every distance; depth-only passes: no early-z to lose). The opaque
// instance classes (shadow_depth.vert) and the terrain (terrain_shadow.vert), both with ATLAS_DEPTH.

layout(location = 3) in vec3 inRelPos;

layout(push_constant) uniform AtlasPush
{
    mat4 viewProj;
    vec4 pad;
    vec4 light;  // xyz the light (camera-relative), w 1 / range
}
atlasPc;

void main()
{
    gl_FragDepth = clamp(length(inRelPos - atlasPc.light.xyz) * atlasPc.light.w, 0.0, 1.0);
}
