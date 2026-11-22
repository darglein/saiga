#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

#include "terrain_common.glsl"
#include "terrain_shading.glsl"

// Position relative to the camera
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) flat in float inLod;
layout(location = 3) in float inSkirt;

layout(location = 0) out vec4 outColor;

void main()
{
    bool water = inLod < 0.0;
    vec3 N;
    // A skirt fragment (a triangle with a skirt vertex) has the interpolated normal of its edge vertices: flat shading
    // would light the vertical skirt as a wall
    bool skirt = inSkirt > 0.0;
    if (frame.shading.z > 0.5 && !water && !skirt)
    {
        // Flat shading: the normal of the triangle
        N = normalize(cross(dFdx(inPos), dFdy(inPos)));
        if (N.y < 0.0) N = -N;
    }
    else
    {
        N = normalize(inNormal);
    }
    vec3 c = clayLighting(N, 1.0);
    if (water) c *= 0.62;
    if (frame.flags.z == 1 && !water) c *= lodTint(inLod);
    c        = applyFog(c, inPos);
    outColor = finalColor(c, 1.0);
}
