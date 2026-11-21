#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"
#include "shading.glsl"

// Position relative to the camera
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) flat in float inLod;

layout(location = 0) out vec4 outColor;

void main()
{
    bool water = inLod < 0.0;
    vec3 N;
    if (world.shading.z > 0.5 && !water)
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
    if (world.terrain2.y > 0.5 && !water) c *= lodTint(inLod);
    c        = applyFog(c, inPos);
    outColor = finalColor(c, 1.0);
}
