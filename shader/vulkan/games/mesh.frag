#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

#include "mesh_common.glsl"

layout(location = 0) in vec3 inRelPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec4 outColor;

// The same exponential height fog as the world (shader/vulkan/world/shading.glsl), bounded exponentials.
float fogAbove(float L, float hA, float hB, float a, float b)
{
    float eA = exp(-b * hA);
    float eB = exp(-b * hB);
    float dh = hB - hA;
    return abs(b * dh) > 1e-4 ? a * L * (eA - eB) / (b * dh) : a * L * 0.5 * (eA + eB);
}

vec3 applyFog(vec3 color, vec3 relPos)
{
    if (game.fog.z < 0.5) return color;
    float dist = length(relPos);
    float a    = game.fog.x;
    float b    = game.fog.y;
    float y0   = game.cameraPos.y - game.fog.w;
    float y1   = y0 + relPos.y;
    float optical;
    if (y0 <= 0.0 && y1 <= 0.0)
    {
        optical = a * dist;
    }
    else if (y0 > 0.0 && y1 > 0.0)
    {
        optical = fogAbove(dist, y0, y1, a, b);
    }
    else
    {
        float below = dist * (y0 <= 0.0 ? -y0 : -y1) / abs(y1 - y0);
        optical     = a * below + fogAbove(dist - below, 0.0, max(y0, y1), a, b);
    }
    return mix(color, game.fogColor.rgb, clamp(1.0 - exp(-optical), 0.0, 1.0));
}

void main()
{
    vec3 N = normalize(inNormal);
    // Two sided (no back face culling): light the side that faces the camera
    if (dot(N, inRelPos) > 0.0) N = -N;
    vec3 L        = normalize(game.sunDir.xyz);
    float diffuse = max(dot(N, L), 0.0) * game.sunDir.w;
    float hemi    = game.shading.x * (0.6 + 0.4 * N.y);
    vec3 c        = inColor * (diffuse + hemi);
    c             = applyFog(c, inRelPos);
    c             = clamp(c, 0.0, 1.0);
    if (game.shading.w > 0.5) c = pow(c, vec3(1.0 / 2.2));
    outColor = vec4(c, 1.0);
}
