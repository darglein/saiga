#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// The terrain and the water plane of S20, lit by shadeSurface (S36) with the prototype's colours (ground -> rock by
// slope, the shore band above the sea level, the water) in the physical path, the studio environment in clay.

#include "terrain_common.glsl"
#include "terrain_shading.glsl"
// S36a1c review: the light terms come from the vertex shader
#define VERTEX_LIGHT_INPUT
#include "lighting/lighting.glsl"

// Position relative to the camera
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) flat in float inLod;
layout(location = 3) in float inSkirt;
layout(location = 4) in vec3 inSunE;
layout(location = 5) in vec3 inMoonE;
layout(location = 6) in vec4 inFog;

layout(location = 0) out vec4 outColor;

void main()
{
    vertexLightInput = VertexLight(inSunE, inMoonE, inFog);
    bool water       = inLod < 0.0;
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
    vec3 albedo;
    if (water)
    {
        albedo = lighting.albedoWater.rgb;
    }
    else
    {
        // ground -> rock by slope, ground -> shore in the 3 m above the sea level
        float rock  = 1.0 - smoothstep(lighting.albedoGround.w - 0.05, lighting.albedoGround.w + 0.05, N.y);
        float above = lighting.sea.x + inPos.y;
        float shore = lighting.sea.y > 0.5 ? 1.0 - smoothstep(0.0, 3.0, above) : 0.0;
        albedo      = mix(mix(lighting.albedoGround.rgb, lighting.albedoShore.rgb, shore), lighting.albedoRock.rgb, rock);
    }
    SurfaceInput si = surfaceInput(prototypeAlbedo(albedo), N, inPos);
    si.roughness    = water ? 0.08 : 0.9;
    // The land's specular environment from the SH (rough, dielectric)
    if (!water) si.flags = surfaceShSpecular;
    // The clay water: S19a's 0.62 on the lit gray
    if (water && !physicalLighting()) si.tint = vec3(0.62);
    if (frame.flags.z == 1 && !water) si.tint = lodTint(inLod);
    vec3 c = shadeSurface(si);
    if (debugViewActive())
    {
        outColor = vec4(c, 1.0);
        return;
    }
    outColor = finalColor(c, 1.0);
}
