#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// The terrain and the water plane of S20. LIGHTING_PBR (S36): shadeSurface with the prototype's colours (ground ->
// rock by slope, the shore band above the sea level, the water) in the physical path, the studio environment in clay.

#include "terrain_common.glsl"
#include "terrain_shading.glsl"
#ifdef LIGHTING_PBR
#    include "lighting/lighting.glsl"
#endif

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
#ifdef LIGHTING_PBR
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
    // The clay water: S19a's 0.62 on the lit gray
    if (water && !physicalLighting()) si.tint = vec3(0.62);
    if (frame.flags.z == 1 && !water) si.tint = lodTint(inLod);
    vec3 c = shadeSurface(si);
#else
    vec3 c = clayLighting(N, 1.0);
    if (water) c *= 0.62;
    if (frame.flags.z == 1 && !water) c *= lodTint(inLod);
    c = applyFog(c, inPos);
#endif
#ifdef LIGHTING_PBR
    if (debugViewActive())
    {
        outColor = vec4(c, 1.0);
        return;
    }
#endif
    outColor = finalColor(c, 1.0);
}
