#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// The grass of S20 (a copy of world/foliage.frag on TerrainFrame): alpha tested cards (alpha to coverage with MSAA).
// The foliage texture comes from RenderScene::setPrototypeEnvironment (binding 1 of set 1, scene_block.glsl; S36).
// LIGHTING_PBR (S36): shadeSurface with the grass colour (wrapped diffuse) in the physical path, the studio in clay.

#include "terrain_common.glsl"
#include "terrain_shading.glsl"
#ifdef LIGHTING_PBR
#    include "lighting/lighting.glsl"
#else
#    include "lighting/scene_block.glsl"
#endif

// Position relative to the camera
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inUvLayer;
layout(location = 3) in float inAo;
layout(location = 4) flat in float inLod;

layout(location = 0) out vec4 outColor;

void main()
{
    float alpha = 1.0;
    if (inUvLayer.z >= 0.0)
    {
        alpha = foliageCoverage(foliage, inUvLayer, frame.grassPhases.w);
        if (alpha <= 0.0 || (frame.grass.w < 1.5 && alpha < 0.5)) discard;
    }

    vec3 N = normalize(inNormal);
#ifdef LIGHTING_PBR
    SurfaceInput si = surfaceInput(prototypeAlbedo(lighting.albedoGrass.rgb), N, inPos);
    si.ao           = inAo;
    si.roughness    = 0.7;
    si.flags        = surfaceFoliage;
    if (frame.flags.z == 1) si.tint = lodTint(inLod);
    vec3 c = shadeSurface(si);
#else
    vec3 c = clayLighting(N, inAo);
    if (frame.flags.z == 1) c *= lodTint(inLod);
    c = applyFog(c, inPos);
#endif
#ifdef LIGHTING_PBR
    if (debugViewActive())
    {
        outColor = vec4(c, clamp(alpha, 0.0, 1.0));
        return;
    }
#endif
    outColor = finalColor(c, clamp(alpha, 0.0, 1.0));
}
