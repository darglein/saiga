#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// The PrototypeFoliage pipeline class of RenderScene (S19a): the prototype's foliage.frag with the lighting block. The
// opaque parts (layer < 0) and the alpha tested leaf cards (alpha to coverage with MSAA, the alpha cutoff of the
// block); the LOD tint from the compacted entry's LOD bits, the prototype's output. The foliage texture is binding 1 of
// set 1 (scene_block.glsl). Lit by shadeSurface (S36) with the leaves' and the bark's colours (wrapped diffuse for the
// cards) and the AO in the physical path, the studio environment's clay albedo in clay mode.

#include "lighting/lighting.glsl"

layout(push_constant) uniform Push
{
    mat4 viewProj;
    float alpha;
    float preExposure;
    uint linear;
    uint pad;
}
pc;

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
        alpha = foliageCoverage(foliage, inUvLayer, lighting.foliageParams.x);
        if (alpha <= 0.0 || (lighting.camera.w < 1.5 && alpha < 0.5)) discard;
    }

    vec3 N = normalize(inNormal);
    bool card       = inUvLayer.z >= 0.0;
    SurfaceInput si = surfaceInput(prototypeAlbedo(card ? lighting.albedoLeaves.rgb : lighting.albedoBark.rgb), N, inPos);
    si.ao           = inAo;
    si.roughness    = card ? 0.6 : 0.9;
    if (card) si.flags |= surfaceFoliage;
    if (lighting.debug.x > 0.5) si.tint = lodTint(inLod);
    vec3 c = shadeSurface(si);
    if (debugViewActive())
    {
        outColor = vec4(c, clamp(alpha, 0.0, 1.0));
        return;
    }
    outColor = protoFinalColor(c, clamp(alpha, 0.0, 1.0), pc.linear != 0u ? pc.preExposure : 0.0);
}
