#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// The PrototypeFoliage pipeline class of RenderScene (S19a): the prototype's foliage.frag with the lighting block. The
// opaque parts (layer < 0) and the alpha tested leaf cards (alpha to coverage with MSAA, the alpha cutoff of the
// block); clay lighting with AO, the LOD tint from the compacted entry's LOD bits, height fog, the prototype's output.

#include "../world/prototype_shading.glsl"
#include "scene_lighting.glsl"

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

layout(set = 1, binding = 1) uniform sampler2DArray foliage;

layout(location = 0) out vec4 outColor;

void main()
{
    float alpha = 1.0;
    if (inUvLayer.z >= 0.0)
    {
        alpha = foliageCoverage(foliage, inUvLayer, lighting.ambient.w);
        if (alpha <= 0.0 || (lighting.camera.w < 1.5 && alpha < 0.5)) discard;
    }

    vec3 N = normalize(inNormal);
    vec3 c = protoClayLighting(N, inAo, lighting.sunDir.xyz, lighting.sunDir.w, lighting.ambient.z, lighting.ambient.x);
    if (lighting.debug.x > 0.5) c *= lodTint(inLod);
    c        = protoApplyFog(c, inPos, lighting.camera.y, lighting.fog, lighting.fogColor.rgb);
    outColor = protoFinalColor(c, clamp(alpha, 0.0, 1.0), pc.linear != 0u ? pc.preExposure : 0.0);
}
