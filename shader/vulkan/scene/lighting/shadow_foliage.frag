#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// The PrototypeFoliage class in the cascades (S36a2, A5): the leaf cards' alpha of the foliage texture (binding 1 of
// the shadow set) against the block's alpha cutoff (no alpha to coverage: a hard cutoff); the opaque parts pass.

#include "scene_block.glsl"

layout(location = 0) in vec3 inUvLayer;

#ifdef ATLAS_DEPTH
// S36b: the shadow atlas: the linear depth distance(light, p) / range (D16: range / 65536 per step at every distance)
layout(location = 1) in vec3 inRelPos;
layout(push_constant) uniform AtlasPush
{
    mat4 viewProj;
    vec4 pad;
    vec4 light;  // xyz the light (camera-relative), w 1 / range
}
atlasPc;
#endif

void main()
{
    if (inUvLayer.z >= 0.0 && texture(foliage, inUvLayer).r < lighting.foliageParams.x) discard;
#ifdef ATLAS_DEPTH
    gl_FragDepth = clamp(length(inRelPos - atlasPc.light.xyz) * atlasPc.light.w, 0.0, 1.0);
#endif
}
