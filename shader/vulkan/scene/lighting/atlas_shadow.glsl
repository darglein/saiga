/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The shadow atlas of the local lights at a receiver (S36b, B4 of docs/stages/S36_lighting.plan.md; ShadowAtlas): the
// light's face (a point light: by the major axis of the receiver's direction; a spot: its one tile) from the tile table
// (binding 14), the receiver's linear depth distance / range minus a bias of atlasParams.y D16 steps, offset along the
// normal by atlasParams.z texels of the face at its distance, one comparison of the 2 x 2 texels (bilinear PCF with the
// linear comparison sampler), clamped half a texel inside the tile; the
// minimum of the static and (when the tile has content) the dynamic layer. Requires scene_block.glsl.
#ifndef ATLAS_SHADOW_GLSL
#define ATLAS_SHADOW_GLSL

layout(set = 1, binding = 13) uniform sampler2DArrayShadow shadowAtlas;

// ShadowAtlas::GpuTile (std430, 96 bytes)
struct AtlasTile
{
    mat4 viewProj;
    vec4 rect;    // xy the tile's origin (uv), zw its size (uv)
    vec4 params;  // x a texel's size per m of distance, y 1 / tile texels, z 1 the dynamic layer has content
};
layout(std430, set = 1, binding = 14) readonly buffer AtlasTiles
{
    AtlasTile atlasTiles[];
};

// One 2 x 2 bilinear comparison at uv of a layer (the linear comparison sampler; D16 supports linear filtering on the
// devices of interest: else one nearest compare)
float atlasCompare(vec2 uv, float layer, float ref)
{
    return texture(shadowAtlas, vec4(uv, layer, ref));
}

// The visibility (0 shadowed .. 1 lit) of a light at a receiver (relPos, its normal n); 1 without rendered tiles
float atlasShadow(vec4 posRange, vec4 dirKind, vec4 intensity, vec4 cone, vec3 relPos, vec3 n)
{
    if (cone.z < 0.5 || intensity.w < 0.0 || lighting.lightCounts.w == 0u) return 1.0;
    // The normal offset: texels of the face at the receiver's distance (towards the light's side of the surface; the
    // texel factor is the same for the six faces of a point light)
    float dist = length(relPos - posRange.xyz);
    vec3 p     = relPos + n * (atlasTiles[uint(intensity.w)].params.x * dist * lighting.atlasParams.z);
    // The face of the offset point (round 1, finding 12: the face of the receiver itself made a point near a cube edge
    // project outside its face, onto the clamped tile border)
    vec3 d    = p - posRange.xyz;
    uint face = 0u;
    if (dirKind.w < 0.5)
    {
        vec3 a = abs(d);
        face   = a.x >= a.y && a.x >= a.z ? (d.x > 0.0 ? 0u : 1u) : a.y >= a.z ? (d.y > 0.0 ? 2u : 3u) : (d.z > 0.0 ? 4u : 5u);
    }
    AtlasTile t = atlasTiles[uint(intensity.w) + face];
    vec4 clip   = t.viewProj * vec4(p, 1.0);
    if (clip.w <= 0.0) return 1.0;
    vec2 uv     = clip.xy / clip.w * 0.5 + 0.5;
    float h     = 0.5 * t.params.y;
    vec2 auv    = t.rect.xy + clamp(uv, vec2(h), vec2(1.0 - h)) * t.rect.zw;
    float ref   = length(p - posRange.xyz) / posRange.w - lighting.atlasParams.y / 65535.0;
    float s     = atlasCompare(auv, 0.0, ref);
    if (t.params.z > 0.5) s = min(s, atlasCompare(auv, 1.0, ref));
    return s;
}

#define LOCAL_LIGHT_SHADOW(L, relPos, n, dist) atlasShadow(L.posRange, L.dirKind, L.intensity, L.cone, relPos, n)

#endif
