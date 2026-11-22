/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The local lights of S36b (B2, B3 of docs/stages/S36_lighting.plan.md; docs/engine_design.md §10.7): the frame's
// lights (binding 10 of set 1, LocalLights: camera-relative, sorted by importance) and their direct light at a
// surface. Falloff: inverse square with Karis's window saturate(1 - (d / range)^4)^2; spot: a smooth cone between the
// cosines. The same BRDF as the sun (pbrDirect). Requires scene_block.glsl and pbr.glsl; fragment shaders only (the
// clustered path reads gl_FragCoord).
#ifndef CLUSTERS_GLSL
#define CLUSTERS_GLSL

// LocalLights::GpuLight (std430, 64 bytes): posRange (xyz camera-relative, w range), dirKind (xyz the spot's axis, w 0
// point, 1 spot), intensity (rgb colour * candela * flicker, w the first atlas tile-table entry, -1: unshadowed),
// cone (x cosOuter, y 1 / (cosInner - cosOuter), z 1 the atlas tiles are rendered, w unused)
struct LocalLight
{
    vec4 posRange;
    vec4 dirKind;
    vec4 intensity;
    vec4 cone;
};

layout(std430, set = 1, binding = 10) readonly buffer LocalLightBuffer
{
    // x: the light count (lighting.lightCounts.x as well)
    uvec4 lightHeader;
    LocalLight localLights[];
};

// The illuminance (lux) of a light at relPos and the direction towards it (l)
vec3 localLightIlluminance(LocalLight L, vec3 relPos, out vec3 l, out float dist)
{
    vec3 d   = L.posRange.xyz - relPos;
    float d2 = max(dot(d, d), 1e-4);
    dist     = sqrt(d2);
    l        = d / dist;
    float r  = dist / L.posRange.w;
    float r2 = r * r;
    float w  = clamp(1.0 - r2 * r2, 0.0, 1.0);
    float a  = w * w / d2;
    if (L.dirKind.w > 0.5)
    {
        float t = clamp((dot(-l, L.dirKind.xyz) - L.cone.x) * L.cone.y, 0.0, 1.0);
        a *= t * t * (3.0 - 2.0 * t);
    }
    return L.intensity.rgb * a;
}

// The shadow of a light (atlas_shadow.glsl from S36b step 4; 1 without it)
#ifndef LOCAL_LIGHT_SHADOW
#    define LOCAL_LIGHT_SHADOW(L, relPos, n, dist) 1.0
#endif

// The reflected radiance of the light with buffer index i
vec3 localLightRadiance(uint i, vec3 n, vec3 v, vec3 relPos, vec3 albedo, float rough, float metalness, vec2 lut,
                        float wrap)
{
    LocalLight L = localLights[i];
    vec3 l;
    float dist;
    vec3 E = localLightIlluminance(L, relPos, l, dist);
    if (max(E.r, max(E.g, E.b)) <= 0.0 || dot(n, l) <= -wrap) return vec3(0.0);
    float shadow = L.intensity.w >= 0.0 ? LOCAL_LIGHT_SHADOW(L, relPos, n, dist) : 1.0;
    return pbrDirect(n, v, l, albedo, rough, metalness, E * shadow, lut, wrap);
}

// The direct light of the frame's local lights at a surface: every light (brute force, LightingParameters::
// clusteredLights false: the reference) or the lights of the fragment's cluster
vec3 localLightsDirect(vec3 n, vec3 v, vec3 relPos, vec3 albedo, float rough, float metalness, vec2 lut, float wrap)
{
    vec3 sum = vec3(0.0);
    uint count = lighting.lightCounts.x;
    for (uint i = 0u; i < count; ++i) sum += localLightRadiance(i, n, v, relPos, albedo, rough, metalness, lut, wrap);
    return sum;
}

#endif
