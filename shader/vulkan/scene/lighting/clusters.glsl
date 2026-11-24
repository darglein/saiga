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

// LocalLights::GpuLight (std430, 80 bytes): posRange (xyz camera-relative, w range), dirKind (xyz the spot's axis, w 0
// point, 1 spot), intensity (rgb colour * candela * flicker, w the first atlas tile-table entry, -1: unshadowed),
// cone (x cosOuter, y 1 / (cosInner - cosOuter), z 1 the atlas tiles are rendered, w unused), clusterRange (the
// clusters it can touch, light_cluster.comp's pre-test: x x0 + 256 x1, y y0 + 256 y1, z z0 + 256 z1)
struct LocalLight
{
    vec4 posRange;
    vec4 dirKind;
    vec4 intensity;
    vec4 cone;
    vec4 clusterRange;
};

layout(std430, set = 1, binding = 10) readonly buffer LocalLightBuffer
{
    // x: the light count (lighting.lightCounts.x as well)
    uvec4 lightHeader;
    LocalLight localLights[];
};

// The cluster lists (LightClusters): per cluster (count, total) and the fixed slots of clusterGrid.w indices
layout(std430, set = 1, binding = 11) readonly buffer ClusterTable
{
    uvec2 clusterTable[];
};
layout(std430, set = 1, binding = 12) readonly buffer ClusterIndices
{
    uint clusterIndices[];
};

// The cluster of the fragment from gl_FragCoord.xy and the view depth (-1: beyond clusterFar)
int fragmentCluster(vec3 relPos)
{
    float z = dot(relPos, lighting.viewForward.xyz);
    if (z > lighting.clusterDepth.z) return -1;
    uvec3 g = lighting.clusterGrid.xyz;
    uint k  = 0u;
    if (z >= lighting.clusterDepth.y)
        k = min(1u + uint(max(log(z / lighting.clusterDepth.y) * lighting.clusterDepth.w, 0.0)), g.z - 1u);
    vec2 px = gl_FragCoord.xy / lighting.viewport.xy;
    uvec2 t = min(uvec2(max(px, vec2(0.0)) * vec2(g.xy)), g.xy - 1u);
    return int((k * g.y + t.y) * g.x + t.x);
}

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

// The direct light of the frame's local lights at a surface: every light (brute force, LightingParameters::
// clusteredLights false: the reference) or the lights of the fragment's cluster, each with its atlas shadow (one loop,
// one inlined BRDF: the scene shaders stay small). minShadow / shadowed: the darkest atlas factor of the shadowed
// lights that reach the surface (only the highlighted light while one is highlighted) and their count (the AtlasShadow
// view, probe_lighting)
vec3 localLightsDirect(vec3 n, vec3 v, vec3 relPos, vec3 albedo, float rough, float metalness, vec2 lut, float wrap,
                       inout float minShadow, inout int shadowed)
{
    vec3 sum        = vec3(0.0);
    bool clustered  = lighting.lightCounts.y != 0u;
    uint count      = lighting.lightCounts.x;
    uint base       = 0u;
    if (clustered)
    {
        int c = fragmentCluster(relPos);
        if (c < 0) return sum;
        count = clusterTable[c].x;
        base  = uint(c) * lighting.clusterGrid.w;
    }
    for (uint k = 0u; k < count; ++k)
    {
        uint i       = clustered ? clusterIndices[base + k] : k;
        LocalLight L = localLights[i];
        vec3 l;
        float dist;
        vec3 E = localLightIlluminance(L, relPos, l, dist);
        if (max(E.r, max(E.g, E.b)) <= 0.0 || dot(n, l) <= -wrap) continue;
        float shadow = 1.0;
        if (L.intensity.w >= 0.0)
        {
            shadow = LOCAL_LIGHT_SHADOW(L, relPos, n, dist);
            if (lighting.lightCounts.z == 0u || lighting.lightCounts.z == i + 1u)
            {
                minShadow = min(minShadow, shadow);
                shadowed++;
            }
        }
        sum += pbrDirect(n, v, l, albedo, rough, metalness, E * shadow, lut, wrap);
    }
    return sum;
}

#endif
