/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The shadow factors of the shadow light (S36): the cascades (a2, binding 8; A5), the cloud shadow (A4: the 2D cloud
// layer, cloudShadowAt of atmosphere_common.glsl) and (a2) the horizon map (binding 9). Requires scene_block.glsl, the
// cloud coverage (binding 3) and atmosphere_common.glsl.
#ifndef SHADOWS_GLSL
#define SHADOWS_GLSL

// The cascades: one D32 layer per cascade, compared with ref <= stored (LESS_OR_EQUAL, not reversed)
layout(set = 1, binding = 8) uniform sampler2DArrayShadow cascadeMap;
// The horizon maps (RG32F, three layers: the block names the previous and the front map)
layout(set = 1, binding = 9) uniform sampler2DArray horizonMaps;

#ifdef SHADOW_INPUT
// SHADOW_INPUT (defined before the include): the cloud shadow comes from the shader's input (shadowInput, set by
// main): the grass evaluates it per tuft in its vertex shader (the cloud shadow's texels are 64 m)
float shadowInput = 1.0;

float cloudShadow(vec3 relPos)
{
    return shadowInput;
}
#else
// The cloud layer's shadow at the receiver relPos (camera-relative): S42's hook (its transmittance map)
float cloudShadow(vec3 relPos)
{
    if (lighting.cloudMotion.w < 0.5 || lighting.shadowLight.w == 0.0) return 1.0;
    return cloudShadowAt(cloudCoverage, lighting.cloudMotion.xy, relPos.xz, lighting.atmosphere.x + relPos.y,
                         normalize(lighting.shadowLight.xyz), lighting.clouds, lighting.cloudMotion.z);
}
#endif

// The display colours of the cascades (Saiga::Vulkan::cascadeColor; gray outside every cascade)
vec3 cascadeTint(int c)
{
    if (c == 0) return vec3(1.0, 0.25, 0.2);
    if (c == 1) return vec3(0.3, 1.0, 0.3);
    if (c == 2) return vec3(0.3, 0.5, 1.0);
    if (c == 3) return vec3(1.0, 0.9, 0.2);
    return vec3(0.6);
}

// The fixed PCF of one cascade at the camera-relative point p (already biased): 3 x 3 bilinear compares (4 x 4
// texels), equal weights (deterministic goldens); without linear filtering of the format each bilinear compare is a
// gather of 4 nearest compares weighted here
float cascadeSample(int c, vec3 p)
{
    vec3 h     = (lighting.cascadeMatrix[c] * vec4(p, 1.0)).xyz;
    vec2 uv    = h.xy * 0.5 + 0.5;
    float ref  = clamp(h.z, 0.0, 1.0);
    vec2 size  = vec2(textureSize(cascadeMap, 0).xy);
    vec2 texel = 1.0 / size;
    float sum  = 0.0;
    if (lighting.shadowParams.w > 0.5)
    {
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x) sum += texture(cascadeMap, vec4(uv + vec2(x, y) * texel, float(c), ref));
        return sum / 9.0;
    }
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
        {
            vec2 st = (uv + vec2(x, y) * texel) * size - 0.5;
            vec2 f  = fract(st);
            // (u0, v1), (u1, v1), (u1, v0), (u0, v0) around the shared corner
            vec4 g = textureGather(cascadeMap, vec3((floor(st) + 1.0) * texel, float(c)), ref);
            sum += mix(mix(g.w, g.z, f.x), mix(g.x, g.y, f.x), f.y);
        }
    return sum / 9.0;
}

// The receiver biased in world metres of cascade c: depthBiasTexels texels towards the light, normalOffsetTexels texels
// along the normal
vec3 cascadeBiased(int c, vec3 relPos, vec3 n, vec3 L)
{
    float texel = lighting.cascadeParams[c].x;
    return relPos + n * (lighting.shadowParams.z * texel) + L * (lighting.shadowParams.y * texel);
}

// The cascades' factor at the receiver (relPos, normal n): the first cascade whose rectangle contains the receiver's
// light-space xy (selection by the xy rectangle; the depth range is shared, D23), blended into the next cascade over
// the band at its border (beyond the last into lit: the horizon map and the clouds take over); cascade: the index, -1
// outside every rectangle (factor 1)
float cascadeShadow(vec3 relPos, vec3 n, out int cascade)
{
    cascade   = -1;
    int count = int(lighting.shadowParams.x);
    if (count == 0) return 1.0;
    vec3 L = normalize(lighting.shadowLight.xyz);
    for (int c = 0; c < count; ++c)
    {
        if (lighting.cascadeParams[c].z < 0.5) continue;
        vec2 ndc   = (lighting.cascadeMatrix[c] * vec4(relPos, 1.0)).xy;
        float edge = max(abs(ndc.x), abs(ndc.y));
        if (edge > 1.0) continue;
        cascade    = c;
        float f    = cascadeSample(c, cascadeBiased(c, relPos, n, L));
        float band = lighting.cascadeParams[c].y;
        float t    = (edge - (1.0 - band)) / max(band, 1e-4);
        if (t > 0.0)
        {
            float g = 1.0;
            if (c + 1 < count && lighting.cascadeParams[c + 1].z > 0.5)
            {
                vec2 next = (lighting.cascadeMatrix[c + 1] * vec4(relPos, 1.0)).xy;
                if (max(abs(next.x), abs(next.y)) <= 1.0) g = cascadeSample(c + 1, cascadeBiased(c + 1, relPos, n, L));
            }
            f = mix(f, g, clamp(t, 0.0, 1.0));
        }
        return f;
    }
    return 1.0;
}

// The shadow heights (R, G) of a horizon map at the receiver, relative to the receiver's height (> 0: the receiver is
// below them); -1e9 outside the map or without it
vec2 horizonHeights(vec4 map, float refRel, vec3 relPos)
{
    const vec2 none = vec2(-1.0e9);
    if (map.w < 0.0) return none;
    float res = lighting.horizonRef.w;
    vec2 st   = (relPos.xz - map.xy) * map.z;  // texels
    if (any(lessThan(st, vec2(0.5))) || any(greaterThan(st, vec2(res - 0.5)))) return none;
    vec2 h;
    if (lighting.horizonParams.w > 0.5)
        h = textureLod(horizonMaps, vec3(st / res, map.w), 0.0).rg;
    else
    {
        // Nearest fetches with bilinear weights (RG32F without linear filtering)
        vec2 p  = st - 0.5;
        ivec2 i = ivec2(floor(p));
        vec2 f  = p - vec2(i);
        int l   = int(map.w);
        vec2 a  = texelFetch(horizonMaps, ivec3(i, l), 0).rg, b = texelFetch(horizonMaps, ivec3(i + ivec2(1, 0), l), 0).rg;
        vec2 c  = texelFetch(horizonMaps, ivec3(i + ivec2(0, 1), l), 0).rg;
        vec2 d  = texelFetch(horizonMaps, ivec3(i + ivec2(1, 1), l), 0).rg;
        h       = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    }
    return h + refRel - relPos.y;
}

// The horizon map's factor at the receiver (S36a2, A6): lit by smoothstep(H - s, H, y) with H the shadow height (G
// inside a cascade with terrain casters: the cascades shadow the nearer terrain; else R), s = max(4 m, 2 texel
// tan(el)); the previous and the front map blended; rg: (R, G) above the receiver
float horizonShadow(vec3 relPos, bool useG, out vec2 rg)
{
    rg = vec2(-1.0e9);
    if (lighting.horizonParams.x < 0.5) return 1.0;
    vec2 a = horizonHeights(lighting.horizonPrev, lighting.horizonRef.x, relPos);
    vec2 b = horizonHeights(lighting.horizonFront, lighting.horizonRef.y, relPos);
    rg     = mix(a, b, lighting.horizonRef.z);
    float H = useG ? rg.y : rg.x;
    float s = max(4.0, 2.0 * lighting.horizonParams.y / max(lighting.horizonFront.z, 1e-6));
    return 1.0 - smoothstep(0.0, s, H);
}

// The visibility of the shadow light at the receiver (relPos, normal n): cascades x horizon x clouds
float shadowFactor(vec3 relPos, vec3 n)
{
    int cascade;
    float c = cascadeShadow(relPos, n, cascade);
    vec2 rg;
    float h = horizonShadow(relPos, cascade >= 0 && lighting.horizonParams.z > 0.5, rg);
    return c * h * cloudShadow(relPos);
}

#endif
