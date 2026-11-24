/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The shadow factors of the shadow light (S36): the cascades (a2, binding 8; A5), the cloud shadow (A4: the 2D cloud
// layer, cloudShadowAt of atmosphere_common.glsl) and (a2) the horizon map (binding 9; A6). Requires scene_block.glsl,
// the cloud coverage (binding 3) and atmosphere_common.glsl.
// - SHADOWS_VERTEX (defined before the include): usable in a vertex shader (explicit gradients, no implicit LOD);
// - SHADOW_INPUT: the terms come from the shader's inputs (shadowInput, set by main): the grass evaluates them per
//   vertex, so its fragment shader (the overdraw of the blades) carries none of the cascade and horizon code;
// - LIGHTING_NO_SHADOWS: the cloud shadow only (the terrain's variant while the cascades and the horizon map are off:
//   no cost of their code when they are not used).
#ifndef SHADOWS_GLSL
#define SHADOWS_GLSL

#ifdef SHADOWS_VERTEX
#    define SHADOW_COMPARE(s, c) textureGrad(s, c, vec2(0.0), vec2(0.0))
#else
#    define SHADOW_COMPARE(s, c) texture(s, c)
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

// The shadow terms of the shadow light at a receiver: the cascades' factor and index, the horizon map's factor and its
// shadow heights above the receiver (R, G), the cloud shadow
struct ShadowTerms
{
    float cascade;
    int index;
    float horizon;
    vec2 rg;
    float cloud;
};

ShadowTerms noShadow()
{
    ShadowTerms t;
    t.cascade = 1.0;
    t.index   = -1;
    t.horizon = 1.0;
    t.rg      = vec2(-1.0e9);
    t.cloud   = 1.0;
    return t;
}

#ifdef SHADOW_INPUT
// Set by the shader's main from its inputs
ShadowTerms shadowInput = ShadowTerms(1.0, -1, 1.0, vec2(-1.0e9), 1.0);

ShadowTerms shadowTerms(vec3 relPos, vec3 n)
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

#ifdef LIGHTING_NO_SHADOWS
ShadowTerms shadowTerms(vec3 relPos, vec3 n)
{
    ShadowTerms t = noShadow();
    t.cloud       = cloudShadow(relPos);
    return t;
}
#else
// The cascades: one D32 layer per cascade, compared with ref <= stored (LESS_OR_EQUAL, not reversed)
layout(set = 1, binding = 8) uniform sampler2DArrayShadow cascadeMap;
// The horizon maps (RG32F, three layers: the block names the previous and the front map)
layout(set = 1, binding = 9) uniform sampler2DArray horizonMaps;

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
            for (int x = -1; x <= 1; ++x)
                sum += SHADOW_COMPARE(cascadeMap, vec4(uv + vec2(x, y) * texel, float(c), ref));
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
// the band at its border; cascade: the index, -1 outside every rectangle (factor 1). In the band of a cascade without a
// next one (the last, or the next not valid) the cascade's own factor is returned and toLit is the band's weight
// towards outside (shadowTerms blends the cascade with the horizon map's G into the outside's R: the terrain between
// 64 m and cascadeCasterDistance has no gap at the border; S36a2 review, major 5); else toLit is 0. One cascadeSample
// call site in a loop (keeps the shaders small)
float cascadeShadow(vec3 relPos, vec3 n, out int cascade, out float toLit)
{
    cascade   = -1;
    toLit     = 0.0;
    int count = int(lighting.shadowParams.x);
    if (count == 0) return 1.0;
    float t = 0.0;
    for (int c = 0; c < count; ++c)
    {
        if (lighting.cascadeParams[c].z < 0.5) continue;
        vec2 ndc   = (lighting.cascadeMatrix[c] * vec4(relPos, 1.0)).xy;
        float edge = max(abs(ndc.x), abs(ndc.y));
        if (edge > 1.0) continue;
        cascade = c;
        // The band ends at the edge margin (the PCF footprint and the normal offset stay inside the layer where the
        // cascade has weight; S36a2 review, finding 12)
        float band   = lighting.cascadeParams[c].y;
        float margin = lighting.cascadeParams[c].w;
        t            = clamp((edge - (1.0 - band)) / max(band - margin, 1e-4), 0.0, 1.0);
        break;
    }
    if (cascade < 0) return 1.0;
    // The next cascade of the band, else towards outside
    int next = -1;
    if (t > 0.0 && cascade + 1 < count && lighting.cascadeParams[cascade + 1].z > 0.5)
    {
        vec2 ndc = (lighting.cascadeMatrix[cascade + 1] * vec4(relPos, 1.0)).xy;
        if (max(abs(ndc.x), abs(ndc.y)) <= 1.0) next = cascade + 1;
    }
    if (next < 0)
    {
        toLit = t;
        if (t >= 1.0) return 1.0;
    }
    vec3 L       = normalize(lighting.shadowLight.xyz);
    float result = 0.0;
    int samples  = next < 0 ? 1 : 2;
    for (int k = 0; k < samples; ++k)
    {
        int c   = k == 0 ? cascade : next;
        float w = next < 0 ? 1.0 : k == 0 ? 1.0 - t : t;
        if (w > 0.0) result += w * cascadeSample(c, cascadeBiased(c, relPos, n, L));
    }
    return result;
}

// The shadow heights (R, G) of a horizon map at the receiver, relative to the receiver's height (> 0: the receiver is
// below them); inside: the map covers the receiver (else -1e9). A texel without an occluder holds -1e9 too (lit)
vec2 horizonHeights(vec4 map, float refRel, vec3 relPos, out bool inside)
{
    const vec2 none = vec2(-1.0e9);
    inside          = false;
    if (map.w < 0.0) return none;
    float res = lighting.horizonRef.w;
    vec2 st   = (relPos.xz - map.xy) * map.z;  // texels
    if (any(lessThan(st, vec2(0.5))) || any(greaterThan(st, vec2(res - 0.5)))) return none;
    inside = true;
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
        vec2 a  = texelFetch(horizonMaps, ivec3(i, l), 0).rg;
        vec2 b  = texelFetch(horizonMaps, ivec3(i + ivec2(1, 0), l), 0).rg;
        vec2 c  = texelFetch(horizonMaps, ivec3(i + ivec2(0, 1), l), 0).rg;
        vec2 d  = texelFetch(horizonMaps, ivec3(i + ivec2(1, 1), l), 0).rg;
        h       = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    }
    return h + refRel - relPos.y;
}

// The horizon map's shadow heights (R, G) above the receiver (S36a2, A6): the previous and the front map blended by
// the front's progress; a map that does not cover the receiver gives way to the other (after an anchor move the
// receivers outside the previous map are shadowed by the front map at once; S36a2 review, major 4); -1e9 outside both
vec2 horizonRg(vec3 relPos)
{
    if (lighting.horizonParams.x < 0.5) return vec2(-1.0e9);
    bool inPrev, inFront;
    vec2 a  = horizonHeights(lighting.horizonPrev, lighting.horizonRef.x, relPos, inPrev);
    vec2 b  = horizonHeights(lighting.horizonFront, lighting.horizonRef.y, relPos, inFront);
    float w = !inPrev ? 1.0 : !inFront ? 0.0 : lighting.horizonRef.z;
    return mix(a, b, w);
}

// The horizon map's factor of a shadow height H above the receiver: lit by smoothstep(H - s, H, y), s = max(4 m,
// 2 texel tan(el))
float horizonFactor(float H)
{
    float s = max(4.0, 2.0 * lighting.horizonParams.y / max(lighting.horizonFront.z, 1e-6));
    return 1.0 - smoothstep(0.0, s, H);
}

// The shadow terms (A5, A6): inside a cascade rectangle with terrain casters the horizon map's G (the cascades shadow
// the nearer terrain), else R. In the band of the last cascade the inside's product (cascade x G) is blended with the
// outside's R by the band's weight; the cascade term is the factor blended towards 1 (as shown by the debug views) and
// the horizon term the rest of the product
ShadowTerms shadowTerms(vec3 relPos, vec3 n)
{
    ShadowTerms t;
    float toLit;
    float c        = cascadeShadow(relPos, n, t.index, toLit);
    t.rg           = horizonRg(relPos);
    float outside  = horizonFactor(t.rg.x);
    float inside   = t.index >= 0 && lighting.horizonParams.z > 0.5 ? horizonFactor(t.rg.y) : outside;
    float combined = mix(c * inside, outside, toLit);
    t.cascade      = mix(c, 1.0, toLit);
    t.horizon      = t.cascade > 1e-6 ? combined / t.cascade : inside;
    t.cloud        = cloudShadow(relPos);
    return t;
}
#endif
#endif

// The visibility of the shadow light at the receiver (relPos, normal n): cascades x horizon x clouds
float shadowFactor(vec3 relPos, vec3 n)
{
    ShadowTerms t = shadowTerms(relPos, n);
    return t.cascade * t.horizon * t.cloud;
}

#endif
