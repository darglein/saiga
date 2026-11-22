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

// The visibility of the shadow light at the receiver (relPos, normal n): cascades x horizon x clouds
float shadowFactor(vec3 relPos, vec3 n)
{
    int cascade;
    return cascadeShadow(relPos, n, cascade) * cloudShadow(relPos);
}

#endif
