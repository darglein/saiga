/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The low-frequency light terms of the physical path (S36a1c review): the direct light of the sun and the moon at a
// receiver (the illuminance times the transmittance through the atmosphere, 0 below the horizon; no shadow) and the
// active fog (rgb: its colour times the factor, a: 1 - the factor). The transmittance depends on the receiver's height
// only and the fog is smooth along a surface, so the grass (per tuft) and the terrain compute them in their vertex
// shader (VERTEX_LIGHT_INPUT in the fragment shader, lighting.glsl); the other passes per fragment through the same
// function. Clay mode: zero light and no fog (its studio fog is per fragment, lighting.glsl). Requires
// scene_block.glsl, the transmittance (binding 6) and the sky-view LUT (binding 7) as transmittanceLut and skyViewLut,
// atmosphere_common.glsl. Vertex or fragment stage (explicit LODs only).
#ifndef VERTEX_LIGHT_GLSL
#define VERTEX_LIGHT_GLSL

#include "../../world/prototype_fog.glsl"

struct VertexLight
{
    vec3 sunE;
    vec3 moonE;
    vec4 fog;
};

// The physical path's fog at relPos: (colour * f, 1 - f). Its colour is the sky at the horizon in relPos's azimuth
// (the sky-view LUT's row of the elevation 0: fog and sky match at the horizon); near surfaces skip the lookup
vec4 physicalFog(vec3 relPos)
{
    if (lighting.fogParams.z < 0.5) return vec4(0.0, 0.0, 0.0, 1.0);
    float f = protoFogFactor(relPos, lighting.fogParams.w, vec4(lighting.fogParams.xyz, 0.0));
    if (f < 1.0 / 2048.0) return vec4(0.0, 0.0, 0.0, 1.0);
    vec2 h        = relPos.xz;
    float u       = dot(h, h) > 1e-8 ? atan(h.y, h.x) / (2.0 * kAtmPi) + 0.5 : 0.5;
    vec3 fogColor = textureLod(skyViewLut, vec2(u, skyViewFogV(lighting.atmosphere.w)), 0.0).rgb / kSkyLutScale;
    return vec4(fogColor * f, 1.0 - f);
}

VertexLight computeVertexLight(vec3 relPos)
{
    VertexLight v;
    v.sunE  = vec3(0.0);
    v.moonE = vec3(0.0);
    v.fog   = vec4(0.0, 0.0, 0.0, 1.0);
    if (lighting.mode.x != 1u) return v;
    // An open space: the bodies' light through the atmosphere at the receiver's height; a body below the receiver's
    // horizon gives no direct light (the transmittance is 0 there; no lookup; CodeRabbit #84: a fixed -0.1 dropped the
    // light of a body above the horizon of a receiver higher than 32 km). The CPU zeroes the moon's w in daylight (no
    // light worth a lookup)
    if (lighting.mode.y == 0u)
    {
        float r       = atmosphereRadius(lighting.atmosphere.x + relPos.y);
        float horizon = limbCosine(r) - 0.004;
        if (lighting.sun.w > 0.0 && lighting.sun.y > horizon)
            v.sunE = lighting.sun.w * bodyTransmittanceAtRadius(transmittanceLut, r, lighting.sun.xyz);
        if (lighting.moon.w > 0.0 && lighting.moon.y > horizon)
            v.moonE = lighting.moon.w * bodyTransmittanceAtRadius(transmittanceLut, r, lighting.moon.xyz);
    }
    v.fog = physicalFog(relPos);
    return v;
}

#endif
