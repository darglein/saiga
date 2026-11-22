/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The lighting of S36 (A3 of docs/stages/S36_lighting.plan.md, docs/engine_design.md §10.7): set 1 of the scene draws
// (SceneLightingSet: 0 the block, 1 the foliage texture, 2 the ambient SH, 3 the cloud coverage, 4 the prefiltered
// environment cube, 5 the BRDF LUT, 6 the transmittance LUT, 7 the sky-view LUT; a2: 8 the cascades, 9 the horizon map;
// b: 10-14 the lights, clusters and atlas) and shadeSurface, which every scene pass calls, so every surface lights the
// same way:
// - clay mode (lighting.mode.x == 0: ToneMap::Identity, or no environment): the studio environment, S19a's model
//   exactly: albedo (sun max(n.l, 0) + ambient) ao, the studio fog; unit-less;
// - physical (an environment and a tone map): GGX (pbr.glsl) for the sun and the moon in lux with the shadow factor of
//   the shadow light, the ambient (ambient.glsl), the active fog; radiance in cd/m^2 before the pre-exposure.
// Fragment shaders only. The caller applies the output (pre-exposure or the display encoding), except in a debug
// frame (debugViewActive: the value of the view is written as it is).
//
// The debug views (A8; lighting.mode.z = LightingDebugView, uniform branches): data views return raw values (albedo,
// n * 0.5 + 0.5, roughness, metalness, (roughness, metalness, ao), the cloud and shadow factors), LightingOnly (white
// albedo, roughness 0.5) and the radiance views (DirectOnly: the sun and the moon; AmbientOnly: the ambient's diffuse
// part; SpecularOnly: the environment's specular reflection) display colours (debugDisplay: the frame's exposure and
// ACES in the physical path; clamped in clay), or raw radiance in the probe frames (debugParams.z; times the probe's
// pre-exposure debugParams.w, fp16: a glossy highlight in sunlight exceeds 65504 cd/m^2), Exposure a heat map of the
// luminance in EV100 (turbo over debugParams.y EV around debugParams.x). The radiance views include the surface's tint
// (as the shaded colour).
#ifndef LIGHTING_GLSL
#define LIGHTING_GLSL

#include "scene_block.glsl"

#include "../../world/prototype_shading.glsl"
#include "../color.glsl"
#include "debug_common.glsl"

layout(set = 1, binding = 3) uniform sampler2D cloudCoverage;
layout(set = 1, binding = 4) uniform samplerCube environmentCube;
layout(set = 1, binding = 5) uniform sampler2D brdfLut;
layout(set = 1, binding = 6) uniform sampler2D transmittanceLut;
layout(set = 1, binding = 7) uniform sampler2D skyViewLut;

#include "atmosphere_common.glsl"
#include "pbr.glsl"
#include "ambient.glsl"
#include "shadows.glsl"

// SurfaceInput::flags: foliage (wrapped diffuse)
const uint surfaceFoliage = 1u;

struct SurfaceInput
{
    vec3 albedo;  // linear
    float roughness;
    float metalness;
    vec3 n;       // normalized, facing the viewer
    vec3 relPos;  // camera-relative position
    vec3 emissive;
    float ao;
    uint flags;
    vec3 tint;  // multiplies the reflected light (the LOD tint, the clay water)
};

SurfaceInput surfaceInput(vec3 albedo, vec3 n, vec3 relPos)
{
    SurfaceInput s;
    s.albedo    = albedo;
    s.roughness = 0.8;
    s.metalness = 0.0;
    s.n         = n;
    s.relPos    = relPos;
    s.emissive  = vec3(0.0);
    s.ao        = 1.0;
    s.flags     = 0u;
    s.tint      = vec3(1.0);
    return s;
}

// The physical path (an environment and a tone map); else clay / studio
bool physicalLighting()
{
    return lighting.mode.x == 1u;
}

// The albedo of a prototype surface: its colour of the block (D21) in the physical path, the clay albedo otherwise
vec3 prototypeAlbedo(vec3 physical)
{
    return physicalLighting() ? physical : vec3(lighting.studioShading.z);
}

// The active fog (studio or physical, chosen on the CPU): S19a's height fog with the camera height above its base; in
// the physical path its colour is the sky at the horizon in the view's azimuth (the sky-view LUT: fog and sky match at
// the horizon)
vec3 lightingFog(vec3 c, vec3 relPos)
{
    if (!physicalLighting() || lighting.fogParams.z < 0.5)
        return protoApplyFog(c, relPos, lighting.fogParams.w, vec4(lighting.fogParams.xyz, 0.0),
                             lighting.fogColorLin.rgb);
    // The fog factor first (the height fog of a black surface in a white fog): near surfaces skip the colour's lookup
    float f = protoApplyFog(vec3(0.0), relPos, lighting.fogParams.w, vec4(lighting.fogParams.xyz, 0.0), vec3(1.0)).r;
    if (f < 1.0 / 2048.0) return c;
    // The LUT's row of the elevation 0 (skyViewFogV of the block's horizon), the azimuth of relPos
    vec2 h        = relPos.xz;
    float u       = dot(h, h) > 1e-8 ? atan(h.y, h.x) / (2.0 * kAtmPi) + 0.5 : 0.5;
    vec3 fogColor = textureLod(skyViewLut, vec2(u, skyViewFogV(lighting.atmosphere.w)), 0.0).rgb / kSkyLutScale;
    return mix(c, fogColor, f);
}

// The direct light's transmittance through the atmosphere at the receiver (relPos) towards the body l: the
// transmittance LUT at the receiver's height, 0 behind the planet
vec3 bodyTransmittance(vec3 l, vec3 relPos)
{
    return bodyTransmittanceAtRadius(transmittanceLut, atmosphereRadius(lighting.atmosphere.x + relPos.y), l);
}

// The debug views (LightingDebugView of LightingTypes.h)
const uint viewNone = 0u, viewLightingOnly = 1u, viewAlbedo = 2u, viewNormals = 3u, viewRoughness = 4u,
           viewMetalness = 5u, viewMaterial = 6u, viewDirectOnly = 7u, viewAmbientOnly = 8u, viewSpecularOnly = 9u,
           viewCloudShadow = 10u, viewShadowFactor = 11u, viewExposure = 12u;

bool debugViewActive()
{
    return lighting.mode.z != viewNone;
}

// A radiance as a display colour of a debug view (raw in the probe frames)
vec3 debugDisplay(vec3 radiance)
{
    if (lighting.debugParams.z > 0.5) return radiance * lighting.debugParams.w;
    if (physicalLighting()) return linearToSrgb(acesNarkowicz(max(radiance, vec3(0.0)) * lighting.exposure.x));
    return linearToSrgb(clamp(radiance, vec3(0.0), vec3(1.0)));
}

// The light of a surface (radiance before the pre-exposure) with the fog; in a debug frame the view's value
vec3 shadeSurface(SurfaceInput s)
{
    const uint view = lighting.mode.z;
    if (view == viewAlbedo) return s.albedo;
    if (view == viewNormals) return s.n * 0.5 + 0.5;
    if (view == viewRoughness) return vec3(s.roughness);
    if (view == viewMetalness) return vec3(s.metalness);
    if (view == viewMaterial) return vec3(s.roughness, s.metalness, s.ao);
    if (view == viewCloudShadow) return vec3(cloudShadow(s.relPos));
    if (view == viewShadowFactor) return vec3(shadowFactor(s.relPos, s.n));
    if (view == viewLightingOnly)
    {
        s.albedo    = vec3(1.0);
        s.roughness = 0.5;
        s.metalness = 0.0;
        s.emissive  = vec3(0.0);
        s.tint      = vec3(1.0);
    }
    vec3 c, direct = vec3(0.0), ambientDiff = vec3(0.0), ambientSpec = vec3(0.0);
    if (!physicalLighting())
    {
        // S19a: albedo (sun + ambient) ao, the material's emissive on top
        vec3 L      = normalize(lighting.studioSun.xyz);
        float d     = max(dot(s.n, L), 0.0) * lighting.studioSun.w;
        c           = s.albedo * (d + studioAmbient(s.n)) * s.ao;
        c           = c * s.tint + s.emissive;
        direct      = s.albedo * d * s.ao;
        ambientDiff = s.albedo * studioAmbient(s.n) * s.ao;
    }
    else
    {
        vec3 v      = normalize(-s.relPos);
        float rough = pbrRoughness(s.roughness);
        float NoV   = max(dot(s.n, v), 1e-4);
        vec2 lut    = texture(brdfLut, vec2(NoV, rough)).rg;
        float wrap  = (s.flags & surfaceFoliage) != 0u ? 0.3 : 0.0;
        if (lighting.mode.y == 0u)
        {
            float shadow = shadowFactor(s.relPos, s.n);
            // A body below the horizon gives no direct light (the transmittance is 0 there; no lookup)
            if (lighting.sun.w > 0.0 && lighting.sun.y > -0.1)
            {
                vec3 sunE = lighting.sun.w * bodyTransmittance(lighting.sun.xyz, s.relPos);
                if (lighting.shadowLight.w == 1.0) sunE *= shadow;
                direct += pbrDirect(s.n, v, lighting.sun.xyz, s.albedo, rough, s.metalness, sunE, lut, wrap);
            }
            if (lighting.moon.w > 0.0 && lighting.moon.y > -0.1)
            {
                vec3 moonE = lighting.moon.w * bodyTransmittance(lighting.moon.xyz, s.relPos);
                if (lighting.shadowLight.w == 2.0) moonE *= shadow;
                direct += pbrDirect(s.n, v, lighting.moon.xyz, s.albedo, rough, s.metalness, moonE, lut, wrap);
            }
        }
        vec3 r = reflect(-v, s.n);
        pbrAmbientParts(s.albedo, rough, s.metalness, NoV, lut, ambientSpecular(r, rough, s.relPos),
                        ambientDiffuse(s.n, s.relPos), ambientSpec, ambientDiff);
        ambientSpec *= s.ao;
        ambientDiff *= s.ao;
        // The emissive colour keeps its displayed brightness (a display value divided by the exposure)
        c = (direct + ambientDiff + ambientSpec) * s.tint + s.emissive / max(lighting.exposure.x, 1e-30);
    }
    if (view == viewDirectOnly) return debugDisplay(direct * s.tint);
    if (view == viewAmbientOnly) return debugDisplay(ambientDiff * s.tint);
    if (view == viewSpecularOnly) return debugDisplay(ambientSpec * s.tint);
    c = lightingFog(c, s.relPos);
    if (view == viewLightingOnly) return debugDisplay(c);
    if (view == viewExposure)
    {
        // The luminance's EV100 (L = 2^EV * 12.5 / 100); clay: its unit-less value
        float ev = log2(max(luma(c), 1e-6) * 8.0);
        return turbo((ev - lighting.debugParams.x) / max(lighting.debugParams.y, 1e-3) + 0.5);
    }
    return c;
}

#endif
