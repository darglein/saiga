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
#include "vertex_light.glsl"

// SurfaceInput::flags: foliage (wrapped diffuse) and surfaceShSpecular (the land of the terrain): rough dielectric
// surfaces whose specular environment is the diffuse irradiance's mean radiance (irradiance(n) / pi, already
// evaluated for the diffuse part) instead of the prefiltered cube: their wide GGX lobe sees about that, the F0 of 0.04
// keeps the difference small, and it saves the cube fetch per fragment of the grass's overdraw (S36a1c review)
const uint surfaceFoliage = 1u, surfaceShSpecular = 2u;

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
// the physical path its colour is the sky at the horizon in the view's azimuth (vertex_light.glsl's physicalFog)
vec3 lightingFog(vec3 c, vec3 relPos)
{
    if (!physicalLighting())
        return protoApplyFog(c, relPos, lighting.fogParams.w, vec4(lighting.fogParams.xyz, 0.0),
                             lighting.fogColorLin.rgb);
    vec4 fog = physicalFog(relPos);
    return c * fog.a + fog.rgb;
}

#ifdef VERTEX_LIGHT_INPUT
// The vertex shader's light terms (set by main from its inputs)
VertexLight vertexLightInput = VertexLight(vec3(0.0), vec3(0.0), vec4(0.0, 0.0, 0.0, 1.0));
#endif

// The light terms of a fragment (vertex_light.glsl): the vertex shader's (VERTEX_LIGHT_INPUT) or per fragment
VertexLight surfaceVertexLight(vec3 relPos)
{
#ifdef VERTEX_LIGHT_INPUT
    return vertexLightInput;
#else
    return computeVertexLight(relPos);
#endif
}

// The debug views (LightingDebugView of LightingTypes.h)
const uint viewNone = 0u, viewLightingOnly = 1u, viewAlbedo = 2u, viewNormals = 3u, viewRoughness = 4u,
           viewMetalness = 5u, viewMaterial = 6u, viewDirectOnly = 7u, viewAmbientOnly = 8u, viewSpecularOnly = 9u,
           viewCloudShadow = 10u, viewShadowFactor = 11u, viewExposure = 12u, viewShadowCascades = 13u;
// The overlay flags drawn in the shaders (LightingOverlay)
const uint overlayCascadeColors = 2u;

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
vec3 shadeSurfaceView(SurfaceInput s)
{
    const uint view = lighting.mode.z;
    if (view == viewAlbedo) return s.albedo;
    if (view == viewNormals) return s.n * 0.5 + 0.5;
    if (view == viewRoughness) return vec3(s.roughness);
    if (view == viewMetalness) return vec3(s.metalness);
    if (view == viewMaterial) return vec3(s.roughness, s.metalness, s.ao);
    if (view == viewCloudShadow) return vec3(cloudShadow(s.relPos));
    if (view == viewShadowFactor) return vec3(shadowFactor(s.relPos, s.n));
    if (view == viewShadowCascades)
    {
        // The cascade factor tinted by the cascade; raw in the probe frames: (factor, cascade + 1, 0)
        int cascade;
        float f = cascadeShadow(s.relPos, s.n, cascade);
        if (lighting.debugParams.z > 0.5) return vec3(f, float(cascade + 1), 0.0);
        return cascadeTint(cascade) * (0.2 + 0.8 * f);
    }
    if (view == viewLightingOnly)
    {
        s.albedo    = vec3(1.0);
        s.roughness = 0.5;
        s.metalness = 0.0;
        s.emissive  = vec3(0.0);
        s.tint      = vec3(1.0);
    }
    vec3 c, direct = vec3(0.0), ambientDiff = vec3(0.0), ambientSpec = vec3(0.0);
    vec4 fog = vec4(0.0, 0.0, 0.0, 1.0);
    if (!physicalLighting())
    {
        // S19a: albedo (sun + ambient) ao, the material's emissive on top
        vec3 L  = normalize(lighting.studioSun.xyz);
        float d = max(dot(s.n, L), 0.0) * lighting.studioSun.w;
        // S36a2: the cascades of the studio sun (only where it lights the surface)
        if (d > 0.0) d *= shadowFactor(s.relPos, s.n);
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
        VertexLight vl = surfaceVertexLight(s.relPos);
        fog            = vl.fog;
        if (lighting.mode.y == 0u)
        {
            // The shadow light's factor only where it reaches the surface (the wrapped diffuse of foliage included)
            float shadow = 1.0;
            if (lighting.shadowLight.w > 0.0 && dot(s.n, lighting.shadowLight.xyz) > -wrap)
                shadow = shadowFactor(s.relPos, s.n);
            // A body below the receiver's horizon gives no direct light (its light terms are 0 there,
            // computeVertexLight)
            if (lighting.sun.w > 0.0 && any(greaterThan(vl.sunE, vec3(0.0))))
            {
                vec3 sunE = vl.sunE;
                if (lighting.shadowLight.w == 1.0) sunE *= shadow;
                direct += pbrDirect(s.n, v, lighting.sun.xyz, s.albedo, rough, s.metalness, sunE, lut, wrap);
            }
            if (lighting.moon.w > 0.0 && any(greaterThan(vl.moonE, vec3(0.0))))
            {
                vec3 moonE = vl.moonE;
                if (lighting.shadowLight.w == 2.0) moonE *= shadow;
                direct += pbrDirect(s.n, v, lighting.moon.xyz, s.albedo, rough, s.metalness, moonE, lut, wrap);
            }
        }
        vec3 irradiance = ambientDiffuse(s.n, s.relPos);
        vec3 specular   = (s.flags & (surfaceFoliage | surfaceShSpecular)) != 0u
                              ? irradiance / kPi
                              : ambientSpecular(reflect(-v, s.n), rough, s.relPos);
        pbrAmbientParts(s.albedo, rough, s.metalness, NoV, lut, specular, irradiance, ambientSpec, ambientDiff);
        ambientSpec *= s.ao;
        ambientDiff *= s.ao;
        // The emissive colour keeps its displayed brightness (a display value divided by the exposure)
        c = (direct + ambientDiff + ambientSpec) * s.tint + s.emissive / max(lighting.exposure.x, 1e-30);
    }
    if (view == viewDirectOnly) return debugDisplay(direct * s.tint);
    if (view == viewAmbientOnly) return debugDisplay(ambientDiff * s.tint);
    if (view == viewSpecularOnly) return debugDisplay(ambientSpec * s.tint);
    // The fog: the physical path's from the light terms, clay's studio fog per fragment
    c = physicalLighting() ? c * fog.a + fog.rgb : lightingFog(c, s.relPos);
    if (view == viewLightingOnly) return debugDisplay(c);
    if (view == viewExposure)
    {
        // The luminance's EV100 (L = 2^EV * 12.5 / 100); clay: its unit-less value
        float ev = log2(max(luma(c), 1e-6) * 8.0);
        return turbo((ev - lighting.debugParams.x) / max(lighting.debugParams.y, 1e-3) + 0.5);
    }
    return c;
}

// shadeSurfaceView with the shader overlays (S36a2: CascadeColors tints every view by the receiver's cascade)
vec3 shadeSurface(SurfaceInput s)
{
    vec3 c = shadeSurfaceView(s);
    if ((lighting.mode.w & overlayCascadeColors) != 0u)
    {
        int cascade;
        cascadeShadow(s.relPos, s.n, cascade);
        c *= mix(vec3(1.0), cascadeTint(cascade), 0.6);
    }
    return c;
}

#endif
