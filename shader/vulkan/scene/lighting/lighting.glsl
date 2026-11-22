/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The lighting of S36 (A3 of docs/stages/S36_lighting.plan.md, docs/engine_design.md §10.7): set 1 of the scene draws
// (SceneLightingSet: 0 the block, 1 the foliage texture, 2 the ambient SH, 3 the cloud coverage, 4 the prefiltered
// environment cube, 5 the BRDF LUT, 6 the transmittance LUT, 7 the sky-view LUT; a2: 8 the cascades, 9 the horizon map;
// b: 10-14 the lights, clusters and atlas) and shadeSurface, which every scene pass of the LIGHTING_PBR variant calls,
// so every surface lights the same way:
// - clay mode (lighting.mode.x == 0: ToneMap::Identity, or no environment): the studio environment, S19a's model
//   exactly: albedo (sun max(n.l, 0) + ambient) ao, the studio fog; unit-less;
// - physical (an environment and a tone map): GGX (pbr.glsl) for the sun and the moon in lux with the shadow factor of
//   the shadow light, the ambient (ambient.glsl), the active fog; radiance in cd/m^2 before the pre-exposure.
// Fragment shaders only. The caller applies the output (pre-exposure or the display encoding).
#ifndef LIGHTING_GLSL
#define LIGHTING_GLSL

#include "scene_block.glsl"

#include "../../world/prototype_shading.glsl"

layout(set = 1, binding = 3) uniform sampler2D cloudCoverage;
layout(set = 1, binding = 4) uniform samplerCube environmentCube;
layout(set = 1, binding = 5) uniform sampler2D brdfLut;
layout(set = 1, binding = 6) uniform sampler2D transmittanceLut;
layout(set = 1, binding = 7) uniform sampler2D skyViewLut;

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

// The active fog (studio or physical, chosen on the CPU): S19a's height fog with the camera height above its base
vec3 lightingFog(vec3 c, vec3 relPos)
{
    return protoApplyFog(c, relPos, lighting.fogParams.w, vec4(lighting.fogParams.xyz, 0.0), lighting.fogColorLin.rgb);
}

// The direct light's attenuation until the atmosphere (a1c: the transmittance LUT): the body below the horizon is off
vec3 bodyTransmittance(vec3 l)
{
    return vec3(smoothstep(-0.02, 0.02, l.y));
}

// The light of a surface (radiance before the pre-exposure) with the fog
vec3 shadeSurface(SurfaceInput s)
{
    vec3 c;
    if (!physicalLighting())
    {
        // S19a: albedo (sun + ambient) ao, the material's emissive on top
        vec3 L  = normalize(lighting.studioSun.xyz);
        float d = max(dot(s.n, L), 0.0) * lighting.studioSun.w;
        c       = s.albedo * (d + studioAmbient(s.n)) * s.ao;
        c       = c * s.tint + s.emissive;
    }
    else
    {
        vec3 v      = normalize(-s.relPos);
        float rough = pbrRoughness(s.roughness);
        float NoV   = max(dot(s.n, v), 1e-4);
        vec2 lut    = texture(brdfLut, vec2(NoV, rough)).rg;
        float wrap  = (s.flags & surfaceFoliage) != 0u ? 0.3 : 0.0;
        vec3 direct = vec3(0.0);
        if (lighting.mode.y == 0u)
        {
            float shadow = shadowFactor(s.relPos, s.n);
            vec3 sunE    = lighting.sun.w * bodyTransmittance(lighting.sun.xyz);
            vec3 moonE   = lighting.moon.w * bodyTransmittance(lighting.moon.xyz);
            if (lighting.shadowLight.w == 1.0) sunE *= shadow;
            if (lighting.shadowLight.w == 2.0) moonE *= shadow;
            if (lighting.sun.w > 0.0)
                direct += pbrDirect(s.n, v, lighting.sun.xyz, s.albedo, rough, s.metalness, sunE, lut, wrap);
            if (lighting.moon.w > 0.0)
                direct += pbrDirect(s.n, v, lighting.moon.xyz, s.albedo, rough, s.metalness, moonE, lut, wrap);
        }
        vec3 r   = reflect(-v, s.n);
        vec3 amb = pbrAmbient(s.albedo, rough, s.metalness, NoV, lut, ambientSpecular(r, rough, s.relPos),
                              ambientDiffuse(s.n, s.relPos));
        // The emissive colour keeps its displayed brightness (a display value divided by the exposure)
        c = (direct + amb * s.ao) * s.tint + s.emissive / max(lighting.exposure.x, 1e-30);
    }
    return lightingFog(c, s.relPos);
}

#endif
