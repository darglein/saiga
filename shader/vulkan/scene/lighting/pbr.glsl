/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The BRDF of S36 (A3, docs/engine_design.md §10.7): GGX normal distribution, height-correlated Smith visibility,
// Schlick Fresnel with F0 = mix(0.04, albedo, metalness), Lambert diffuse weighted by (1 - F) (1 - metalness), and the
// multi-scatter energy compensation of Fdez-Agüera (2019) with the split-sum BRDF LUT (brdf_lut.comp). Shared by
// lighting.glsl and the LUT's compute shader. roughness is perceptual (alpha = roughness^2).
#ifndef PBR_GLSL
#define PBR_GLSL

const float kPi = 3.14159265358979;

// The smallest perceptual roughness (fp16 / specular aliasing)
float pbrRoughness(float r)
{
    return clamp(r, 0.045, 1.0);
}

float distributionGgx(float NoH, float a)
{
    float a2 = a * a;
    float f  = (NoH * a2 - NoH) * NoH + 1.0;
    return a2 / (kPi * f * f);
}

// The height-correlated Smith visibility V = G / (4 NoV NoL)
float visibilitySmithGgx(float NoV, float NoL, float a)
{
    float a2 = a * a;
    float gv = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
    float gl = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-7);
}

vec3 fresnelSchlick(float VoH, vec3 f0)
{
    float f = pow(1.0 - VoH, 5.0);
    return f0 + (vec3(1.0) - f0) * f;
}

vec3 pbrF0(vec3 albedo, float metalness)
{
    return mix(vec3(0.04), albedo, metalness);
}

// The energy compensation of the single-scattering specular lobe (direct light): 1 + F0 (1 / Ess - 1) with the LUT's
// directional albedo Ess = A + B
vec3 pbrEnergyCompensation(vec3 f0, vec2 lut)
{
    return vec3(1.0) + f0 * (1.0 / max(lut.x + lut.y, 1e-4) - 1.0);
}

// The reflected radiance of a directional light of illuminance E (lux) from l (towards the light): (diffuse + specular)
// E NoL; v towards the viewer. wrap > 0: wrapped diffuse for foliage (n.l + wrap) / (1 + wrap)
vec3 pbrDirect(vec3 n, vec3 v, vec3 l, vec3 albedo, float roughness, float metalness, vec3 E, vec2 lut, float wrap)
{
    float NoLs = dot(n, l);
    float NoLd = clamp((NoLs + wrap) / (1.0 + wrap), 0.0, 1.0);
    if (NoLd <= 0.0) return vec3(0.0);
    float NoL = clamp(NoLs, 0.0, 1.0);
    // v + l is zero for a light exactly behind the viewer, which the wrapped diffuse admits: the normal instead (NoL
    // is 0 there, the specular term vanishes, the Fresnel stays finite)
    vec3 hv   = v + l;
    float hl  = dot(hv, hv);
    vec3 h    = hl > 1e-12 ? hv * inversesqrt(hl) : n;
    float NoV = max(dot(n, v), 1e-4);
    float NoH = clamp(dot(n, h), 0.0, 1.0);
    float VoH = clamp(dot(v, h), 0.0, 1.0);
    float a   = roughness * roughness;
    vec3 f0   = pbrF0(albedo, metalness);
    vec3 F    = fresnelSchlick(VoH, f0);
    vec3 spec = distributionGgx(NoH, a) * visibilitySmithGgx(NoV, NoL, a) * F * pbrEnergyCompensation(f0, lut);
    vec3 diff = (vec3(1.0) - F) * (1.0 - metalness) * albedo / kPi;
    return (diff * NoLd + spec * NoL) * E;
}

// The image-based terms of Fdez-Agüera: radiance = the prefiltered specular radiance in the reflection direction,
// irradiance = the diffuse irradiance (lux); the reflected radiance in its specular (single scattering) and diffuse
// (multiple scattering and Lambert) parts
void pbrAmbientParts(vec3 albedo, float roughness, float metalness, float NoV, vec2 lut, vec3 radiance,
                     vec3 irradiance, out vec3 specular, out vec3 diffuse)
{
    vec3 f0     = pbrF0(albedo, metalness);
    vec3 Fr     = max(vec3(1.0 - roughness), f0) - f0;
    vec3 kS     = f0 + Fr * pow(1.0 - NoV, 5.0);
    vec3 FssEss = kS * lut.x + lut.y;
    float Ess   = lut.x + lut.y;
    float Ems   = 1.0 - Ess;
    vec3 Favg   = f0 + (vec3(1.0) - f0) / 21.0;
    vec3 Fms    = FssEss * Favg / (vec3(1.0) - Ems * Favg);
    vec3 Edss   = vec3(1.0) - (FssEss + Fms * Ems);
    vec3 kD     = albedo * (1.0 - metalness) * Edss;
    specular    = FssEss * radiance;
    diffuse     = (Fms * Ems + kD) * irradiance / kPi;
}

vec3 pbrAmbient(vec3 albedo, float roughness, float metalness, float NoV, vec2 lut, vec3 radiance, vec3 irradiance)
{
    vec3 specular, diffuse;
    pbrAmbientParts(albedo, roughness, metalness, NoV, lut, radiance, irradiance, specular, diffuse);
    return specular + diffuse;
}

// Importance sampling of the GGX lobe (the LUT and later the environment prefilter): the half vector in tangent space
// (n = +z) for the 2D sample xi
vec3 importanceSampleGgx(vec2 xi, float a)
{
    float phi      = 2.0 * kPi * xi.x;
    float cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    return vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

// The Hammersley point i of n (deterministic sample sets)
vec2 hammersley(uint i, uint n)
{
    uint b = bitfieldReverse(i);
    return vec2(float(i) / float(n), float(b) * 2.3283064365386963e-10);
}

#endif
