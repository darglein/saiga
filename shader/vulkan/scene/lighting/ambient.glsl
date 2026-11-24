/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The ambient light of S36 (A3, docs/engine_design.md §10.7; AmbientProvider / SkyAmbient on the CPU): the diffuse
// irradiance from the SH9 buffer (binding 2, already convolved: evalIrradianceSh gives lux) times the ambient scale and
// at least the floor, and the specular environment radiance. a1b: a uniform sky's SH and the SH as the specular
// environment; a1c: the environment map's SH and its prefiltered cube. Probes or occlusion methods replace SkyAmbient
// later without shader changes outside this file. Requires scene_block.glsl and pbr.glsl.
#ifndef AMBIENT_GLSL
#define AMBIENT_GLSL

layout(std430, set = 1, binding = 2) readonly buffer AmbientShBuffer
{
    // The irradiance SH9 (lux), rgb: 1, y, z, x, xy, yz, 3z^2 - 1, xz, x^2 - y^2 (the convolution and the basis
    // constants folded in on the CPU)
    vec4 ambientSh[9];
};

vec3 evalIrradianceSh(vec3 n)
{
    return ambientSh[0].rgb + ambientSh[1].rgb * n.y + ambientSh[2].rgb * n.z + ambientSh[3].rgb * n.x +
           ambientSh[4].rgb * (n.x * n.y) + ambientSh[5].rgb * (n.y * n.z) + ambientSh[6].rgb * (3.0 * n.z * n.z - 1.0) +
           ambientSh[7].rgb * (n.x * n.z) + ambientSh[8].rgb * (n.x * n.x - n.y * n.y);
}

// Clay mode (the studio environment): S19a's hemisphere (0.6 + 0.4 n.y) or constant ambient, unit-less
float studioAmbient(vec3 n)
{
    float a = lighting.studioShading.x;
    if (lighting.studioShading.y > 0.5) a *= 0.6 + 0.4 * n.y;
    return a;
}

// The diffuse irradiance (lux) for the normal n; an enclosed space: its template's ambient
vec3 ambientDiffuse(vec3 n, vec3 relPos)
{
    if (lighting.mode.y != 0u) return lighting.enclosedAmbient.rgb;
    return max(evalIrradianceSh(n) * lighting.minAmbient.w, lighting.minAmbient.rgb);
}

// The specular environment radiance in the reflection direction r (cd/m^2); an enclosed space: a dim constant
vec3 ambientSpecular(vec3 r, float roughness, vec3 relPos)
{
    if (lighting.mode.y != 0u) return lighting.enclosedAmbient.rgb * (0.25 / kPi);
    return ambientDiffuse(r, relPos) / kPi;
}

#endif
