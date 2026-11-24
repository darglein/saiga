/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The prototype's exponential height fog (S19a; prototype_shading.glsl, S36: lighting.glsl and vertex_light.glsl):
// no derivatives, usable in vertex shaders. Guarded.
#ifndef PROTOTYPE_FOG_GLSL
#define PROTOTYPE_FOG_GLSL

// Optical depth of a ray segment of length L above the water level whose heights above the water go from hA to hB
// (both >= 0): the integral of a * exp(-b * height), computed from the bounded densities at both ends (no exp of a
// large positive number, which overflowed for a high camera looking down).
float fogAbove(float L, float hA, float hB, float a, float b)
{
    float eA = exp(-b * hA);
    float eB = exp(-b * hB);
    float dh = hB - hA;
    return abs(b * dh) > 1e-4 ? a * L * (eA - eB) / (b * dh) : a * L * 0.5 * (eA + eB);
}

// Exponential height fog: the density a * exp(-b * height) decreases with the height above the base (the water level)
// and is constant (a) below it. relPos: the position relative to the camera; fog: x density a, y falloff b, z enabled,
// w base height. protoFogFactor: the factor of the fog colour (0 without fog), protoApplyFog: the fogged colour.
float protoFogFactor(vec3 relPos, float cameraY, vec4 fog)
{
    if (fog.z < 0.5) return 0.0;
    float dist = length(relPos);
    float a    = fog.x;
    float b    = fog.y;
    // Signed heights above the base at both ends of the ray
    float y0 = cameraY - fog.w;
    float y1 = y0 + relPos.y;
    float optical;
    if (y0 <= 0.0 && y1 <= 0.0)
    {
        optical = a * dist;
    }
    else if (y0 > 0.0 && y1 > 0.0)
    {
        optical = fogAbove(dist, y0, y1, a, b);
    }
    else
    {
        // The ray crosses the base: constant density below, exponential above
        float below = dist * (y0 <= 0.0 ? -y0 : -y1) / abs(y1 - y0);
        optical     = a * below + fogAbove(dist - below, 0.0, max(y0, y1), a, b);
    }
    float f = 1.0 - exp(-optical);
    return clamp(f, 0.0, 1.0);
}

vec3 protoApplyFog(vec3 color, vec3 relPos, float cameraY, vec4 fog, vec3 fogColor)
{
    if (fog.z < 0.5) return color;
    return mix(color, fogColor, protoFogFactor(relPos, cameraY, fog));
}

#endif
