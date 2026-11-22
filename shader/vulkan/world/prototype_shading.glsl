/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The prototype's shading helpers with explicit parameters (S19a; S36: the light is lighting.glsl's shadeSurface):
// exponential height fog (lighting.glsl's lightingFog), the LOD tint, the output encoding and the alpha sharpening of
// the foliage cards. Shared by the terrain, water, grass and sky passes and RenderScene's PrototypeFoliage class.
// Fragment shaders only (derivatives). Guarded: lighting.glsl and terrain_shading.glsl both include it.
#ifndef PROTOTYPE_SHADING_GLSL
#define PROTOTYPE_SHADING_GLSL

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
// w base height.
vec3 protoApplyFog(vec3 color, vec3 relPos, float cameraY, vec4 fog, vec3 fogColor)
{
    if (fog.z < 0.5) return color;
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
    return mix(color, fogColor, clamp(f, 0.0, 1.0));
}

vec3 lodTint(float lod)
{
    const vec3 colors[6] = vec3[](vec3(1.0, 0.45, 0.45), vec3(0.45, 1.0, 0.45), vec3(0.5, 0.55, 1.0),
                                  vec3(1.0, 1.0, 0.45), vec3(1.0, 0.45, 1.0), vec3(0.45, 1.0, 1.0));
    return colors[int(max(lod, 0.0)) % 6];
}

// The output (S17's contract): linearScale > 0: linear values for RenderScene's tone map, multiplied by the
// pre-exposure (linearScale), only negatives clamped; linearScale == 0: display values (clay mode), clamped to [0, 1]
// and encoded for a UNORM target (the prototype's pow(1 / 2.2)).
vec4 protoFinalColor(vec3 c, float alpha, float linearScale)
{
    if (linearScale > 0.0) return vec4(max(c, vec3(0.0)) * linearScale, alpha);
    c = clamp(c, 0.0, 1.0);
    return vec4(pow(c, vec3(1.0 / 2.2)), alpha);
}

// The coverage of a foliage card (alpha to coverage): the texture's alpha, kept in the smaller mip levels and sharpened
// to about one pixel around the cutoff. The caller discards at <= 0 (and below 0.5 without MSAA).
float foliageCoverage(sampler2DArray foliage, vec3 uvLayer, float cutoff)
{
    float alpha = texture(foliage, uvLayer).r;
    // Keep the coverage in the smaller mip levels
    alpha *= 1.0 + max(textureQueryLod(foliage, uvLayer.xy).x, 0.0) * 0.25;
    // Sharpen the edge to about one pixel (alpha to coverage)
    return (alpha - cutoff) / max(fwidth(alpha), 1e-4) + 0.5;
}
#endif
