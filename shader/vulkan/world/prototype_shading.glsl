/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The prototype's shading helpers with explicit parameters (S19a; S36: the light is lighting.glsl's shadeSurface):
// exponential height fog (lighting.glsl's lightingFog), the LOD tint, the output encoding and the alpha sharpening of
// the foliage cards. Shared by the terrain, water, grass and sky passes and RenderScene's PrototypeFoliage class.
// Fragment shaders only (derivatives; the fog alone is prototype_fog.glsl, also for vertex shaders). Guarded:
// lighting.glsl and terrain_shading.glsl both include it.
#ifndef PROTOTYPE_SHADING_GLSL
#define PROTOTYPE_SHADING_GLSL

#include "prototype_fog.glsl"

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
