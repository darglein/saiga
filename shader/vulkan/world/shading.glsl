/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Shading of the world passes (terrain, water, grass, sky): prototype_shading.glsl with the world uniforms.
// Requires common.glsl.

#include "prototype_shading.glsl"

vec3 clayLighting(vec3 N, float ao)
{
    return protoClayLighting(N, ao, world.sunDir.xyz, world.sunDir.w, world.shading.x, world.shading.y);
}

vec3 applyFog(vec3 color, vec3 relPos)
{
    return protoApplyFog(color, relPos, world.cameraPos.y, world.fog, world.fogColor.rgb);
}

// shading.w > 0: linear values times the pre-exposure (shading.w); 0: display values (clay mode)
vec4 finalColor(vec3 c, float alpha)
{
    return protoFinalColor(c, alpha, world.shading.w);
}
