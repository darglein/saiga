/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Shading of the terrain, water, grass and sky passes (S20; a copy of world/shading.glsl on TerrainFrame):
// prototype_shading.glsl with the frame's uniforms. Requires terrain_common.glsl.

#include "../world/prototype_shading.glsl"

vec3 clayLighting(vec3 N, float ao)
{
    return protoClayLighting(N, ao, frame.sunDir.xyz, frame.sunDir.w, frame.shading.x, frame.shading.y);
}

// The fog's base is 0 here: fog.w is the camera height above it (computed in double)
vec3 applyFog(vec3 color, vec3 relPos)
{
    return protoApplyFog(color, relPos, frame.fog.w, vec4(frame.fog.xyz, 0.0), frame.fogColor.rgb);
}

// shading.w > 0: linear values times the pre-exposure (shading.w); 0: display values (clay mode)
vec4 finalColor(vec3 c, float alpha)
{
    return protoFinalColor(c, alpha, frame.shading.w);
}
