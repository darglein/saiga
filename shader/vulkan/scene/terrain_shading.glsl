/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The output of the terrain, water, grass and sky passes (S20; S36: their light is shadeSurface's, lighting.glsl):
// prototype_shading.glsl's encoding with the frame's pre-exposure. Requires terrain_common.glsl.

#include "../world/prototype_shading.glsl"

// shading.w > 0: linear values times the pre-exposure (shading.w); 0: display values (clay mode)
vec4 finalColor(vec3 c, float alpha)
{
    return protoFinalColor(c, alpha, frame.shading.w);
}
