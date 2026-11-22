/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Helpers of S36's debug views and previews (A8).
#ifndef DEBUG_COMMON_GLSL
#define DEBUG_COMMON_GLSL

// The Turbo colour map (Google, 2019; the polynomial approximation), t in [0, 1] -> display colour
vec3 turbo(float t)
{
    const vec4 kRedVec4   = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const vec4 kGreenVec4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const vec4 kBlueVec4  = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const vec2 kRedVec2   = vec2(-152.94239396, 59.28637943);
    const vec2 kGreenVec2 = vec2(4.27729857, 2.82956604);
    const vec2 kBlueVec2  = vec2(-89.90310912, 27.34824973);
    t                     = clamp(t, 0.0, 1.0);
    vec4 v4               = vec4(1.0, t, t * t, t * t * t);
    vec2 v2               = v4.zw * v4.z;
    return clamp(vec3(dot(v4, kRedVec4) + dot(v2, kRedVec2), dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
                      dot(v4, kBlueVec4) + dot(v2, kBlueVec2)),
                 0.0, 1.0);
}

#endif
