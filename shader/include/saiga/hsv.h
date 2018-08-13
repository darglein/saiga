/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#ifndef HSV_H
#define HSV_H

#include "shaderConfig.h"

/**
 * Conversion from normalized HSV to RGB and back.
 * All HSV elements are in the range [0,1]
 */

FUNC_DECL vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

FUNC_DECL vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0f - K.www);
    return c.z * mix( vec3(K.xxx), vec3(clamp(p - K.xxx, vec3(0.0f), vec3(1.0f) )), c.y);
}



/**
 * Creates a smooth HSV color transition from Blue(=0) to Red(=1).
 * The returned value is in RGB space.
 */
FUNC_DECL vec3 colorizeBlueRed(float alpha)
{
    alpha = 1.f - clamp(alpha,0.f,1.f);
    return hsv2rgb(vec3(
                       alpha * (240.0f/360.0f),
                       1,
                       1
                       ));
}

/**
 * Similar to above but from Red(=0) to Green(=1).
 */
FUNC_DECL vec3 colorizeRedGreen(float alpha)
{
    alpha = clamp(alpha,0.f,1.f);
    return hsv2rgb(vec3(
                       alpha * (120.0f/360.0f),
                       1,
                       1
                       ));
}

#endif
