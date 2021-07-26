/**
 * Copyright (c) 2021 Darius Rückert
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

    vec4 p = mix(vec4(c[2], c[1], K[3], K[2]), vec4(c[1], c[2], K[0], K[1]), mix(1.f, 0.f, c[2] < c[1]));
    vec4 q = mix(vec4(p[0], p[1], p[3], c[0]), vec4(c[0], p[1], p[2], p[0]), mix(1.f, 0.f, p[0] < c[0]));

    float d = q[0] - min(q[3], q[1]);
    float e = 1.0e-10;
    return vec3(abs(q[2] + (q[3] - q[1]) / (6.0 * d + e)), d / (q[0] + e), q[0]);
}

FUNC_DECL vec3 hsv2rgb(vec3 c)
{
    vec4 K   = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 fra = fract(vec3(c[0], c[0], c[0]) + vec3(K[0], K[1], K[2]));
    // vec3 fra = (vec3(c[0], c[0], c[0]) + vec3(K[0], K[1], K[2]));
    // fra      = fra.array() - fra.array().floor();
    vec3 p = vec3(fra * 6.0f - vec3(K[3], K[3], K[3]));
    p      = abs(p);
    return c[2] * mix(make_vec3(K[0]), clamp(vec3(p - make_vec3(K[0])), make_vec3(0.0), make_vec3(1.0)), c[1]);
}

#endif
