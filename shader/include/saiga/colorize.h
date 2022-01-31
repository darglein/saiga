/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#ifndef COLORIZE_H
#define COLORIZE_H

#include "hsv.h"
#include "shaderConfig.h"

/**
 * Creates a smooth HSV color transition from Blue(=0) to Red(=1).
 * The returned value is in RGB space.
 */
FUNC_DECL vec3 colorizeBlueRed(float alpha)
{
    alpha = 1.f - saturate(alpha);
    return hsv2rgb(vec3(alpha * (240.0f / 360.0f), 1, 1));
}

/**
 * Similar to above but from Red(=0) to Green(=1).
 */
FUNC_DECL vec3 colorizeRedGreen(float alpha)
{
    alpha = saturate(alpha);
    return hsv2rgb(vec3(alpha * (120.0f / 360.0f), 1, 1));
}

/**
 * A temperature like color sheme from 0=black to 1=white
 */
FUNC_DECL vec3 colorizeFusion(float x)
{
    float t = saturate(x);
    return saturate(vec3(sqrt(t), t * t * t, max(sin(3.1415 * 1.75 * t), pow(t, 12.0))));
}

FUNC_DECL vec3 colorizeFusionHDR(float x)
{
    float t = saturate(x);
    return colorizeFusion(sqrt(t)) * (t * 2.0f + 0.5f);
}

/**
 * All credits to:
 * Viridis approximation, Jerome Liard, August 2016
 * https://www.shadertoy.com/view/XtGGzG
 */
FUNC_DECL vec3 colorizeInferno(float x)
{
    x       = saturate(x);
    vec4 x1 = vec4(1.0, x, x * x, x * x * x);  // 1 x x2 x3
    vec4 x2 = x1 * x1[3] * x;                  // x4 x5 x6 x7
    return vec3(dot(vec4(x1), vec4(-0.027780558, +1.228188385, +0.278906882, +3.892783760)) +
                    dot(vec2(x2[0], x2[1]), vec2(-8.490712758, +4.069046086)),
                dot(vec4(x1), vec4(+0.014065206, +0.015360518, +1.605395918, -4.821108251)) +
                    dot(vec2(x2[0], x2[1]), vec2(+8.389314011, -4.193858954)),
                dot(vec4(x1), vec4(-0.019628385, +3.122510347, -5.893222355, +2.798380308)) +
                    dot(vec2(x2[0], x2[1]), vec2(-3.608884658, +4.324996022)));
}

FUNC_DECL vec3 colorizeViridis(float t)
{
    const vec3 c0 = vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
    const vec3 c1 = vec3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
    const vec3 c2 = vec3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
    const vec3 c3 = vec3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
    const vec3 c4 = vec3(6.228269936347081, 14.17993336680509, 56.69055260068105);
    const vec3 c5 = vec3(4.776384997670288, -13.74514537774601, -65.35303263337234);
    const vec3 c6 = vec3(-5.435455855934631, 4.645852612178535, 26.3124352495832);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}


FUNC_DECL vec3 colorizeBone(float x)
{
    float r, g, b;

    if (x < 0.75)
    {
        r = 8.0 / 9.0 * x - (13.0 + 8.0 / 9.0) / 1000.0;
    }
    else
    {
        r = (13.0 + 8.0 / 9.0) / 10.0 * x - (3.0 + 8.0 / 9.0) / 10.0;
    }

    if (x <= 0.375)
    {
        g = 8.0 / 9.0 * x - (13.0 + 8.0 / 9.0) / 1000.0;
    }
    else if (x <= 0.75)
    {
        g = (1.0 + 2.0 / 9.0) * x - (13.0 + 8.0 / 9.0) / 100.0;
    }
    else
    {
        g = 8.0 / 9.0 * x + 1.0 / 9.0;
    }

    if (x <= 0.375)
    {
        b = (1.0 + 2.0 / 9.0) * x - (13.0 + 8.0 / 9.0) / 1000.0;
    }
    else
    {
        b = 8.0 / 9.0 * x + 1.0 / 9.0;
    }

    r = clamp(r, 0.0f, 1.0f);
    g = clamp(g, 0.0f, 1.0f);
    b = clamp(b, 0.0f, 1.0f);
    return vec3(r, g, b);
}

FUNC_DECL vec3 colorizeMagma(float x)
{
    x       = saturate(x);
    vec4 x1 = vec4(1.0, x, x * x, x * x * x);  // 1 x x2 x3
    vec4 x2 = x1 * x1[3] * x;                  // x4 x5 x6 x7
    return vec3(dot(vec4(x1), vec4(-0.023226960, +1.087154378, -0.109964741, +6.333665763)) +
                    dot(vec2(x2[0], x2[1]), vec2(-11.640596589, +5.337625354)),
                dot(vec4(x1), vec4(+0.010680993, +0.176613780, +1.638227448, -6.743522237)) +
                    dot(vec2(x2[0], x2[1]), vec2(+11.426396979, -5.523236379)),
                dot(vec4(x1), vec4(-0.008260782, +2.244286052, +3.005587601, -24.279769818)) +
                    dot(vec2(x2[0], x2[1]), vec2(+32.484310068, -12.688259703)));
}

FUNC_DECL vec3 colorizePlasma(float x)
{
    x       = saturate(x);
    vec4 x1 = vec4(1.0, x, x * x, x * x * x);  // 1 x x2 x3
    vec4 x2 = x1 * x1[3] * x;                  // x4 x5 x6 x7
    return vec3(dot(x1, vec4(+0.063861086, +1.992659096, -1.023901152, -0.490832805)) +
                    dot(vec2(x2[0], x2[1]), vec2(+1.308442123, -0.914547012)),
                dot(x1, vec4(+0.049718590, -0.791144343, +2.892305078, +0.811726816)) +
                    dot(vec2(x2[0], x2[1]), vec2(-4.686502417, +2.717794514)),
                dot(x1, vec4(+0.513275779, +1.580255060, -5.164414457, +4.559573646)) +
                    dot(vec2(x2[0], x2[1]), vec2(-1.916810682, +0.570638854)));
}

// Blog: https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html
// Code: https://gist.github.com/mikhailov-work/0d177465a8151eb6ede1768d51d476c7
FUNC_DECL vec3 colorizeTurbo(float x)
{
    const vec4 kRedVec4   = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const vec4 kGreenVec4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const vec4 kBlueVec4  = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const vec2 kRedVec2   = vec2(-152.94239396, 59.28637943);
    const vec2 kGreenVec2 = vec2(4.27729857, 2.82956604);
    const vec2 kBlueVec2  = vec2(-89.90310912, 27.34824973);

    x       = saturate(x);
    vec4 v4 = vec4(1.0, x, x * x, x * x * x);
    // vec2 v2 = v4.zw * v4.z;
    vec2 v2 = vec2(v4[2], v4[3]) * v4[2];
    return vec3(dot(v4, kRedVec4) + dot(v2, kRedVec2), dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
                dot(v4, kBlueVec4) + dot(v2, kBlueVec2));
}

#endif
