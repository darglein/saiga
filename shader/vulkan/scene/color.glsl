/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Color helpers of RenderScene's resolve and post passes (S17).

// The largest finite half float
const float maxHalf = 65504.0;

// Per channel: NaN -> 0, negative -> 0, +inf and values above the half float range -> 65504
vec3 sanitize(vec3 c)
{
    bvec3 nan = isnan(c);
    c         = clamp(c, vec3(0.0), vec3(maxHalf));
    return mix(c, vec3(0.0), nan);
}

float luma(vec3 c)
{
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// The piecewise sRGB curve (IEC 61966-2-1), c in [0, 1]
vec3 linearToSrgb(vec3 c)
{
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, greaterThan(c, vec3(0.0031308)));
}

vec3 srgbToLinear(vec3 c)
{
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), greaterThan(c, vec3(0.04045)));
}

// ACES filmic tone curve, the fit of Krzysztof Narkowicz (2016)
vec3 acesNarkowicz(vec3 x)
{
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
