/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <cmath>

namespace Saiga
{

template <typename T>
constexpr T epsilon()
{
    return std::numeric_limits<T>::epsilon();
}


template <typename T>
constexpr T pi()
{
    return T(3.14159265358979323846);
}

template <typename T>
constexpr T two_pi()
{
    return pi<T>() * T(2);
}


constexpr float clamp(float v, float mi, float ma)
{
    return std::min(ma, std::max(mi, v));
}

HD inline float smoothstep(float edge0, float edge1, float x)
{
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}


HD inline float fract(float a)
{
    return a - std::floor(a);
}

HD inline float degrees(float a)
{
    return a * 180.0 / pi<float>();
}

HD inline float radians(float a)
{
    return a / 180.0 * pi<float>();
}


HD inline float mix(const float& a, const float& b, float alpha)
{
    return (1 - alpha) * a + alpha * b;
}

}  // namespace Saiga


