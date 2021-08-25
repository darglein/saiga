/**
 * Copyright (c) 2021 Darius Rückert
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
    static_assert(std::is_floating_point<T>::value, "Only allowed for floating point types.");
    return std::numeric_limits<T>::epsilon();
}


template <typename T>
constexpr T pi()
{
    static_assert(std::is_floating_point<T>::value, "Only allowed for floating point types.");
    return T(3.14159265358979323846);
}

template <typename T>
constexpr T two_pi()
{
    static_assert(std::is_floating_point<T>::value, "Only allowed for floating point types.");
    return pi<T>() * T(2);
}


constexpr float clamp(float v, float mi, float ma)
{
    return std::min(ma, std::max(mi, v));
}

#ifndef IS_CUDA
// Already defined in CUDA's device_functions.hpp
SAIGA_HOST constexpr float saturate(float v)
{
    return clamp(v, 0, 1);
}
#endif

constexpr float smoothstep(float edge0, float edge1, float x)
{
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}


HD inline float fract(float a)
{
    return a - std::floor(a);
}


HD inline double fract(double a)
{
    return a - std::floor(a);
}

template <typename Derived>
HD constexpr typename Derived::PlainObject fract(const Eigen::MatrixBase<Derived>& v)
{
    return (v.array() - v.array().floor());
}

template <typename T>
constexpr T degrees(T a)
{
    static_assert(std::is_floating_point<T>::value, "Only allowed for floating point types.");
    return a * T(180.0) / pi<T>();
}

template <typename T>
constexpr T radians(T a)
{
    static_assert(std::is_floating_point<T>::value, "Only allowed for floating point types.");
    return a / T(180.0) * pi<T>();
}


HD inline float mix(const float& a, const float& b, float alpha)
{
    return (1 - alpha) * a + alpha * b;
}

// Maybe use more advanced implementation from boost?
// https://www.boost.org/doc/libs/1_51_0/boost/math/special_functions/sinc.hpp
template <typename T>
inline T sinc(const T x)
{
    if (abs(x) >= std::numeric_limits<T>::epsilon())
    {
        return (sin(x) / x);
    }
    else
    {
        return T(1);
    }
}

}  // namespace Saiga
