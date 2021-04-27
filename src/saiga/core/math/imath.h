/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <type_traits>

namespace Saiga
{
// upwards rounding division for positve numbers
// returns the smallest x number with: x * b >= a
template <typename T, typename U>
HD constexpr T iDivUp(T a, U b)
{
    static_assert(std::is_integral<T>::value, "T must be integral!");
    return (a + b - T(1)) / b;
}

template <typename T>
HD constexpr T iDivDown(T a, T b)
{
    static_assert(std::is_integral<T>::value, "T must be integral!");
    return a / b;
}

/**
 * Align the value \p a to the next multiple of \p b
 * @tparam T Type of the value to align
 * @tparam U Type of the value to align to
 * @param a Value to align
 * @param b Value of alignment
 * @return The smallest multiple of \p b that is not less than \p a
 */
template <typename T, typename U>
HD constexpr T iAlignUp(T a, U b)
{
    static_assert(std::is_integral<T>::value && std::is_integral<U>::value, "only applicable to integral types");
    return (a % b != 0) ? (a - a % b + b) : a;
}

/**
 * finds the largest number that is smaller or equal than a and divisible by b
 * @tparam T Type of the value to align
 * @tparam U Type of the value to align to
 * @param a Value to align
 * @param b Value of alignment
 * @return The largest multiple of \p b that is not greater than \p a
 */
//
template <typename T, typename U>
HD constexpr T iAlignDown(T a, U b)
{
    static_assert(std::is_integral<T>::value && std::is_integral<U>::value, "only applicable to integral types");
    return a - a % b;
}

template <typename T>
HD constexpr int iFloor(T value)
{
    return ((int)value) - (((int)value) > value);
}

HD constexpr int iCeil(float value)
{
    return ((int)value) + (((int)value) < value);
}

HD constexpr int iRound(float value)
{
    return (int)(value + (value >= 0 ? 0.5f : -0.5f));
}

// Like integer division, but also rounds down on negative numbers.
HD constexpr int iFloorDiv(int a, int b)
{
    int d = a / b;
    return d * b == a ? d : d - ((a < 0) ^ (b < 0));
}

}  // namespace Saiga
