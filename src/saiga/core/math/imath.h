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
    T r = a % b;
    if (r == 0) return a;
    // For negative a the remainder is negative, so a - r is already the next multiple above a.
    if constexpr (std::is_signed<T>::value)
    {
        if (r < 0) return a - r;
    }
    return a - r + b;
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
    T r = a % b;
    if constexpr (std::is_signed<T>::value)
    {
        if (r < 0) return a - r - b;
    }
    return a - r;
}

template <typename T>
HD constexpr int iFloor(T value)
{
    return ((int)value) - (((int)value) > value);
}

template <typename T = int, typename U = float>
HD constexpr T iCeil(U value)
{
    return ((T)value) + (((T)value) < value);
}

// Rounds half away from zero. Adding 0.5 before truncating would round the float just below 0.5 up to 1.
template <typename T>
HD constexpr int iRound(T value)
{
    int t  = (int)value;
    T frac = value - T(t);
    if (frac >= T(0.5)) return t + 1;
    if (frac <= T(-0.5)) return t - 1;
    return t;
}

// Like integer division, but also rounds down on negative numbers.
HD constexpr int iFloorDiv(int a, int b)
{
    int d = a / b;
    return d * b == a ? d : d - ((a < 0) ^ (b < 0));
}

HD constexpr bool ispow2(unsigned x)
{
    return x != 0 && !(x & (x - 1));
}

}  // namespace Saiga
