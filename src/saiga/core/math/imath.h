/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <type_traits>

namespace Saiga
{
// returns the smallest x number with: x * b >= a
template <typename T>
HD SAIGA_CONSTEXPR inline T iDivUp(T a, T b)
{
    static_assert(std::is_integral<T>::value, "T must be integral!");
    return (a + b - T(1)) / b;
}

template <typename T>
HD SAIGA_CONSTEXPR inline T iDivDown(T a, T b)
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
HD SAIGA_CONSTEXPR inline T iAlignUp(T a, U b)
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
HD SAIGA_CONSTEXPR inline T iAlignDown(T a, U b)
{
    static_assert(std::is_integral<T>::value && std::is_integral<U>::value, "only applicable to integral types");
    return a - a % b;
}


HD constexpr inline int iFloor(float value)
{
    int i = (int)value;
    return i - (i > value);
}

HD constexpr inline int iCeil(float value)
{
    int i = (int)value;
    return i + (i < value);
}

HD constexpr inline int iRound(float value)
{
    return (int)(value + (value >= 0 ? 0.5f : -0.5f));
}


}  // namespace Saiga
