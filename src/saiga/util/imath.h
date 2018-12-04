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
HD SAIGA_CONSTEXPR inline int iDivUp(int a, int b)
{
    return (a + b - 1) / b;
}

HD SAIGA_CONSTEXPR inline int iDivDown(int a, int b)
{
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


HD inline int iFloor(float value)
{
    int i = (int)value;
    return i - (i > value);
}

HD inline int iCeil(float value)
{
    int i = (int)value;
    return i + (i < value);
}

HD inline int iRound(float value)
{
    return (int)(value + (value >= 0 ? 0.5f : -0.5f));
}


}  // namespace Saiga
