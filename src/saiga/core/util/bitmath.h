/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <cstring>

namespace Saiga
{
// Type punning through memcpy (C++17 has no std::bit_cast). A pointer cast violates strict aliasing.
HD inline float bitsToFloat(unsigned int x)
{
    static_assert(sizeof(float) == sizeof(unsigned int));
    float f;
    memcpy(&f, &x, sizeof(f));
    return f;
}

HD inline unsigned int floatToBits(float f)
{
    unsigned int x;
    memcpy(&x, &f, sizeof(x));
    return x;
}

// Unsigned int to normalized signed float in the range [-1,1]
HD inline float uintToNSFloat(unsigned int x)
{
    // Use High Bits of x.
    // Remove this line to use low bits.
    x = x >> (32 - 23);

    // Write into the mantissa of the float.
    x = (x & 0x007fffff);

    // Set exponent to 1 so we land in the range [2,4].
    // 2^1 * 1.Mantissa
    x = x | 0x40000000;
    return bitsToFloat(x) - 3.0f;
}

// Normalized float in the range [0,1] to unsigned int.
HD inline unsigned int NSFloatToUint(float f)
{
    f += 3.0f;
    unsigned int x = floatToBits(f);

    // Remove everything but the mantissa
    x = (x & 0x007fffff);

    // Use High Bits of x.
    // Remove this line to use low bits.
    x = x << (32 - 23);

    return x;
}

// Unsigned int to normalized float in the range [0,1]
HD inline float uintToNFloat(unsigned int x)
{
    // Use High Bits of x.
    // Remove this line to use low bits.
    x = x >> (32 - 23);

    // Write into the mantissa of the float.
    x = (x & 0x007fffff);

    // Set exponent to 1 so we land in the range [1,2].
    // 2^0 * 1.Mantissa
    x = x | 0x3f800000;
    return bitsToFloat(x) - 1.0f;
}

// Normalized float in the range [0,1] to unsigned int.
HD inline unsigned int NFloatToUint(float f)
{
    f += 1.0f;
    unsigned int x = floatToBits(f);

    // Remove everything but the mantissa
    x = (x & 0x007fffff);

    // Use High Bits of x.
    // Remove this line to use low bits.
    x = x << (32 - 23);

    return x;
}

// source https://www.geeksforgeeks.org/count-trailing-zero-bits-using-lookup-table/
HD inline int countTrailingZero(int x)
{
    // Map a bit value mod 37 to its position
    static const int lookup[] = {32, 0,  1,  26, 2,  23, 27, 0,  3, 16, 24, 30, 28, 11, 0,  13, 4,  7, 17,
                                 0,  25, 22, 31, 15, 29, 10, 12, 6, 0,  21, 14, 9,  5,  20, 8,  19, 18};

    // x & -x isolates the lowest set bit. Negate the unsigned value, -x overflows for INT_MIN.
    unsigned int u = static_cast<unsigned int>(x);
    return lookup[(u & (0u - u)) % 37];
}

}  // namespace Saiga
