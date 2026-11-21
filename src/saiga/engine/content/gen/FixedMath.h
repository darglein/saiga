/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// FixedMath (docs/engine_design.md §12.1, R-CONTENT-1; S23a): the integer and fixed-point helpers of the content
// generators. Header only, integer only, the same results on GCC, Clang and MSVC.
//
// The rules of content code (src/saiga/engine/content/gen/, the generator interface headers, games' content/
// directories; enforced by review, the ContentLint test and UBSan on the Clang CI job):
// - multiply narrow unsigned types only after a cast to uint32_t / int64_t (uint16_t * uint16_t promotes to int);
// - shift left only unsigned values or non-negative values known to fit; no std::abs of INT64_MIN;
// - the floor helpers below for possibly negative values ('/' and '%' truncate and would mirror content at the axes);
// - sort only on unique keys (std::sort is not stable);
// - no 'long' (32 bits on Windows) and no 'size_t': container sizes are converted with an explicit uint32_t(...) cast;
// - no floating literals and no libm, also not through a header that includes <cmath>;
// - every narrowing conversion (yaw and slot arithmetic in uint16_t / uint8_t, int64_t to int32_t) is an explicit
//   cast (MSVC's C4244 / C4267 are errors in CI).

#pragma once

#include "saiga/core/util/assert.h"

#include <cstdint>
#include <limits>

namespace Saiga
{
namespace Engine
{
// Candidate and prefab x / z positions: 1/256 m; heights: 1/1024 m (S20's height unit)
constexpr int64_t kPosUnitsPerMetre    = 256;
constexpr int64_t kHeightUnitsPerMetre = 1024;

// Implementation defined in C++17; GCC, Clang and MSVC shift signed values arithmetically
static_assert((int64_t(-3) >> 1) == -2, "the content code needs an arithmetic right shift");

// a / b rounded toward negative infinity, b > 0
constexpr int64_t floorDiv(int64_t a, int64_t b)
{
    SAIGA_ASSERT(b > 0, "floorDiv: the divisor must be positive");
    int64_t q = a / b;
    if (a % b != 0 && a < 0) --q;
    return q;
}

// a / b rounded toward positive infinity, b > 0
constexpr int64_t ceilDiv(int64_t a, int64_t b)
{
    SAIGA_ASSERT(b > 0, "ceilDiv: the divisor must be positive");
    int64_t q = a / b;
    if (a % b != 0 && a > 0) ++q;
    return q;
}

// The remainder of floorDiv: in [0, b)
constexpr int64_t floorMod(int64_t a, int64_t b)
{
    SAIGA_ASSERT(b > 0, "floorMod: the divisor must be positive");
    const int64_t r = a % b;
    return r < 0 ? r + b : r;
}

// a / 2^s rounded toward negative infinity (an arithmetic right shift), 0 <= s < 63
constexpr int64_t floorShift(int64_t a, int s)
{
    SAIGA_ASSERT(s >= 0 && s < 63, "floorShift: 0 <= s < 63");
    return a >> s;
}

// The product a * b fits into int64 (checked with SAIGA_ASSERTS by mulShift)
constexpr bool productFits(int64_t a, int64_t b)
{
    if (a == std::numeric_limits<int64_t>::min() || b == std::numeric_limits<int64_t>::min()) return false;
    const int64_t aa = a < 0 ? -a : a;
    const int64_t bb = b < 0 ? -b : b;
    return bb == 0 || aa <= std::numeric_limits<int64_t>::max() / bb;
}

// floorShift(a * b, s); the product must fit (asserted with SAIGA_ASSERTS; no __int128, MSVC has none)
constexpr int64_t mulShift(int64_t a, int64_t b, int s)
{
    SAIGA_ASSERT(productFits(a, b), "mulShift: the product overflows int64");
    return floorShift(a * b, s);
}

// a + (b - a) * t / 65536 (floor), t in [0, 65536]
constexpr int64_t lerpQ16(int64_t a, int64_t b, int64_t t)
{
    SAIGA_ASSERT(t >= 0 && t <= 65536, "lerpQ16: t in [0, 65536]");
    return a + floorShift((b - a) * t, 16);
}

// The quintic fade 6t^5 - 15t^4 + 10t^3 in Q16, t in [0, 65536], as a Horner scheme with Q32 intermediates:
//   r = (6t - 15 * 2^16) * t + 10 * 2^32;  r = r * t / 2^16;  r = r * t / 2^16;  result = r * t / 2^32  (floor shifts)
// Every product is below 2^53. Exact endpoints (0 and 65536), monotone, every value in [0, 65536] and within 1 LSB of
// the exact polynomial (all tested over every input). The formula is fixed: it enters the committed content hashes.
constexpr int64_t smoothQ16(int64_t t)
{
    SAIGA_ASSERT(t >= 0 && t <= 65536, "smoothQ16: t in [0, 65536]");
    int64_t r = (6 * t - 15 * 65536) * t + 10 * (int64_t(1) << 32);
    r         = floorShift(r * t, 16);
    r         = floorShift(r * t, 16);
    return floorShift(r * t, 32);
}

// The exact floor of the square root, bitwise (digit by digit)
constexpr uint64_t isqrt64(uint64_t v)
{
    uint64_t result = 0;
    uint64_t bit    = uint64_t(1) << 62;
    while (bit > v) bit >>= 2;
    while (bit != 0)
    {
        if (v >= result + bit)
        {
            v -= result + bit;
            result = (result >> 1) + bit;
        }
        else
            result >>= 1;
        bit >>= 2;
    }
    return result;
}

template <class T>
constexpr T clampI(T v, T lo, T hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

}  // namespace Engine
}  // namespace Saiga
