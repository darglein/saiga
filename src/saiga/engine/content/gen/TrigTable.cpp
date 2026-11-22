/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/content/gen/TrigTable.h"

#include "saiga/engine/content/gen/FixedMath.h"

namespace Saiga
{
namespace Engine
{
namespace
{
// pi / 2 in Q45 (rounded to nearest): x = i * (pi / 2) / 2^15 in Q30 is i * kHalfPiQ45 / 2^30
constexpr int64_t kHalfPiQ45 = 55267482437722;

// sin(x), x in Q30 within [0, pi / 2], by the Taylor series of degree 15 in the Horner form
//   sin x = x * (1 - x^2/(2*3) * (1 - x^2/(4*5) * (... * (1 - x^2/(14*15)))))
// with floor shifts. Error bound: the truncation after degree 15 is below (pi/2)^17 / 17! < 2^-37 (far below 1 LSB of
// Q30). Every step makes at most one floor error of 1 LSB (x^2, the seven Horner steps, the final product), and every
// later step scales the earlier errors by x^2 / (2k (2k + 1)) < 0.42; the rounding of x itself (0.5 LSB) passes through
// with a factor cos(x) <= 1. So the total stays below 4 LSB (tested against libm with an 8 LSB margin).
int32_t seriesSin(int64_t x)
{
    const int64_t x2 = mulShift(x, x, 30);  // below 2.65 * 10^9
    int64_t r        = kTrigOne;
    for (int64_t k = 7; k >= 1; --k) r = kTrigOne - floorDiv(mulShift(x2, r, 30), (2 * k) * (2 * k + 1));
    return int32_t(clampI<int64_t>(mulShift(x, r, 30), 0, kTrigOne));
}

std::vector<int32_t> buildTable()
{
    std::vector<int32_t> t(kTrigTableSize);
    for (uint32_t i = 0; i < kTrigTableSize; ++i)
    {
        // x * 2^30 = i * (pi / 2) * 2^15 = i * kHalfPiQ45 / 2^30, rounded to nearest (i * kHalfPiQ45 < 2^15 * 2^46)
        const int64_t x = floorShift(int64_t(i) * kHalfPiQ45 + (int64_t(1) << 29), 30);
        t[i]            = seriesSin(x);
    }
    return t;
}
}  // namespace

const std::vector<int32_t>& trigQuarterTable()
{
    static const std::vector<int32_t> table = buildTable();
    return table;
}

int32_t sinQ30(uint32_t a17)
{
    const std::vector<int32_t>& t = trigQuarterTable();
    const uint32_t a              = a17 & (kTrigSteps - 1);
    const uint32_t quadrant       = a >> 15;
    const uint32_t i              = a & (kTrigQuarter - 1);
    switch (quadrant)
    {
        case 0:
            return t[i];
        case 1:
            return t[kTrigQuarter - i];
        case 2:
            return -t[i];
        default:
            return -t[kTrigQuarter - i];
    }
}

int32_t cosQ30(uint32_t a17)
{
    return sinQ30(a17 + kTrigQuarter);
}

}  // namespace Engine
}  // namespace Saiga
