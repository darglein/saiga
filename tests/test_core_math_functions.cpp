/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/math/Polynomial4.h"
#include "saiga/core/math/all.h"
#include "saiga/core/math/half.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

using namespace Saiga;

TEST(MathFunctions, AlignUpDown)
{
    EXPECT_EQ(iAlignUp(10, 4), 12);
    EXPECT_EQ(iAlignUp(12, 4), 12);
    EXPECT_EQ(iAlignUp(0, 4), 0);
    EXPECT_EQ(iAlignUp(-3, 4), 0);
    EXPECT_EQ(iAlignUp(-4, 4), -4);
    EXPECT_EQ(iAlignUp(-5, 4), -4);
    EXPECT_EQ(iAlignUp(10u, 4u), 12u);

    EXPECT_EQ(iAlignDown(10, 4), 8);
    EXPECT_EQ(iAlignDown(12, 4), 12);
    EXPECT_EQ(iAlignDown(-3, 4), -4);
    EXPECT_EQ(iAlignDown(-4, 4), -4);
    EXPECT_EQ(iAlignDown(-5, 4), -8);
    EXPECT_EQ(iAlignDown(10u, 4u), 8u);
}

TEST(MathFunctions, CeilDouble)
{
    // 2 + 1e-9 is exactly 2 in float precision.
    EXPECT_EQ(iCeil(2.0 + 1e-9), 3);
    EXPECT_EQ(iCeil(2.0), 2);
    EXPECT_EQ(iCeil(1.5f), 2);
    EXPECT_EQ(iCeil(-1.5f), -1);
}

TEST(MathFunctions, OnbIsRotation)
{
    for (vec3 n : {vec3(0, 0, 1), vec3(0, 0, -1), vec3(1, 2, 3).normalized(), vec3(-3, 1, -2).normalized()})
    {
        mat3 R = onb(n);
        EXPECT_NEAR((R.col(2) - n).norm(), 0, 1e-6);
        EXPECT_NEAR(R.col(0).dot(R.col(1)), 0, 1e-6);
        EXPECT_NEAR(R.col(0).norm(), 1, 1e-6);
        EXPECT_NEAR(R.col(1).norm(), 1, 1e-6);
        // Right handed: x cross y = z
        EXPECT_NEAR((R.col(0).cross(R.col(1)) - R.col(2)).norm(), 0, 1e-6);
    }
}

TEST(MathFunctions, Sinc)
{
    EXPECT_FLOAT_EQ(sinc(0.5f), std::sin(0.5f) / 0.5f);
    EXPECT_DOUBLE_EQ(sinc(0.25), std::sin(0.25) / 0.25);
    EXPECT_EQ(sinc(0.f), 1.f);
}

#ifndef SAIGA_USE_EIGEN
TEST(MathFunctions, ArrayMult)
{
    EXPECT_EQ(ArrayMult(vec3(1, 2, 3), vec3(4, 5, 6)), vec3(4, 10, 18));
}
#endif

TEST(MathFunctions, MatrixUniformInteger)
{
    Random::setSeed(3);
    bool seen_low = false, seen_high = false;
    for (int i = 0; i < 2000; ++i)
    {
        ucvec3 v = Random::MatrixUniform<ucvec3>(0, 255);
        ivec3 w  = Random::MatrixUniform<ivec3>(-1, 1);
        for (int j = 0; j < 3; ++j)
        {
            seen_high |= v(j) == 255;
            seen_low |= w(j) == -1;
            EXPECT_GE(w(j), -1);
            EXPECT_LE(w(j), 1);
        }
    }
    EXPECT_TRUE(seen_high);
    EXPECT_TRUE(seen_low);
}

TEST(MathFunctions, SphericalRandPrecision)
{
    for (int i = 0; i < 100; ++i)
    {
        EXPECT_NEAR(Random::sphericalRand(2.0).norm(), 2.0, 1e-12);
    }
}

TEST(MathFunctions, HalfRoundtrip)
{
    const half h(1.5f);
    EXPECT_EQ(float(h), 1.5f);
    EXPECT_EQ(float(half(-0.25f)), -0.25f);
}

static float FloatFromBits(uint32_t b)
{
    float f;
    std::memcpy(&f, &b, sizeof(f));
    return f;
}

// Regression test: float -> half produced NaN (0x7c01) for values slightly above 65536, inf for NaNs with a small
// payload, the smallest denormal instead of zero for tiny floats (a shift by >= 32 bits), 2^-15 instead of 2^-14 for
// floats just below 2^-14 and rounded ties away from zero instead of to even.
TEST(MathFunctions, HalfFromFloatEdgeCases)
{
    EXPECT_EQ(half(0.f).h, 0x0000);
    EXPECT_EQ(half(-0.f).h, 0x8000);
    EXPECT_EQ(half(65504.f).h, 0x7bff);
    EXPECT_EQ(half(65519.99f).h, 0x7bff);
    EXPECT_EQ(half(65520.f).h, 0x7c00);
    EXPECT_EQ(half(65568.f).h, 0x7c00);
    EXPECT_EQ(half(-1e10f).h, 0xfc00);
    EXPECT_EQ(half(std::numeric_limits<float>::infinity()).h, 0x7c00);
    EXPECT_EQ(half(-std::numeric_limits<float>::infinity()).h, 0xfc00);
    EXPECT_TRUE(std::isnan(float(half(FloatFromBits(0x7f800001)))));
    EXPECT_TRUE(std::isnan(float(half(std::numeric_limits<float>::quiet_NaN()))));

    EXPECT_EQ(half(1e-37f).h, 0x0000);
    EXPECT_EQ(half(-1e-37f).h, 0x8000);
    EXPECT_EQ(half(std::numeric_limits<float>::denorm_min()).h, 0x0000);
    EXPECT_EQ(half(FloatFromBits(0x387fffff)).h, 0x0400);  // just below 2^-14
    EXPECT_EQ(half(std::ldexp(1.f, -14)).h, 0x0400);
    EXPECT_EQ(half(std::ldexp(1.f, -24)).h, 0x0001);
    EXPECT_EQ(half(std::ldexp(1.f, -25)).h, 0x0000);                       // tie -> even (0)
    EXPECT_EQ(half(std::nextafter(std::ldexp(1.f, -25), 1.f)).h, 0x0001);  // above the tie
    EXPECT_EQ(half(std::ldexp(3.f, -25)).h, 0x0002);                       // tie -> even (2)

    EXPECT_EQ(half(1.f + std::ldexp(1.f, -11)).h, 0x3c00);  // tie -> even
    EXPECT_EQ(half(1.f + std::ldexp(3.f, -11)).h, 0x3c02);  // tie -> even
    EXPECT_EQ(half(2047.f + 0.5f).h, half(2048.f).h);
}

TEST(MathFunctions, HalfAllValues)
{
    for (uint32_t i = 0; i < 0x10000; ++i)
    {
        const uint16_t bits = i;
        const float f       = half(bits);
        const bool is_nan   = (bits & 0x7c00) == 0x7c00 && (bits & 0x03ff) != 0;
        EXPECT_EQ(std::isnan(f), is_nan) << i;
        if (is_nan) continue;
        // Every half is exactly representable as float and converts back to itself.
        EXPECT_EQ(half(f).h, bits) << i;
        EXPECT_EQ(std::signbit(f), (bits & 0x8000) != 0) << i;
    }

    // The midpoint between two neighbouring finite halves is exact in float and rounds to the even one.
    for (uint32_t i = 0; i < 0x7bff; ++i)
    {
        const float lo  = half(uint16_t(i));
        const float hi  = half(uint16_t(i + 1));
        const float mid = (lo + hi) / 2;
        EXPECT_EQ(half(mid).h, (i & 1) ? i + 1 : i) << i;
        EXPECT_EQ(half(std::nextafter(mid, 0.f)).h, i) << i;
        EXPECT_EQ(half(std::nextafter(mid, 1e6f)).h, i + 1) << i;
        EXPECT_EQ(half(-mid).h, ((i & 1) ? i + 1 : i) | 0x8000) << i;
    }
}

TEST(MathFunctions, Round)
{
    EXPECT_EQ(iRound(0.49999997f), 0);
    EXPECT_EQ(iRound(0.5f), 1);
    EXPECT_EQ(iRound(2.5), 3);
    EXPECT_EQ(iRound(-2.5), -3);
    EXPECT_EQ(iRound(-0.49999997f), 0);
    EXPECT_EQ(iRound(-1.2), -1);
    EXPECT_EQ(iRound(7.0), 7);
}

TEST(MathFunctions, IsPow2)
{
    EXPECT_FALSE(ispow2(0));
    EXPECT_TRUE(ispow2(1));
    EXPECT_TRUE(ispow2(64));
    EXPECT_FALSE(ispow2(6));
}

TEST(MathFunctions, FromIniString)
{
    Mat2 m;
    fromIniString("0.1  0.2 0.30000000000000004 4 ", m);
    EXPECT_EQ(m(0, 0), 0.1);
    EXPECT_EQ(m(0, 1), 0.2);
    EXPECT_EQ(m(1, 0), 0.30000000000000004);
    EXPECT_EQ(m(1, 1), 4);

    Mat2 m2;
    fromIniString(toIniString(m), m2);
    EXPECT_EQ(m, m2);

    EXPECT_THROW(fromIniString("1 2 3", m), std::invalid_argument);
}

TEST(MathFunctions, RandomQuatIsUniform)
{
    // For uniformly distributed rotations with w >= 0, the mean of w is 4 / (3 pi).
    Random::setSeed(5);
    const int n = 100000;
    double sum  = 0;
    for (int i = 0; i < n; ++i)
    {
        auto q = Random::randomQuat<double>();
        EXPECT_NEAR(q.norm(), 1, 1e-12);
        EXPECT_GE(q.w(), 0);
        sum += q.w();
    }
    EXPECT_NEAR(sum / n, 4.0 / (3.0 * pi<double>()), 0.003);
}

TEST(MathFunctions, IniStringRoundtrip)
{
    Matrix<float, 2, 2> f;
    f(0, 0) = 0.1f;
    f(0, 1) = 1.f / 3.f;
    f(1, 0) = -12345.678f;
    f(1, 1) = 3e-20f;
    Matrix<float, 2, 2> f2;
    fromIniString(toIniString(f), f2);
    EXPECT_EQ(f, f2);

    ivec3 i(1, -2, 123456789);
    ivec3 i2;
    fromIniString(toIniString(i), i2);
    EXPECT_EQ(i, i2);
}

// Regression test: for Q == 0 the cubic solver used pow(2 * R, 1 / 3.0), which is NaN for negative R (x^3 + 1 = 0).
TEST(MathFunctions, SolveCubicNegativeCubeRoot)
{
    double x0 = 0, x1 = 0, x2 = 0;
    EXPECT_EQ(solve_deg3(1, 0, 0, 1, x0, x1, x2), 1);
    EXPECT_NEAR(x0, -1, 1e-12);

    EXPECT_EQ(solve_deg3(2, 0, 0, -16, x0, x1, x2), 1);
    EXPECT_NEAR(x0, 2, 1e-12);

    // (x + 1)^3 - 8 = 0 -> x = 1, shifted so that b != 0.
    EXPECT_EQ(solve_deg3(1, 3, 3, -7, x0, x1, x2), 1);
    EXPECT_NEAR(x0, 1, 1e-12);

    EXPECT_EQ(solve_deg3(1, 3, 3, 9, x0, x1, x2), 1);
    EXPECT_NEAR(x0, -3, 1e-12);
}

TEST(MathFunctions, SolvePolynomials)
{
    double x0 = 0, x1 = 0, x2 = 0, x3 = 0;

    EXPECT_EQ(solve_deg2(1, -3, 2, x0, x1), 2);
    EXPECT_NEAR(std::min(x0, x1), 1, 1e-12);
    EXPECT_NEAR(std::max(x0, x1), 2, 1e-12);
    EXPECT_EQ(solve_deg2(1, 0, 1, x0, x1), 0);

    // (x - 1)(x - 2)(x + 3) = x^3 - 7x + 6
    EXPECT_EQ(solve_deg3(1, 0, -7, 6, x0, x1, x2), 3);
    std::vector<double> r = {x0, x1, x2};
    std::sort(r.begin(), r.end());
    EXPECT_NEAR(r[0], -3, 1e-10);
    EXPECT_NEAR(r[1], 1, 1e-10);
    EXPECT_NEAR(r[2], 2, 1e-10);

    // Degenerate leading coefficients fall back to lower degrees.
    EXPECT_EQ(solve_deg3(0, 1, -3, 2, x0, x1, x2), 2);
    EXPECT_EQ(solve_deg3(0, 0, 2, -4, x0, x1, x2), 1);
    EXPECT_NEAR(x0, 2, 1e-12);

    // (x-1)(x-2)(x-3)(x-4)
    EXPECT_EQ(solve_deg4(1, -10, 35, -50, 24, x0, x1, x2, x3), 4);
    r = {x0, x1, x2, x3};
    std::sort(r.begin(), r.end());
    for (int i = 0; i < 4; ++i) EXPECT_NEAR(r[i], i + 1, 1e-8);

    // x^4 - 1 has the real roots -1 and 1.
    Matrix<double, 5, 1> coeffs;
    coeffs(0) = 1;
    coeffs(1) = 0;
    coeffs(2) = 0;
    coeffs(3) = 0;
    coeffs(4) = -1;
    Vec4 roots;
    EXPECT_EQ(solve_deg4(coeffs, roots), 2);
    EXPECT_NEAR(std::min(roots(0), roots(1)), -1, 1e-10);
    EXPECT_NEAR(std::max(roots(0), roots(1)), 1, 1e-10);

    // x^4 + 1 has no real roots.
    EXPECT_EQ(solve_deg4(1, 0, 0, 0, 1, x0, x1, x2, x3), 0);
}
