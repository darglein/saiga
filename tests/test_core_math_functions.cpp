/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/math/all.h"
#include "saiga/core/math/half.h"

#include "gtest/gtest.h"

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
