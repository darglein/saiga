/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/image/all.h"
#include "saiga/vision/cameraModel/Distortion.h"
#include "saiga/vision/util/Random.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"
#include "numeric_derivative.h"
using namespace Saiga;

#ifndef WIN32
TEST(Distortion, Solve)
{
    IntrinsicsPinholed K(608.894, 608.742, 638.974, 364.492, 0);

    Vector<double, 8> c;
    c << -0.0284351, -2.47131, 1.7389, -0.145427, -2.26192, 1.63544, 0.00128128, -0.000454588;

    Distortion d(c);


    for (int x = 0; x < 1280; ++x)
    {
        for (int y = 0; y < 720; ++y)
        {
            Vec2 p_norm = K.unproject2(Vec2(x, y));
            Vec2 p2     = undistortPointGN(p_norm, p_norm, d, 10);
            double e2   = (distortNormalizedPoint(p2, d) - p_norm).norm();
            EXPECT_LE(e2, 1e-3);
        }
    }
}

TEST(NumericDerivative, DistortionPoint)
{
    Vector<double, 8> c;
    c.setRandom();
    Distortion d(c);

    Vec2 p = Vec2::Random();
    Matrix<double, 2, 2> J1, J2;

    Vec2 res1 = distortNormalizedPoint(p, d, &J1);
    Vec2 res2 = EvaluateNumeric([&](auto p) { return distortNormalizedPoint(p, d); }, p, &J2, 1e-8);

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J1, J2, 1e-5);
}

TEST(NumericDerivative, DistortionDist)
{
    Vector<double, 8> dist_8;
    dist_8.setRandom();
    dist_8 *= 0.1;

    Vec2 p = Vec2::Random();
    Matrix<double, 2, 8> J1, J2;

    Vec2 res1 = distortNormalizedPoint<double>(p, Distortion(dist_8), nullptr, &J1);
    Vec2 res2 = EvaluateNumeric([&](auto d) { return distortNormalizedPoint(p, Distortion(d)); }, dist_8, &J2, 1e-8);

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J1, J2, 1e-5);
}
#endif