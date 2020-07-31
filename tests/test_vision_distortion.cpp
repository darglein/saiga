/**
 * Copyright (c) 2017 Darius Rückert
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


TEST(Distortion, Derivative)
{
    Vector<double, 8> c;
    c << -0.283408, 0.5, -1, 0.2, 0.4, 0.7, 1, 2;
    //    c.setZero();
    Distortion d(c);


    Vec2 ref = Vec2::Random();
    Vec2 p   = Vec2::Random();
    Matrix<double, 2, 2> J1, J2;

    Vec2 res1 = distortNormalizedPoint(p, d, &J1);
    Vec2 res2 = EvaluateNumeric([&](auto p) { return distortNormalizedPoint(p, d); }, p, &J2, 1e-8);

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J1, J2, 1e-5);
}

TEST(Distortion, Solve)
{
    Intrinsics4 K(608.894, 608.742, 638.974, 364.492);

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
