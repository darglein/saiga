/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vision/kernels/Robust.h"
#include "saiga/vision/util/Random.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"
#include "numeric_derivative.h"
namespace Saiga
{
TEST(Robust, Huber)
{
    double threshold = 10;


    for (double t = 0; t <= 20; t += 0.1)
    {
        double res_2 = t * t;
        auto rw      = Kernel::HuberLoss<double>(threshold, res_2);

        double test_t     = t * sqrt(rw(1));
        double test_res_2 = test_t * test_t;

        double test_res_2_direct = rw(1) * res_2;

        EXPECT_NEAR(rw(0), test_res_2, 1e-10);
        EXPECT_NEAR(rw(0), test_res_2_direct, 1e-10);
    }
}

TEST(Robust, Cauchy)
{
    double threshold = 10;


    for (double t = 0; t <= 20; t += 0.1)
    {
        double res_2 = t * t;
        auto rw      = Kernel::CauchyLoss<double>(threshold, res_2);

        double test_t     = t * sqrt(rw(1));
        double test_res_2 = test_t * test_t;

        double test_res_2_direct = rw(1) * res_2;


        EXPECT_NEAR(rw(0), test_res_2, 1e-10);
        EXPECT_NEAR(rw(0), test_res_2_direct, 1e-10);
    }
}

}  // namespace Saiga
