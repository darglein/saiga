/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/core/math/all.h"

#include "gtest/gtest.h"

namespace Saiga
{
inline bool ExpectCloseRelative(double x, double y, double max_abs_relative_difference)
{
    double absolute_difference = fabs(x - y);
    double relative_difference = absolute_difference / std::max(fabs(x), fabs(y));
    if (x == 0 || y == 0)
    {
        // If x or y is exactly zero, then relative difference doesn't have any
        // meaning. Take the absolute difference instead.
        relative_difference = absolute_difference;
    }

    EXPECT_NEAR(relative_difference, 0.0, max_abs_relative_difference);
    return relative_difference <= max_abs_relative_difference;
}

inline bool ExpectClose(double x, double y, double max_difference)
{
    double absolute_difference = fabs(x - y);
    EXPECT_NEAR(x, y, max_difference);
    return absolute_difference <= max_difference;
}

template <typename Derived1, typename Derived2>
inline bool ExpectCloseRelative(const Eigen::DenseBase<Derived1>& a, const Eigen::DenseBase<Derived2>& b,
                                double max_abs_relative_difference, bool relative = true)
{
    EXPECT_EQ(a.rows(), b.rows());
    EXPECT_EQ(a.cols(), b.cols());

    if (a.rows() != b.rows() || a.cols() != b.cols())
    {
        return false;
    }

    bool found = false;
    for (int i = 0; i < a.rows(); ++i)
    {
        for (int j = 0; j < a.cols(); ++j)
        {
            auto x     = a(i, j);
            auto y     = b(i, j);
            auto d     = max_abs_relative_difference;
            bool close = relative ? ExpectCloseRelative(x, y, d) : ExpectClose(x, y, d);
            if (!close)
            {
                found = true;
                break;
            }
        }
        if (found) break;
    }
    if (found)
    {
        // Make it easier understandable
        std::cout << "Matrix ExpectCloseRelative failed." << std::endl;
        std::cout << "a: " << std::endl << a << std::endl << std::endl;
        std::cout << "b: " << std::endl << b << std::endl << std::endl;
    }

    return found;
}

}  // namespace Saiga
