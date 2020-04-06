/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/core/math/all.h"
#include "gtest/gtest.h"

namespace Saiga{


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

template <typename Derived>
inline bool ExpectCloseRelative(const Eigen::DenseBase<Derived>& a, const Eigen::DenseBase<Derived>& b, double max_abs_relative_difference)
{
    bool found = false;
    for(int i =0;i < a.rows(); ++i)
    {
        for(int j =0;j < a.cols(); ++j)
        {
            if(!ExpectCloseRelative(a(i,j),b(i,j),max_abs_relative_difference)){
                found = true;
                break;
            }
        }
        if(found) break;
    }
    if(found){
        // Make it easier understandable
        std::cout << "Matrix ExpectCloseRelative failed." << std::endl;
        std::cout << "a: " << std::endl << a << std::endl << std::endl;
        std::cout << "b: " << std::endl <<b << std::endl << std::endl;
    }

    return found;
}

}
