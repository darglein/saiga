/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"

namespace Saiga
{
namespace Kernel
{
/**
 * Huber Loss function.
 * Similar to ceres::HuberLoss.
 * delta is the outlier threshold.
 *
 * Usage in Gauss Newton:
 *
 * Eigen::Matrix<double, 3, 6> Jrow;
 * Vec3 res ;
 * // Fill Jacobi and Residual
 * // ....
 *
 * auto rw = huberWeight(deltaStereo, res.squaredNorm());
 * Jrow *= rw;
 * res *= rw;
 *
 */
template <typename T>
inline T huberWeight(T _delta, T e)
{
    T dsqr = _delta * _delta;
    if (e <= dsqr)
    {
        // inlier
        return 1;
    }
    else
    {
        // outlier
        T sqrte = sqrt(e);      // absolut value of the error
        return _delta / sqrte;  // rho'(e)  = delta / sqrt(e)
    }
}



struct IdentityRobustification
{
    template <typename ResidualType, typename JacobiType>
    void operator()(ResidualType&, JacobiType&) const
    {
    }
};

template <typename T>
struct HuberRobustification
{
    T delta;
    HuberRobustification(T _delta = 1) : delta(_delta) {}

    template <typename ResidualType, typename JacobiType>
    void apply(ResidualType& res, JacobiType& Jrow) const
    {
        auto e = res.squaredNorm();
        auto w = huberWeight(delta, e);
        res *= w;
        Jrow *= w;
    }
};


}  // namespace Kernel
}  // namespace Saiga
