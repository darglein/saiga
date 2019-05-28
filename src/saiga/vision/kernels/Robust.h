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
 * auto sqrtLoss = sqrt(rw(1));
 * Jrow *= sqrtLoss;
 * res *= sqrtLoss;
 *
 * // Build quadratic form...
 * // Note: If you multiply the loss directly to the quadractic form
 * // you can save the sqrt(rw(1));
 *
 *
 */
template <typename T>
inline Eigen::Matrix<T, 3, 1> huberWeight(T _deltaChi1, T residualSquared)
{
    Eigen::Matrix<T, 3, 1> result;
    T thresholdChi2 = _deltaChi1 * _deltaChi1;
    if (residualSquared <= thresholdChi2)
    {
        // inlier
        result(0) = residualSquared;
        result(1) = 1;
        result(2) = 0;
        return result;
    }
    else
    {
        // outlier
        T sqrte   = sqrt(residualSquared);  // absolut value of the error
        result(0) = 2 * sqrte * _deltaChi1 - thresholdChi2;
        result(1) = _deltaChi1 / sqrte;
        result(2) = -0.5 * result(1) / residualSquared;
        return result;  // rho'(e)  = delta / sqrt(e)
    }
}


#if 1
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
        auto w = huberWeight(delta, e)(1);
        res *= w;
        Jrow *= w;
    }
};
#endif

}  // namespace Kernel
}  // namespace Saiga
