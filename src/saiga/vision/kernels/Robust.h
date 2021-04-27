/**
 * Copyright (c) 2021 Darius Rückert
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
inline Eigen::Matrix<T, 2, 1> HuberLoss(T _deltaChi1, T residualSquared)
{
    Eigen::Matrix<T, 2, 1> result;
    T thresholdChi2 = _deltaChi1 * _deltaChi1;
    if (residualSquared <= thresholdChi2)
    {
        // inlier
        result(0) = residualSquared;
        result(1) = 1;
        //        result(2) = 0;
        return result;
    }
    else
    {
        // outlier
        T sqrte   = sqrt(residualSquared);  // absolut value of the error
        result(0) = 2 * sqrte * _deltaChi1 - thresholdChi2;
        result(1) = std::max(std::numeric_limits<T>::min(), result(0) / residualSquared);
        // result(2) = -0.5 * result(1) / residualSquared;
        return result;
    }
}

// Squared loss near 0 until the threshold.
// After that log(x) to strongly reduce weight of outliers.
template <typename T>
inline Eigen::Matrix<T, 2, 1> CauchyLoss(T _deltaChi1, T residualSquared)
{
    Eigen::Matrix<T, 2, 1> result;

    auto b_ = _deltaChi1 * _deltaChi1;
    auto c_ = (1 / b_);

    const double sum = 1.0 + residualSquared * c_;

    // 'sum' and 'inv' are always positive, assuming that 's' is.
    result(0) = b_ * log(sum);
    result(1) = std::max(std::numeric_limits<double>::min(), result(0) / residualSquared);
    //    result(2) = -c_ * (inv * inv);
    return result;
}

template <typename T>
inline Eigen::Matrix<T, 2, 1> IdentityLoss(T _deltaChi1, T residualSquared)
{
    Eigen::Matrix<T, 2, 1> result;
    result(0) = residualSquared;
    result(1) = 1;
    return result;
}


enum class LossFunction
{
    Identity,
    Huber,
    Cauchy,
};


template <typename T>
inline Eigen::Matrix<T, 2, 1> Loss(LossFunction function, T _deltaChi1, T residualSquared)
{
    switch (function)
    {
        case LossFunction::Identity:
            return IdentityLoss(_deltaChi1, residualSquared);
        case LossFunction::Huber:
            return HuberLoss(_deltaChi1, residualSquared);
        case LossFunction::Cauchy:
            return CauchyLoss(_deltaChi1, residualSquared);
    }
    return Eigen::Matrix<T, 2, 1>::Zero();
}


}  // namespace Kernel
}  // namespace Saiga
