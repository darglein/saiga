/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"

#include "ceres/autodiff_cost_function.h"

namespace Saiga
{
/**
 * Ceres kernels for the BAL projection function.
 *
 * https://grail.cs.washington.edu/projects/bal/
 *
 * See also BALDataset.h
 */
struct CostBALDistortion
{
    // Helper function to simplify the "add residual" part for creating ceres problems
    using CostType = CostBALDistortion;
    // Note: The first number is the number of residuals
    //       The following number sthe size of the residual blocks (without local parametrization)
    using CostFunctionType = ceres::AutoDiffCostFunction<CostType, 2, 2>;
    template <typename... Types>
    static CostFunctionType* create(Types... args)
    {
        return new CostFunctionType(new CostType(args...));
    }

    template <typename T>
    bool operator()(const T* const _undistortedPoint, T* _residuals) const
    {
        Eigen::Map<const Eigen::Matrix<T, 2, 1>> up(_undistortedPoint);
        Eigen::Map<Eigen::Matrix<T, 2, 1>> residual(_residuals);
        Eigen::Matrix<T, 2, 1> dp = distortedPoint.cast<T>();

        T r2                         = up(0) * up(0) + up(1) * up(1);
        T d                          = T(1.0) + T(k1) * r2 + T(k2) * r2 * r2;
        Eigen::Matrix<T, 2, 1> guess = d * up;
        residual                     = dp - guess;

        return true;
    }

    CostBALDistortion(Vec2 distortedPoint, double k1, double k2) : distortedPoint(distortedPoint), k1(k1), k2(k2) {}

    Vec2 distortedPoint;
    double k1, k2;
};


}  // namespace Saiga
