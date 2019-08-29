/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/kernels/BAPosePoint.h"

#include "ceres/autodiff_cost_function.h"

namespace Saiga
{
struct CostSmoothBA
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Helper function to simplify the "add residual" part for creating ceres problems
    using CostType = CostSmoothBA;
    // Note: The first number is the number of residuals
    //       The following number sthe size of the residual blocks (without local parametrization)
    using CostFunctionType = ceres::AutoDiffCostFunction<CostType, 6, 7,7, 7>;
    template <typename... Types>
    static CostFunctionType* create(const Types&... args)
    {
        return new CostFunctionType(new CostType(args...));
    }

    template <typename T>
    bool operator()(const T* const _extrinsics1, const T* const _extrinsics2, const T* const _extrinsics3, T* _residuals) const
    {
        using SE3 = Sophus::SE3<T>;
        using Vec6 = Eigen::Matrix<T, 6, 1>;

        Eigen::Map<SE3 const> const p1(_extrinsics1);
        Eigen::Map<SE3 const> const p2(_extrinsics2);
        Eigen::Map<SE3 const> const p3(_extrinsics3);
        Eigen::Map<Vec6> residual(_residuals);


//        {
//            auto t1 = p1.inverse().translation();
//            auto t2 = p2.inverse().translation();
//            auto t3 = p3.inverse().translation();

//            residual.setZero();

//            residual.template segment<3>(0) = (t2-t1) - (t3-t2);
//        }
        auto rel12 = (p2 *p1.inverse());
        auto rel23 = (p3 *p2.inverse());

//        auto rel12 = (p1.inverse() * p2);
//        auto rel23 = (p2.inverse() * p3);


//        Vec6 t = (rel23 *rel12.inverse()).log();
//        residual = t;
        Vec6 v1 = rel12.log();
        Vec6 v2 = rel23.log();
        residual = weight * (v1 - v2);

        return true;
    }

    CostSmoothBA( double weight = 1)
        : weight(weight)
    {
    }

    double weight;
};



}  // namespace Saiga
