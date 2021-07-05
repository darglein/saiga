/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/kernels/BA.h"

#include "ceres/autodiff_cost_function.h"

namespace Saiga
{
struct CostBARSMono
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Helper function to simplify the "add residual" part for creating ceres problems
    using CostType = CostBARSMono;
    // Note: The first number is the number of residuals
    //       The following number sthe size of the residual blocks (without local parametrization)
    using CostFunctionType = ceres::AutoDiffCostFunction<CostType, 2, 7, 7, 3>;
    template <typename... Types>
    static CostFunctionType* create(const Types&... args)
    {
        return new CostFunctionType(new CostType(args...));
    }

    template <typename T>
    bool operator()(const T* const _extrinsics1, const T* const _extrinsics2, const T* const _worldPoint,
                    T* _residuals) const
    {
        using SE3  = Sophus::SE3<T>;
        using Vec2 = Eigen::Matrix<T, 2, 1>;
        using Vec3 = Eigen::Matrix<T, 3, 1>;
        using Vec6 = Eigen::Matrix<T, 6, 1>;

        Eigen::Map<SE3 const> const start(_extrinsics1);
        Eigen::Map<SE3 const> const end(_extrinsics2);
        Eigen::Map<Vec6 const> const velocity(_extrinsics2);

        Eigen::Map<const Vec3> wp(_worldPoint);
        Eigen::Map<Vec2> residual(_residuals);



        double h    = observed.y();
        double hrel = h / 480;


        SE3 vse = SE3::exp(velocity * hrel);

        //        SE3 se3 = slerp<T>(start,end,T(hrel));
        SE3 se3 = vse * start;



        Vec3 pc = se3 * wp;
        T x     = pc(0) / pc(2);
        T y     = pc(1) / pc(2);

        x = T(intr.fx) * x + T(intr.s) * y + T(intr.cx);
        y = T(intr.fy) * y + T(intr.cy);

        residual(0) = (T(observed(0)) - x) * T(weight);
        residual(1) = (T(observed(1)) - y) * T(weight);
        return true;
    }

    CostBARSMono(const IntrinsicsPinholed& intr, const Eigen::Vector2d& observed, double weight = 1)
        : intr(intr), observed(observed), weight(weight)
    {
    }

    IntrinsicsPinholed intr;
    Eigen::Vector2d observed;
    double weight;
};



}  // namespace Saiga
