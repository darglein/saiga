/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/kernels/BA.h"

#include "ceres/autodiff_cost_function.h"

namespace Saiga
{
struct CostBAMono
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Helper function to simplify the "add residual" part for creating ceres problems
    using CostType = CostBAMono;
    // Note: The first number is the number of residuals
    //       The following number sthe size of the residual blocks (without local parametrization)
    using CostFunctionType = ceres::AutoDiffCostFunction<CostType, 2, 7, 3>;
    template <typename... Types>
    static CostFunctionType* create(const Types&... args)
    {
        return new CostFunctionType(new CostType(args...));
    }

    template <typename T>
    bool operator()(const T* const _extrinsics, const T* const _worldPoint, T* _residuals) const
    {
        Eigen::Map<Sophus::SE3<T> const> const se3(_extrinsics);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> wp(_worldPoint);
        Eigen::Map<Eigen::Matrix<T, 2, 1>> residual(_residuals);

        Eigen::Matrix<T, 3, 1> pc = se3 * wp;


        T x = pc(0) / pc(2);
        T y = pc(1) / pc(2);

        x = T(intr.fx) * x + T(intr.s) * y + T(intr.cx);
        y = T(intr.fy) * y + T(intr.cy);

        residual(0) = (T(observed(0)) - x) * T(weight);
        residual(1) = (T(observed(1)) - y) * T(weight);
        return true;
    }

    CostBAMono(const IntrinsicsPinholed& intr, const Eigen::Vector2d& observed, double weight = 1)
        : intr(intr), observed(observed), weight(weight)
    {
    }

    IntrinsicsPinholed intr;
    Eigen::Vector2d observed;
    double weight;
};


template <bool INV_DEPTH_DIFF = false>
struct CostBAStereo
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Helper function to simplify the "add residual" part for creating ceres problems
    using CostType = CostBAStereo;
    // Note: The first number is the number of residuals
    //       The following number sthe size of the residual blocks (without local parametrization)
    using CostFunctionType = ceres::AutoDiffCostFunction<CostType, 3, 7, 3>;
    template <typename... Types>
    static CostFunctionType* create(const Types&... args)
    {
        return new CostFunctionType(new CostType(args...));
    }


    template <typename T>
    bool operator()(const T* const extrinsics, const T* const worldPoint, T* _residuals) const
    {
        Eigen::Map<Sophus::SE3<T> const> const se3(extrinsics);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> wp(worldPoint);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residual(_residuals);

        Eigen::Matrix<T, 3, 1> pc = se3 * wp;

        T z = pc(2);

        T x = pc(0) / z;
        T y = pc(1) / z;

        x = T(intr.fx) * x + T(intr.s) * y + T(intr.cx);
        y = T(intr.fy) * y + T(intr.cy);

        residual(0) = (T(observed(0)) - x);
        residual(1) = (T(observed(1)) - y);

        if (INV_DEPTH_DIFF)
        {
            T invz      = T(1) / z;
            T invzobs   = T(1) / T(stereoPoint);
            residual(2) = (invzobs - invz) * T(bf);
        }
        else
        {
            T disparity = x - T(bf) / z;
            residual(2) = (stereoPoint - disparity);
        }

        residual(0) *= T(weights(0));
        residual(1) *= T(weights(0));
        residual(2) *= T(weights(1));


        return true;
    }

    CostBAStereo(const IntrinsicsPinholed& intr, const Eigen::Vector2d& observed, double stereoPoint, double bf,
                 const Vec2& weights)
        : intr(intr), observed(observed), stereoPoint(stereoPoint), bf(bf), weights(weights)
    {
    }

    IntrinsicsPinholed intr;
    Eigen::Vector2d observed;
    double stereoPoint;
    double bf;
    Vec2 weights;
};
#if 0

class CostBAMonoAnalytic : public ceres::SizedCostFunction<2, 7, 3>
{
   public:
    using T = double;


    CostBAMonoAnalytic(const Intrinsics4& intr, const Eigen::Vector2d& observed, double weight = 1)
        : intr(intr), observed(observed), weight(weight)
    {
    }

    virtual ~CostBAMonoAnalytic() {}

    virtual bool Evaluate(double const* const* _parameters, double* _residuals, double** _jacobians) const
    {
        Eigen::Map<Sophus::SE3<T> const> const se3(_parameters[0]);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> wp(_parameters[1]);
        Eigen::Map<Eigen::Matrix<T, 2, 1>> residual(_residuals);



        if (!_jacobians)
        {
            // only compute residuals
//            residual = Kernel::evaluateResidual(intr, se3, wp, observed, weight);
        }
        else
        {
            // compute both

            Kernel::PoseJacobiType jpose;
            Kernel::PointJacobiType jpoint;
            Kernel::ResidualType res;
            Kernel::evaluateResidualAndJacobian(intr, se3, wp, observed, weight, res, jpose, jpoint);

            residual = res;

            if (_jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<T, 2, 7, Eigen::RowMajor>> jpose2(_jacobians[0]);
                jpose2.setZero();
                jpose2.block<2, 6>(0, 0) = jpose;
                //                std::cout << jpose2 << std::endl;
            }
            if (_jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<T, 2, 3, Eigen::RowMajor>> jpoint2(_jacobians[1]);
                jpoint2 = jpoint;
                //                std::cout << jpoint2 << std::endl;
            }

        }

        return true;
    }

   private:
    Intrinsics4 intr;
    Eigen::Vector2d observed;
    double weight;
};

class CostBAStereoAnalytic : public ceres::SizedCostFunction<3, 7, 3>
{
   public:
    using T = double;

    using Kernel = Saiga::Kernel::BAPosePointStereo<double>;

    CostBAStereoAnalytic(const Kernel::CameraType& intr, const Eigen::Vector2d& observed, double observedDepth,
                         double weight)
        : intr(intr), observed(observed), weight(weight), observedDepth(observedDepth)
    {
    }

    virtual ~CostBAStereoAnalytic() {}

    virtual bool Evaluate(double const* const* _parameters, double* _residuals, double** _jacobians) const
    {
        Eigen::Map<Sophus::SE3<T> const> const se3(_parameters[0]);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> wp(_parameters[1]);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residual(_residuals);



        if (!_jacobians)
        {
            // only compute residuals
            residual = Kernel::evaluateResidual(intr, se3, wp, observed, observedDepth, weight, weight);
        }
        else
        {
            // compute both
            Kernel::PoseJacobiType jpose;
            Kernel::PointJacobiType jpoint;
            Kernel::ResidualType res;
            Kernel::evaluateResidualAndJacobian(intr, se3, wp, observed, observedDepth, weight, weight, res, jpose,
                                                jpoint);

            residual = res;

            if (_jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<T, 3, 7, Eigen::RowMajor>> jpose2(_jacobians[0]);
                jpose2.setZero();
                jpose2.block<3, 6>(0, 0) = jpose;
                //                std::cout << jpose2 << std::endl;
            }
            if (_jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<T, 3, 3, Eigen::RowMajor>> jpoint2(_jacobians[1]);
                jpoint2 = jpoint;
                //                std::cout << jpoint2 << std::endl;
            }
        }

        return true;
    }

   private:
    Kernel::CameraType intr;
    Eigen::Vector2d observed;
    double weight, observedDepth;
};
#endif
}  // namespace Saiga
