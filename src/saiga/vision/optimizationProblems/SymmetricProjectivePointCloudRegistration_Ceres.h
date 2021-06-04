/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/vision/ceres/CeresHelper.h"
#include "saiga/vision/ceres/local_parameterization_se3.h"
#include "saiga/vision/ceres/local_parameterization_sim3.h"

#include "SymmetricProjectivePointCloudRegistration.h"

#include "ceres/autodiff_cost_function.h"
#include "ceres/loss_function.h"
namespace Saiga
{
template <typename ScalarType, bool invert = false>
struct SymmetricProjectivePointCloudRegistrationCeresCostSE3
{
    // Helper function to simplify the "add residual" part for creating ceres problems
    using CostType = SymmetricProjectivePointCloudRegistrationCeresCostSE3;
    // Note: The first number is the number of residuals
    //       The following number sthe size of the residual blocks (without local parametrization)
    using CostFunctionType = ceres::AutoDiffCostFunction<CostType, 2, 7>;
    template <typename... Types>
    static CostFunctionType* create(Types... args)
    {
        return new CostFunctionType(new CostType(args...));
    }

    template <typename T>
    bool operator()(const T* const _extrinsics, T* _residuals) const
    {
        Eigen::Map<Sophus::SE3<T> const> const se3(_extrinsics);
        Eigen::Map<Eigen::Matrix<T, 2, 1>> residual(_residuals);

        Eigen::Matrix<T, 3, 1> wp       = _wp.cast<T>();
        auto intr                       = _intr.cast<T>();
        Eigen::Matrix<T, 2, 1> observed = _observed.cast<T>();
        Eigen::Matrix<T, 3, 1> pc;
        if (invert)
            pc = se3.inverse() * wp;
        else
            pc = se3 * wp;
        Eigen::Matrix<T, 2, 1> proj = intr.project(pc);
        residual                    = T(weight) * (observed - proj);
        return true;
    }

    SymmetricProjectivePointCloudRegistrationCeresCostSE3(IntrinsicsPinholed intr, Vec2 observed, Vec3 wp, double weight = 1)
        : _intr(intr), _observed(observed), _wp(wp), weight(weight)
    {
    }

    IntrinsicsPinholed _intr;
    Vec2 _observed;
    Vec3 _wp;
    double weight;
};

template <typename ScalarType, bool invert = false>
struct SymmetricProjectivePointCloudRegistrationCeresCostDSim3
{
    // Helper function to simplify the "add residual" part for creating ceres problems
    using CostType = SymmetricProjectivePointCloudRegistrationCeresCostDSim3;
    // Note: The first number is the number of residuals
    //       The following number sthe size of the residual blocks (without local parametrization)
    using CostFunctionType = ceres::AutoDiffCostFunction<CostType, 2, 7>;
    template <typename... Types>
    static CostFunctionType* create(Types... args)
    {
        return new CostFunctionType(new CostType(args...));
    }

    template <typename T>
    bool operator()(const T* const _extrinsics, T* _residuals) const
    {
        //        Eigen::Map<Sophus::DSim3 const> const se3(_extrinsics);
        const Sophus::DSim3<T>& se3 = ((const Sophus::DSim3<T>*)(_extrinsics))[0];

        Eigen::Map<Eigen::Matrix<T, 2, 1>> residual(_residuals);

        Eigen::Matrix<T, 3, 1> wp       = _wp.cast<T>();
        auto intr                       = _intr.cast<T>();
        Eigen::Matrix<T, 2, 1> observed = _observed.cast<T>();
        Eigen::Matrix<T, 3, 1> pc;
        if (invert)
            pc = se3.inverse() * wp;
        else
            pc = se3 * wp;
        Eigen::Matrix<T, 2, 1> proj = intr.project(pc);
        residual                    = T(weight) * (observed - proj);
        return true;
    }

    SymmetricProjectivePointCloudRegistrationCeresCostDSim3(IntrinsicsPinholed intr, Vec2 observed, Vec3 wp, double weight = 1)
        : _intr(intr), _observed(observed), _wp(wp), weight(weight)
    {
    }

    IntrinsicsPinholed _intr;
    Vec2 _observed;
    Vec3 _wp;
    double weight;
};


template <bool FIX_SCALE, typename ScalarType, typename TransformationType>
OptimizationResults optimize_PPCR_ceres(SymmetricProjectivePointCloudRegistration<TransformationType>& scene,
                                        const ceres::Solver::Options& ceres_options = ceres::Solver::Options())
{
    using Trans = TransformationType;
    ceres::Problem::Options problemOptions;
    ceres::Problem problem(problemOptions);

    double huber = sqrt(scene.chi2Threshold);

    ceres::LocalParameterization* camera_parameterization;

    if constexpr (Trans::DoF == 7)
    {
        camera_parameterization = new Saiga::test::LocalParameterizationSim3<FIX_SCALE>;
    }
    else
    {
        camera_parameterization = new Sophus::test::LocalParameterizationSE3;
    }

    problem.AddParameterBlock(scene.T.data(), 7, camera_parameterization);



    for (auto& e : scene.obs1)
    {
        if (e.wp == -1) continue;
        auto& wp = scene.points2[e.wp];

        ceres::CostFunction* cost;
        if constexpr (Trans::DoF == 7)
        {
            cost = SymmetricProjectivePointCloudRegistrationCeresCostDSim3<ScalarType, true>::create(
                scene.K, e.imagePoint, wp, e.weight);
        }
        else
        {
            cost = SymmetricProjectivePointCloudRegistrationCeresCostSE3<ScalarType, true>::create(
                scene.K, e.imagePoint, wp, e.weight);
        }
        ceres::LossFunction* loss = nullptr;
        if (huber > 0) loss = new ceres::HuberLoss(huber);
        problem.AddResidualBlock(cost, loss, scene.T.data());
    }

    for (auto& e : scene.obs2)
    {
        if (e.wp == -1) continue;
        auto& wp = scene.points1[e.wp];
        ceres::CostFunction* cost;
        if constexpr (Trans::DoF == 7)
        {
            cost = SymmetricProjectivePointCloudRegistrationCeresCostDSim3<ScalarType, false>::create(
                scene.K, e.imagePoint, wp, e.weight);
        }
        else
        {
            cost = SymmetricProjectivePointCloudRegistrationCeresCostSE3<ScalarType, false>::create(
                scene.K, e.imagePoint, wp, e.weight);
        }
        ceres::LossFunction* loss = nullptr;
        if (huber > 0) loss = new ceres::HuberLoss(huber);
        problem.AddResidualBlock(cost, loss, scene.T.data());
    }
    return ceres_solve(ceres_options, problem);
}

}  // namespace Saiga
