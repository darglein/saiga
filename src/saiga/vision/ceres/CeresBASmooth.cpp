/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "CeresBASmooth.h"

#include "saiga/core/time/timer.h"
#include "saiga/vision/ceres/CeresHelper.h"
#include "saiga/vision/ceres/CeresKernel_BARS_Intr4.h"
#include "saiga/vision/ceres/CeresKernel_BA_Intr4.h"
#include "saiga/vision/ceres/CeresKernel_SmoothBA.h"
#include "saiga/vision/ceres/local_parameterization_se3.h"
#include "saiga/vision/scene/Scene.h"

#include "ceres/ceres.h"
#include "ceres/problem.h"
#include "ceres/rotation.h"
#include "ceres/solver.h"

#define BA_AUTODIFF


namespace Saiga
{
OptimizationResults CeresBASmooth::initAndSolve()
{
    auto& scene = *_scene;

    SAIGA_OPTIONAL_BLOCK_TIMER(optimizationOptions.debugOutput);

    ceres::Problem::Options problemOptions;
    problemOptions.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problemOptions.cost_function_ownership          = ceres::DO_NOT_TAKE_OWNERSHIP;
    problemOptions.loss_function_ownership          = ceres::DO_NOT_TAKE_OWNERSHIP;

    ceres::Problem problem(problemOptions);


    Sophus::test::LocalParameterizationSE3 camera_parameterization;

    for (size_t i = 0; i < scene.extrinsics.size(); ++i)
    {
        problem.AddParameterBlock(scene.extrinsics[i].se3.data(), 7, &camera_parameterization);
        if (scene.extrinsics[i].constant) problem.SetParameterBlockConstant(scene.extrinsics[i].se3.data());
    }


    for (auto& wp : scene.worldPoints)
    {
        problem.AddParameterBlock(wp.p.data(), 3);
        problem.SetParameterBlockConstant(wp.p.data());
    }

    ceres::HuberLoss lossFunctionMono(baOptions.huberMono);
    ceres::HuberLoss lossFunctionStereo(baOptions.huberStereo);



    int monoCount   = 0;
    int stereoCount = 0;
    for (auto& img : scene.images)
    {
        for (auto& ip : img.stereoPoints)
        {
            if (!ip) continue;
            if (ip.depth > 0)
            {
                stereoCount++;
            }
            else
            {
                monoCount++;
            }
        }
    }



    std::vector<std::unique_ptr<CostBAMonoAnalytic>> monoCostFunctions;
    std::vector<std::unique_ptr<CostBAStereoAnalytic>> stereoCostFunctions;

    monoCostFunctions.reserve(monoCount);
    stereoCostFunctions.reserve(stereoCount);


    for (auto& img : scene.images)
    {
        auto& extr   = scene.extrinsics[img.extr].se3;
        auto& camera = scene.intrinsics[img.intr];
        for (auto& ip : img.stereoPoints)
        {
            if (!ip) continue;
            auto& wp = scene.worldPoints[ip.wp].p;
            double w = ip.weight * scene.scale();


            auto cost_function                = CostBAMono::create(camera, ip.point, w);
            ceres::LossFunction* lossFunction = baOptions.huberMono > 0 ? &lossFunctionMono : nullptr;
            problem.AddResidualBlock(cost_function, lossFunction, extr.data(), wp.data());
        }
    }

    for (auto& sc : scene.smoothnessConstraints)
    {
        auto& p1 = scene.extrinsics[scene.images[sc.ex1].extr].se3;
        auto& p2 = scene.extrinsics[scene.images[sc.ex2].extr].se3;
        auto& p3 = scene.extrinsics[scene.images[sc.ex3].extr].se3;
        auto w   = sc.weight;

        auto cost_function = CostSmoothBA::create(w);
        problem.AddResidualBlock(cost_function, nullptr, p1.data(), p2.data(), p3.data());
    }


    ceres::Solver::Options ceres_options = make_options(optimizationOptions);

    OptimizationResults result = ceres_solve(ceres_options, problem);



    return result;
}

}  // namespace Saiga
