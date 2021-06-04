/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "CeresBARS.h"

#include "saiga/core/time/timer.h"
#include "saiga/vision/ceres/CeresHelper.h"
#include "saiga/vision/ceres/CeresKernel_BARS_Intr4.h"
#include "saiga/vision/ceres/CeresKernel_BA_Intr4.h"
#include "saiga/vision/ceres/local_parameterization_se3.h"
#include "saiga/vision/scene/Scene.h"

#include "ceres/ceres.h"
#include "ceres/problem.h"
#include "ceres/rotation.h"
#include "ceres/solver.h"

#define BA_AUTODIFF


namespace Saiga
{
OptimizationResults CeresBARS::initAndSolve()
{
    auto& scene = *_scene;

    SAIGA_OPTIONAL_BLOCK_TIMER(optimizationOptions.debugOutput);

    ceres::Problem::Options problemOptions;
    problemOptions.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problemOptions.cost_function_ownership          = ceres::DO_NOT_TAKE_OWNERSHIP;
    problemOptions.loss_function_ownership          = ceres::DO_NOT_TAKE_OWNERSHIP;

    ceres::Problem problem(problemOptions);


    Sophus::test::LocalParameterizationSE3 camera_parameterization;

    for (size_t i = 0; i < scene.images.size(); ++i)
    {
        problem.AddParameterBlock(scene.images[i].se3.data(), 7, &camera_parameterization);
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

    auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();


    //    std::vector<std::unique_ptr<CostBAMonoAnalytic>> monoCostFunctions;
    //    std::vector<std::unique_ptr<CostBAStereoAnalytic>> stereoCostFunctions;

    //    monoCostFunctions.reserve(monoCount);
    //    stereoCostFunctions.reserve(stereoCount);


    std::vector<std::array<double, 2 * 8>> mixedData;


    for (auto& img : scene.images)
    {
        auto& extr = img.se3;


        std::array<double, 2 * 8> data;


        Eigen::Map<Sophus::SE3<double>> s1(data.data());

        Eigen::Map<Vec6> s2(data.data() + 8);
        //        Eigen::Map<Sophus::SE3<double>> s2(data.data() + 8);


        s1 = extr;
        s2.setZero();
        //        s2 = extr;
        mixedData.push_back(data);
    }

    for (auto& img : scene.images)
    {
        //        auto& extr   = scene.extrinsics[img.extr].se3;
        //        auto& extr = scene.extrinsics[img.extr].se3;
        auto& extr   = img.se3;
        auto& camera = scene.intrinsics[img.intr];

        for (auto& ip : img.stereoPoints)
        {
            if (!ip) continue;
            auto& wp = scene.worldPoints[ip.wp].p;
            double w = ip.weight * scene.scale();


            auto cost_function                = CostBARSMono::create(camera, ip.point, w);
            ceres::LossFunction* lossFunction = baOptions.huberMono > 0 ? &lossFunctionMono : nullptr;
            problem.AddResidualBlock(cost_function, lossFunction, extr.data(), extr.data() + 8, wp.data());

            // With this ordering the schur complement is computed in the correct order
            //            ordering->AddElementToGroup(wp.data(), 0);
            //            ordering->AddElementToGroup(extr.data(), 1);
        }
    }



    ceres::Solver::Options ceres_options = make_options(optimizationOptions);
    //    ceres_options.linear_solver_ordering = ordering;


    OptimizationResults result = ceres_solve(ceres_options, problem);


    //    for (auto& img : scene.images)
    {
        //        auto& extr = img.se3;


        SAIGA_EXIT_ERROR("not implemented");
        //        auto data = mixedData[img.extr];


        //        Eigen::Map<Sophus::SE3<double>> s1(data.data());

        //        Eigen::Map<Sophus::SE3<double>> s2(data.data() + 8);
        //        auto rel = s2 * s1.inverse();
        //        std::cout << "Rel: " << rel << std::endl;

        //        Eigen::Map<Vec6> s2(data.data() + 8);
        //        std::cout << s2.transpose() << std::endl;

        //        extr = s1;
    }


    return result;
}

}  // namespace Saiga
