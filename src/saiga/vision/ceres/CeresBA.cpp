/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "CeresBA.h"

#include "saiga/core/time/timer.h"
#include "saiga/vision/scene/Scene.h"

#include "CeresHelper.h"
#include "Eigen/Sparse"
#include "Eigen/SparseCholesky"
#include "ceres/ceres.h"
#include "ceres/problem.h"
#include "ceres/rotation.h"
#include "ceres/solver.h"

#include "CeresKernel_BA_Intr4.h"
#include "local_parameterization_se3.h"

#define BA_AUTODIFF

namespace Saiga
{
OptimizationResults CeresBA::initAndSolve()
{
    auto& scene = *_scene;

    SAIGA_OPTIONAL_BLOCK_TIMER(optimizationOptions.debugOutput);

    ceres::Problem::Options problemOptions;
    problemOptions.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problemOptions.cost_function_ownership          = ceres::DO_NOT_TAKE_OWNERSHIP;
    problemOptions.loss_function_ownership          = ceres::DO_NOT_TAKE_OWNERSHIP;

    ceres::Problem problem(problemOptions);


#ifdef BA_AUTODIFF
    Sophus::test::LocalParameterizationSE3 camera_parameterization;
#else
    Sophus::test::LocalParameterizationSE32 camera_parameterization;
#endif
    for (size_t i = 0; i < scene.extrinsics.size(); ++i)
    {
        problem.AddParameterBlock(scene.extrinsics[i].se3.data(), 7, &camera_parameterization);
        if (scene.extrinsics[i].constant) problem.SetParameterBlockConstant(scene.extrinsics[i].se3.data());
    }

    ceres::HuberLoss lossFunctionMono(baOptions.huberMono);
    ceres::HuberLoss lossFunctionStereo(baOptions.huberStereo);

    ceres::LossFunction* lossStereo = baOptions.huberStereo > 0 ? &lossFunctionStereo : nullptr;
    ceres::LossFunction* lossMono   = baOptions.huberMono > 0 ? &lossFunctionMono : nullptr;

    int monoCount   = 0;
    int stereoCount = 0;
    for (auto& img : scene.images)
    {
        if (!img) continue;
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


    std::vector<std::unique_ptr<CostBAMonoAnalytic>> monoCostFunctions;
    std::vector<std::unique_ptr<CostBAStereoAnalytic>> stereoCostFunctions;

    monoCostFunctions.reserve(monoCount);
    stereoCostFunctions.reserve(stereoCount);


    for (auto& img : scene.images)
    {
        if (!img) continue;

        auto& extr   = scene.extrinsics[img.extr].se3;
        auto& camera = scene.intrinsics[img.intr];
        StereoCamera4 scam(camera, scene.bf);



        for (auto& ip : img.stereoPoints)
        {
            if (!ip) continue;
            auto& wp = scene.worldPoints[ip.wp].p;
            double w = ip.weight * scene.scale();


            if (ip.depth > 0)
            {
#ifdef BA_AUTODIFF
                auto stereoPoint   = ip.point(0) - scene.bf / ip.depth;
                auto cost_function = CostBAStereo<>::create(camera, ip.point, stereoPoint, scene.bf, Vec2(w, w));

                problem.AddResidualBlock(cost_function, lossStereo, extr.data(), wp.data());
#else
                auto* cost = new CostBAStereoAnalytic(scam, ip.point, ip.depth, w);
                stereoCostFunctions.emplace_back(cost);
                problem.AddResidualBlock(cost, baOptions.huberStereo > 0 ? &lossFunctionStereo : nullptr, extr.data(),
                                         wp.data());
#endif
                // With this ordering the schur complement is computed in the correct order
                ordering->AddElementToGroup(wp.data(), 0);
                ordering->AddElementToGroup(extr.data(), 1);
            }
            else
            {
#ifdef BA_AUTODIFF
                auto cost_function = CostBAMono::create(camera, ip.point, w);

                problem.AddResidualBlock(cost_function, lossMono, extr.data(), wp.data());
#else
                auto* cost = new CostBAMonoAnalytic(camera, ip.point, w);
                monoCostFunctions.emplace_back(cost);
                problem.AddResidualBlock(cost, baOptions.huberMono > 0 ? &lossFunctionMono : nullptr, extr.data(),
                                         wp.data());
#endif
                // With this ordering the schur complement is computed in the correct order
                ordering->AddElementToGroup(wp.data(), 0);
                ordering->AddElementToGroup(extr.data(), 1);
            }
        }
    }



    //    double costInit = 0;
    //    ceres::Problem::EvaluateOptions defaultEvalOptions;
    //    problem.Evaluate(defaultEvalOptions, &costInit, nullptr, nullptr, nullptr);


    ceres::Solver::Options ceres_options = make_options(optimizationOptions);
    ceres_options.linear_solver_ordering = ordering;



#if 0

    ceres::Problem::EvaluateOptions defaultEvalOptions;
    ceres::CRSMatrix matrix;
    double costFinal = 0;
    problem.Evaluate(defaultEvalOptions, &costFinal, nullptr, nullptr, &matrix);

    std::cout << "num residuals: " << problem.NumResiduals() << std::endl;

    std::cout << matrix.num_rows << "x" << matrix.num_cols << std::endl;

    {
        Eigen::SparseMatrix<double, Eigen::RowMajor> ematrix(matrix.num_rows, matrix.num_cols);
        ematrix.resizeNonZeros(matrix.values.size());

        for (int i = 0; i < matrix.num_rows + 1; ++i)
        {
            ematrix.outerIndexPtr()[i] = matrix.rows[i];
        }
        for (int i = 0; i < matrix.values.size(); ++i)
        {
            ematrix.valuePtr()[i]      = matrix.values[i];
            ematrix.innerIndexPtr()[i] = matrix.cols[i];
        }
        std::cout << ematrix.toDense() << std::endl;
    }
    return;
#endif

    //    exit(0);


    //    ceres::Solver::Summary summaryTest;

    //    {
    //        SAIGA_OPTIONAL_BLOCK_TIMER(optimizationOptions.debugOutput);
    //        ceres::Solve(ceres_options, &problem, &summaryTest);
    //    }

    //    std::cout << "linear solver time " << summaryTest.linear_solver_time_in_seconds << "s." << std::endl;
    //    std::cout << summaryTest.FullReport() << std::endl;

    OptimizationResults result = ceres_solve(ceres_options, problem);

    //    result.name               = name;
    //    result.cost_initial       = summaryTest.initial_cost * 2.0;
    //    result.cost_final         = summaryTest.final_cost * 2.0;
    //    result.linear_solver_time = summaryTest.linear_solver_time_in_seconds * 1000;
    //    result.total_time         = summaryTest.total_time_in_seconds * 1000;
    //    if (summaryTest.iterations.size() < optimizationOptions.maxIterations + 1)
    //    {
    //        std::cout << "less iterations than expected: " << summaryTest.iterations.size() << std::endl;
    //    }
    return result;

    //    std::cout << "optimizePoints residual: " << costInit << " -> " << costFinal << std::endl;
}

}  // namespace Saiga
