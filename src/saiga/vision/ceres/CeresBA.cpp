/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "CeresBA.h"

#include "saiga/time/timer.h"

#include "Eigen/Sparse"
#include "Eigen/SparseCholesky"
#include "ceres/ceres.h"
#include "ceres/problem.h"
#include "ceres/rotation.h"
#include "ceres/solver.h"

#include "CeresKernel_BA_Intr4.h"
#include "local_parameterization_se3.h"

namespace Saiga
{
void CeresBA::optimize(Scene& scene, int its)
{
    SAIGA_BLOCK_TIMER();
    ceres::Problem problem;


    Sophus::test::LocalParameterizationSE3* camera_parameterization = new Sophus::test::LocalParameterizationSE3;
    for (size_t i = 0; i < scene.extrinsics.size(); ++i)
    {
        problem.AddParameterBlock(scene.extrinsics[i].se3.data(), 7, camera_parameterization);
        //        problem.SetParameterization(scene.extrinsics[i].se3.data(), camera_parameterization);
    }

    for (auto& img : scene.images)
    {
        auto& extr   = scene.extrinsics[img.extr].se3;
        auto& camera = scene.intrinsics[img.intr];

        for (auto& ip : img.monoPoints)
        {
            if (!ip) continue;
            auto& wp           = scene.worldPoints[ip.wp].p;
            double w           = ip.weight * scene.scale();
            auto cost_function = CostBAMono::create(camera, ip.point, w);
            problem.AddResidualBlock(cost_function, nullptr, extr.data(), wp.data());
        }

        for (auto& ip : img.stereoPoints)
        {
            if (!ip) continue;
            auto& wp           = scene.worldPoints[ip.wp].p;
            double w           = ip.weight * scene.scale();
            auto stereoPoint   = ip.point(0) - scene.bf / ip.depth;
            auto cost_function = CostBAStereo<>::create(camera, ip.point, stereoPoint, scene.bf, Vec2(w, w));
            problem.AddResidualBlock(cost_function, nullptr, extr.data(), wp.data());
        }
    }


    //    double costInit = 0;
    //    ceres::Problem::EvaluateOptions defaultEvalOptions;
    //    problem.Evaluate(defaultEvalOptions, &costInit, nullptr, nullptr, nullptr);

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations           = its;

    options.linear_solver_type           = ceres::LinearSolverType::CGNR;
    options.max_linear_solver_iterations = 20;
    options.min_linear_solver_iterations = 20;
    ceres::Solver::Summary summaryTest;

    {
        SAIGA_BLOCK_TIMER("Solve");
        ceres::Solve(options, &problem, &summaryTest);
    }

    //    double costFinal = 0;
    //    problem.Evaluate(defaultEvalOptions, &costFinal, nullptr, nullptr, nullptr);

    //    std::cout << "optimizePoints residual: " << costInit << " -> " << costFinal << endl;
}

}  // namespace Saiga
