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


    for (auto& img : scene.images)
    {
        for (auto& ip : img.monoPoints)
        {
            if (!ip) continue;
            auto& wp     = scene.worldPoints[ip.wp].p;
            auto& extr   = scene.extrinsics[img.extr].se3;
            auto& camera = scene.intrinsics[img.intr];
            double w     = ip.weight * scene.scale();

            auto cost_function = CostBAMono::create(camera, ip.point, w);
            problem.AddResidualBlock(cost_function, nullptr, extr.data(), wp.data());
        }
    }

    Sophus::test::LocalParameterizationSE3* camera_parameterization = new Sophus::test::LocalParameterizationSE3;
    for (size_t i = 0; i < scene.extrinsics.size(); ++i)
    {
        problem.SetParameterization(scene.extrinsics[i].se3.data(), camera_parameterization);
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
