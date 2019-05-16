/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "CeresPGO.h"

#include "saiga/core/time/timer.h"

#include "CeresHelper.h"
#include "Eigen/Sparse"
#include "Eigen/SparseCholesky"
#include "ceres/ceres.h"
#include "ceres/problem.h"
#include "ceres/rotation.h"
#include "ceres/solver.h"

#include "CeresKernel_PGO.h"
#include "local_parameterization_se3.h"

namespace Saiga
{
OptimizationResults CeresPGO::initAndSolve()
{
    auto& scene = *_scene;
    SAIGA_OPTIONAL_BLOCK_TIMER(optimizationOptions.debugOutput);

    ceres::Problem::Options problemOptions;
    problemOptions.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problemOptions.cost_function_ownership          = ceres::DO_NOT_TAKE_OWNERSHIP;
    problemOptions.loss_function_ownership          = ceres::DO_NOT_TAKE_OWNERSHIP;

    ceres::Problem problem(problemOptions);



    //    Sophus::test::LocalParameterizationSE3* camera_parameterization = new Sophus::test::LocalParameterizationSE3;
    Sophus::test::LocalParameterizationSE32 camera_parameterization;
    for (size_t i = 0; i < scene.poses.size(); ++i)
    {
        problem.AddParameterBlock(scene.poses[i].se3.data(), 7, &camera_parameterization);
    }


    std::vector<std::unique_ptr<CostPGOAnalytic>> monoCostFunctions;

    // Add all transformation edges
    for (auto& e : scene.edges)
    {
        auto vertex_from = scene.poses[e.from].se3.data();
        auto vertex_to   = scene.poses[e.to].se3.data();

        CostPGOAnalytic* cost = new CostPGOAnalytic(e.meassurement.inverse());
        monoCostFunctions.emplace_back(cost);
        problem.AddResidualBlock(cost, nullptr, vertex_from, vertex_to);
    }



    //    double costInit = 0;
    //    ceres::Problem::EvaluateOptions defaultEvalOptions;
    //    problem.Evaluate(defaultEvalOptions, &costInit, nullptr, nullptr, nullptr);

    ceres::Solver::Options ceres_options = make_options(optimizationOptions);

    OptimizationResults result = ceres_solve(ceres_options, problem);
    return result;
}

}  // namespace Saiga
