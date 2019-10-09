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
#include "local_parameterization_sim3.h"

#define AUTO_DIFF

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



#ifdef AUTO_DIFF

    Saiga::test::LocalParameterizationSim3<false> camera_parameterization;

#else
#    ifdef PGO_SIM3
    test::LocalParameterizationSim3_IdentityJ camera_parameterization;
    camera_parameterization.fixScale = scene.fixScale;
#    else
    Sophus::test::LocalParameterizationSE32 camera_parameterization;
#    endif
#endif

    for (size_t i = 0; i < scene.poses.size(); ++i)
    {
        problem.AddParameterBlock(scene.poses[i].se3.data(), 7, &camera_parameterization);
        if (scene.poses[i].constant)
        {
            problem.SetParameterBlockConstant(scene.poses[i].se3.data());
        }
    }
#ifdef AUTO_DIFF
    using CostFunctionType = CostPGO::CostFunctionType;
#else
    using CostFunctionType           = CostPGOAnalytic;
#endif


    std::vector<std::unique_ptr<CostFunctionType>> monoCostFunctions;

    // Add all transformation edges
    for (auto& e : scene.edges)
    {
        auto vertex_from = scene.poses[e.from].se3.data();
        auto vertex_to   = scene.poses[e.to].se3.data();

#ifdef AUTO_DIFF
        CostFunctionType* cost = CostPGO::create(e.meassurement.inverse());
#else
        CostFunctionType* cost = new CostFunctionType(e.meassurement.inverse());
#endif
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
