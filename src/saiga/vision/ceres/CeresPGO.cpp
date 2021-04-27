/**
 * Copyright (c) 2021 Darius Rückert
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


    Sophus::test::LocalParameterizationSE3_Autodiff camera_parameterization_se3;
    Sophus::test::LocalParameterizationDSim3_Autodiff camera_parameterization_sim3;


    for (size_t i = 0; i < scene.vertices.size(); ++i)
    {
        ceres::LocalParameterization* lp = scene.fixScale
                                               ? (ceres::LocalParameterization*)&camera_parameterization_se3
                                               : (ceres::LocalParameterization*)&camera_parameterization_sim3;

        auto global_size = scene.fixScale ? 7 : 8;
        problem.AddParameterBlock(scene.vertices[i].T_w_i.data(), global_size, lp);
        //        problem.AddParameterBlock(scene.poses[i].se3.data(), 7);
        if (scene.vertices[i].constant)
        {
            problem.SetParameterBlockConstant(scene.vertices[i].T_w_i.data());
        }
    }

    std::vector<std::unique_ptr<ceres::CostFunction>> monoCostFunctions;

    // Add all transformation edges
    for (auto& e : scene.edges)
    {
        auto vertex_from = scene.vertices[e.from].T_w_i.data();
        auto vertex_to   = scene.vertices[e.to].T_w_i.data();

        ceres::CostFunction* cost = scene.fixScale ? (ceres::CostFunction*)CostPGO::create(e.T_i_j.se3())
                                                   : (ceres::CostFunction*)CostPGODSim3::create(e.T_i_j);
        problem.AddResidualBlock(cost, nullptr, vertex_from, vertex_to);
        monoCostFunctions.emplace_back(cost);

        //        auto test = CostPGODSim3::create(e.T_i_j);
        //        CostPGODSim3 test(e.T_i_j);
        //        Eigen::Matrix<double, 7, 1> residual;
        //        test(vertex_from, vertex_to, residual.data());
        //        std::cout << "i " << scene.vertices[e.from].T_w_i << std::endl;
        //        std::cout << "res " << residual.transpose() << std::endl;
    }

    ceres::Solver::Options ceres_options = make_options(optimizationOptions);
    OptimizationResults result           = ceres_solve(ceres_options, problem);
    return result;
}

}  // namespace Saiga
