/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/Optimizer.h"
#include "saiga/vision/VisionTypes.h"

#include "ceres/solver.h"

namespace Saiga
{
inline void makeGaussNewtonOptions(ceres::Solver::Options& options)
{
    options.min_trust_region_radius     = 1e-32;
    options.max_trust_region_radius     = 1e51;
    options.initial_trust_region_radius = 1e30;
    //            options.min_trust_region_radius = 10e50;
    options.min_lm_diagonal = 1e-50;
    options.max_lm_diagonal = 1e-49;
}

inline ceres::Solver::Options make_options(const Saiga::OptimizationOptions& optimizationOptions)
{
    ceres::Solver::Options ceres_options;
    ceres_options.minimizer_progress_to_stdout = optimizationOptions.debugOutput;
    ceres_options.max_num_iterations           = optimizationOptions.maxIterations;
    ceres_options.max_linear_solver_iterations = optimizationOptions.maxIterativeIterations;
    ceres_options.min_linear_solver_iterations = optimizationOptions.maxIterativeIterations;
    ceres_options.min_relative_decrease        = 1e-50;
    ceres_options.function_tolerance           = 1e-50;
    ceres_options.initial_trust_region_radius  = 1.0 / optimizationOptions.initialLambda;

    switch (optimizationOptions.solverType)
    {
        case OptimizationOptions::SolverType::Direct:
            ceres_options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
            break;
        case OptimizationOptions::SolverType::Iterative:
            ceres_options.linear_solver_type = ceres::LinearSolverType::ITERATIVE_SCHUR;
            break;
    }

    return ceres_options;
}

inline Saiga::OptimizationResults ceres_solve(const ceres::Solver::Options& ceres_options, ceres::Problem& problem)
{
    ceres::Solver::Summary summaryTest;

    {
        ceres::Solve(ceres_options, &problem, &summaryTest);
    }

    //    cout << "linear solver time " << summaryTest.linear_solver_time_in_seconds << "s." << endl;
    //    cout << summaryTest.FullReport() << endl;

    OptimizationResults result;
    result.cost_initial       = summaryTest.initial_cost * 2.0;
    result.cost_final         = summaryTest.final_cost * 2.0;
    result.linear_solver_time = summaryTest.linear_solver_time_in_seconds * 1000;
    result.total_time         = summaryTest.total_time_in_seconds * 1000;
    return result;
}

}  // namespace Saiga
