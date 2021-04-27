/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/util/Optimizer.h"

#include "g2o/core/solver.h"

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/pcg/linear_solver_pcg.h"


#ifdef SAIGA_USE_CHOLMOD
#    include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#endif

namespace Saiga
{
template <typename BlockSolver>
auto g2o_make_linearSolver(const Saiga::OptimizationOptions& optimizationOptions)
{
    using LinearSolver = std::unique_ptr<typename BlockSolver::LinearSolverType>;

    LinearSolver linearSolver;
    switch (optimizationOptions.solverType)
    {
        case OptimizationOptions::SolverType::Direct:
        {
#ifdef SAIGA_USE_CHOLMOD
            auto ls = std::make_unique<g2o::LinearSolverCholmod<typename BlockSolver::PoseMatrixType>>();
#else
            auto ls = std::make_unique<g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>>();
#endif
            linearSolver = std::move(ls);
            break;
        }
        case OptimizationOptions::SolverType::Iterative:
        {
            auto ls = g2o::make_unique<g2o::LinearSolverPCG<typename BlockSolver::PoseMatrixType>>();
            ls->setMaxIterations(optimizationOptions.maxIterativeIterations);
            ls->setTolerance(optimizationOptions.iterativeTolerance * optimizationOptions.iterativeTolerance);
            linearSolver = std::move(ls);
            break;
        }
    }
    return linearSolver;
}

template <typename BlockSolver>
auto g2o_make_optimizationAlgorithm(const Saiga::OptimizationOptions& optimizationOptions)
{
    using LinearSolver = std::unique_ptr<typename BlockSolver::LinearSolverType>;

    LinearSolver linearSolver;
    switch (optimizationOptions.solverType)
    {
        case OptimizationOptions::SolverType::Direct:
        {
#ifdef SAIGA_USE_CHOLMOD
            auto ls = std::make_unique<g2o::LinearSolverCholmod<typename BlockSolver::PoseMatrixType>>();
#else
            auto ls = std::make_unique<g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>>();
#endif
            linearSolver = std::move(ls);
            break;
        }
        case OptimizationOptions::SolverType::Iterative:
        {
            auto ls = g2o::make_unique<g2o::LinearSolverPCG<typename BlockSolver::PoseMatrixType>>();
            ls->setMaxIterations(optimizationOptions.maxIterativeIterations);
            ls->setTolerance(optimizationOptions.iterativeTolerance * optimizationOptions.iterativeTolerance);
            linearSolver = std::move(ls);
            break;
        }
    }

    using OptimizationAlgorithm = g2o::OptimizationAlgorithmLevenberg;

    OptimizationAlgorithm* solver = new OptimizationAlgorithm(std::make_unique<BlockSolver>(std::move(linearSolver)));
    //    solver->setUserLambdaInit(optimizationOptions.initialLambda * 2);
    solver->setMaxTrialsAfterFailure(2);
    //    g2o::SparseOptimizer optimizer;
    //    optimizer.setVerbose(optimizationOptions.debugOutput);
    //    optimizer.setComputeBatchStatistics(options.debugOutput);
    //    optimizer.setComputeBatchStatistics(true);
    //    optimizer.setAlgorithm(solver);
    return solver;
}

}  // namespace Saiga
