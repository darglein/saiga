/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/util/Optimizer.h"

#include "Eigen/Sparse"
#include "ceres/problem.h"
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

inline ceres::Solver::Options make_options(const Saiga::OptimizationOptions& optimizationOptions, bool schur = true)
{
    ceres::Solver::Options ceres_options;
    ceres_options.minimizer_progress_to_stdout = optimizationOptions.debugOutput;
    ceres_options.max_num_iterations           = optimizationOptions.maxIterations;
    ceres_options.max_linear_solver_iterations = optimizationOptions.maxIterativeIterations;
    ceres_options.min_linear_solver_iterations = optimizationOptions.maxIterativeIterations;

    ceres_options.min_relative_decrease       = 1e-50;
    ceres_options.function_tolerance          = 1e-50;
    ceres_options.initial_trust_region_radius = 1.0 / optimizationOptions.initialLambda;

    switch (optimizationOptions.solverType)
    {
        case OptimizationOptions::SolverType::Direct:
            ceres_options.linear_solver_type =
                schur ? ceres::LinearSolverType::SPARSE_SCHUR : ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;

            break;
        case OptimizationOptions::SolverType::Iterative:
            ceres_options.linear_solver_type =
                schur ? ceres::LinearSolverType::ITERATIVE_SCHUR : ceres::LinearSolverType::CGNR;
            break;
    }

    return ceres_options;
}

inline Saiga::OptimizationResults ceres_solve(const ceres::Solver::Options& ceres_options, ceres::Problem& problem,
                                              bool print_full_report = false)
{
    ceres::Solver::Summary summaryTest;

    {
        ceres::Solve(ceres_options, &problem, &summaryTest);
    }

    //    std::cout << "linear solver time " << summaryTest.linear_solver_time_in_seconds << "s." << std::endl;
    if (print_full_report || summaryTest.termination_type == ceres::FAILURE)
    {
        std::cout << summaryTest.FullReport() << std::endl;
    }



    OptimizationResults result;
    result.cost_initial       = summaryTest.initial_cost * 2.0;
    result.cost_final         = summaryTest.final_cost * 2.0;
    result.linear_solver_time = summaryTest.linear_solver_time_in_seconds * 1000;
    result.total_time         = summaryTest.total_time_in_seconds * 1000;
    result.success            = summaryTest.IsSolutionUsable();
    return result;
}


inline void printDebugSmall(ceres::Problem& problem)
{
    ceres::Problem::EvaluateOptions defaultEvalOptions;

    double costFinal = 0;

    std::vector<double> residuals, gradient;
    problem.Evaluate(defaultEvalOptions, &costFinal, &residuals, &gradient, nullptr);

    std::cout << "num residuals: " << problem.NumResiduals() << std::endl;
    std::cout << "cost: " << costFinal << std::endl;



    //        std::cout << "gradient" << std::endl;
    //        for (auto d : gradient) std::cout << d << " ";
    //        std::cout << std::endl << "residuals" << std::endl;
    //        for (auto d : residuals) std::cout << d << " ";
    //        std::cout << std::endl;
}

inline void printDebugJacobi(ceres::Problem& problem, int maxRows = 10, bool printJtJ = false)
{
    std::cout << "[Ceres Problem Info]" << std::endl;
    ceres::Problem::EvaluateOptions defaultEvalOptions;
    ceres::CRSMatrix matrix;
    double costFinal = 0;

    std::vector<double> residuals, gradient;
    problem.Evaluate(defaultEvalOptions, &costFinal, &residuals, &gradient, &matrix);


    std::cout << "num residuals: " << problem.NumResiduals() << std::endl;
    std::cout << matrix.num_rows << "x" << matrix.num_cols << std::endl;

    std::cout << "cost: " << costFinal << std::endl;

    {
        Eigen::Matrix<double, -1, 1> r(std::min<int>(maxRows, residuals.size()));

        for (int i = 0; i < r.rows(); ++i)
        {
            r(i) = residuals[i];
        }
        std::cout << "[Residual r]" << std::endl;
        std::cout << r.transpose() << std::endl;
        std::cout << "|r| = " << r.norm() << std::endl << std::endl;
    }

    {
        Eigen::Matrix<double, -1, 1> g(std::min<int>(maxRows, gradient.size()));
        for (int i = 0; i < g.rows(); ++i)
        {
            g(i) = gradient[i];
        }
        std::cout << "[Gradient -Jr]" << std::endl;
        std::cout << g.transpose() << std::endl;
        std::cout << "|g| = " << g.norm() << std::endl << std::endl;
    }



    {
        std::cout << "[Jacobian J]" << std::endl;

        using SM = Eigen::SparseMatrix<double, Eigen::RowMajor>;
        SM ematrix(matrix.num_rows, matrix.num_cols);
        ematrix.resizeNonZeros(matrix.values.size());

        for (int i = 0; i < matrix.num_rows + 1; ++i)
        {
            ematrix.outerIndexPtr()[i] = matrix.rows[i];
        }
        for (int i = 0; i < (int)matrix.values.size(); ++i)
        {
            ematrix.valuePtr()[i]      = matrix.values[i];
            ematrix.innerIndexPtr()[i] = matrix.cols[i];
        }
        //        std::cout << "gradient" << std::endl;
        //        for (auto d : gradient) std::cout << d << " ";
        //        std::cout << std::endl << "residuals" << std::endl;
        //        for (auto d : residuals) std::cout << d << " ";
        //        std::cout << std::endl;

#if 0
        // Only print a few rows because it could get very big
        for (int i = 0; i < std::min<int>(maxRows, ematrix.rows()); ++i)
        {
            for (SM::InnerIterator it(ematrix, i); it; ++it)
            {
                std::cout << it.value() << " ";
            }
            std::cout << std::endl;
        }
#else
        std::cout << ematrix.toDense() << std::endl;
#endif
        std::cout << "|J| = " << ematrix.norm() << std::endl;


        if (printJtJ)
        {
            SM JtJ = (ematrix.transpose() * ematrix);  // .triangularView<Eigen::Upper>();
            std::cout << "[JtJ]" << std::endl;
            std::cout << JtJ.toDense() << std::endl << std::endl;
            std::cout << "|J| = " << JtJ.norm() << std::endl;
        }
    }
}

}  // namespace Saiga
