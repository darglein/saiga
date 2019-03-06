/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/Optimizer.h"
#include "saiga/vision/VisionTypes.h"

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


inline void printDebugSmall(ceres::Problem& problem)
{
    ceres::Problem::EvaluateOptions defaultEvalOptions;

    double costFinal = 0;

    std::vector<double> residuals, gradient;
    problem.Evaluate(defaultEvalOptions, &costFinal, &residuals, &gradient, nullptr);

    cout << "num residuals: " << problem.NumResiduals() << endl;
    cout << "cost: " << costFinal << endl;



    //        cout << "gradient" << endl;
    //        for (auto d : gradient) cout << d << " ";
    //        cout << endl << "residuals" << endl;
    //        for (auto d : residuals) cout << d << " ";
    //        cout << endl;
}

inline void printDebugJacobi(ceres::Problem& problem, int maxRows = 10, bool printJtJ = false)
{
    cout << "[Ceres Problem Info]" << endl;
    ceres::Problem::EvaluateOptions defaultEvalOptions;
    ceres::CRSMatrix matrix;
    double costFinal = 0;

    std::vector<double> residuals, gradient;
    problem.Evaluate(defaultEvalOptions, &costFinal, &residuals, &gradient, &matrix);


    cout << "num residuals: " << problem.NumResiduals() << endl;
    cout << matrix.num_rows << "x" << matrix.num_cols << endl;

    cout << "cost: " << costFinal << endl;

    {
        cout << "[Residual r]" << endl;
        double residualSum = 0;
        for (int i = 0; i < std::min<int>(maxRows, residuals.size()); ++i)
        {
            auto g = residuals[i];
            residualSum += g;
            cout << g << " ";
        }
        cout << endl << "Sum = " << residualSum << endl;
    }

    {
        cout << "[Gradient -Jr]" << endl;
        double gradientSum = 0;
        for (int i = 0; i < std::min<int>(maxRows, gradient.size()); ++i)
        {
            auto g = gradient[i];
            gradientSum += g;
            cout << g << " ";
        }
        cout << endl << "Sum = " << gradientSum << endl;
    }



    {
        cout << "[Jacobian J]" << endl;

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
        //        cout << "gradient" << endl;
        //        for (auto d : gradient) cout << d << " ";
        //        cout << endl << "residuals" << endl;
        //        for (auto d : residuals) cout << d << " ";
        //        cout << endl;


        // Only print a few rows because it could get very big
        for (int i = 0; i < std::min<int>(maxRows, ematrix.rows()); ++i)
        {
            for (SM::InnerIterator it(ematrix, i); it; ++it)
            {
                cout << it.value() << " ";
            }
            cout << endl;
        }

        cout << "|J| = " << ematrix.norm() << endl;


        if (printJtJ)
        {
            SM JtJ = (ematrix.transpose() * ematrix).triangularView<Eigen::Upper>();
            cout << "JtJ" << endl << JtJ.toDense() << endl << endl;
        }
    }
}

}  // namespace Saiga
