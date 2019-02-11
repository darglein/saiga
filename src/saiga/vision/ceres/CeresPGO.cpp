/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "CeresPGO.h"

#include "saiga/core/time/timer.h"

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
void CeresPGO::solve(PoseGraph& scene, const PGOOptions& options)
{
    SAIGA_OPTIONAL_BLOCK_TIMER(options.debugOutput);

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

    ceres::Solver::Options ceres_options;
    ceres_options.minimizer_progress_to_stdout = options.debugOutput;
    //    ceres_options.minimizer_progress_to_stdout = true;
    ceres_options.max_num_iterations           = options.maxIterations;
    ceres_options.max_linear_solver_iterations = options.maxIterativeIterations;
    ceres_options.min_linear_solver_iterations = options.maxIterativeIterations;
    ceres_options.min_relative_decrease        = 1e-50;
    ceres_options.function_tolerance           = 1e-50;
    ceres_options.gradient_tolerance           = 1e-50;
    ceres_options.parameter_tolerance          = 1e-50;


    switch (options.solverType)
    {
        case PGOOptions::SolverType::Direct:
            ceres_options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
            break;
        case PGOOptions::SolverType::Iterative:
            ceres_options.linear_solver_type = ceres::LinearSolverType::ITERATIVE_SCHUR;
            break;
    }

    ceres::Solver::Summary summaryTest;

#if 0

    ceres::Problem::EvaluateOptions defaultEvalOptions;
    ceres::CRSMatrix matrix;
    double costFinal = 0;
    problem.Evaluate(defaultEvalOptions, &costFinal, nullptr, nullptr, &matrix);

    cout << "num residuals: " << problem.NumResiduals() << endl;

    cout << matrix.num_rows << "x" << matrix.num_cols << endl;

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
        cout << ematrix.toDense() << endl;
    }
#endif

    //    exit(0);


    {
        SAIGA_OPTIONAL_BLOCK_TIMER(options.debugOutput);
        ceres::Solve(ceres_options, &problem, &summaryTest);

        if (options.debugOutput)
        {
            for (auto t : summaryTest.iterations)
            {
                cout << t.step_is_successful << " " << t.cost_change << " " << t.gradient_max_norm << " "
                     << t.relative_decrease << " " << t.step_norm << " " << endl;
            }
            cout << "Termination: " << summaryTest.termination_type << endl;
        }
    }


    //    std::cout << "optimizePoints residual: " << costInit << " -> " << costFinal << endl;
}

}  // namespace Saiga
