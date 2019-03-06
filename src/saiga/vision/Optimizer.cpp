/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Optimizer.h"

#include "saiga/core/imgui/imgui.h"

namespace Saiga
{
std::ostream& operator<<(std::ostream& strm, const OptimizationResults& op)
{
    strm << "[" << op.name << "] " << op.cost_initial << "->" << op.cost_final << " in " << op.total_time / 1000.0
         << "(" << op.linear_solver_time / 1000.0 << ")";
    if (!op.success) strm << " FAILED!";
    return strm;
}


void OptimizationOptions::imgui()
{
    ImGui::InputInt("maxIterations", &maxIterations);
    int currentItem             = (int)solverType;
    static const char* items[2] = {"Iterative", "Direct"};
    ImGui::Combo("SolverType", &currentItem, items, 2);
    solverType = (SolverType)currentItem;

    if (solverType == SolverType::Iterative)
    {
        ImGui::InputInt("maxIterativeIterations", &maxIterativeIterations);
        ImGui::InputDouble("iterativeTolerance", &iterativeTolerance);
    }

    ImGui::Checkbox("debugOutput", &debugOutput);
}


std::ostream& operator<<(std::ostream& strm, const OptimizationOptions& op)
{
    strm << "[Optimization Options]" << endl;
    strm << " Iterations: " << op.maxIterations << endl;
    strm << " Initial Lambda: " << op.initialLambda << endl;

    if (op.solverType == OptimizationOptions::SolverType::Iterative)
    {
        strm << " solverType: CG Schur" << endl;
        strm << " maxIterativeIterations: " << op.maxIterativeIterations << endl;
        strm << " iterativeTolerance: " << op.iterativeTolerance << endl;
    }
    else
    {
        strm << " solverType: LDLT Schur" << endl;
    }
    return strm;
}

OptimizationResults LMOptimizer::solve()
{
    OptimizationResults result;

    lambda = optimizationOptions.initialLambda;

    {
        Saiga::ScopedTimer<double> timer(result.total_time);
        init();

        double current_chi2 = 0;

        result.linear_solver_time = 0;

        for (auto i = 0; i < optimizationOptions.maxIterations; ++i)
        {
            double chi2 = computeQuadraticForm();


            if (optimizationOptions.debug)
            {
                // A small sanity check in debug mode to see if compute cost is correct
                double test = computeCost();
                if (chi2 != test)
                {
                    cerr << "Warning " << chi2 << "!=" << test << endl;
                    SAIGA_ASSERT(chi2 == test);
                }
            }



            addLambda(lambda);
            if (i == 0)
            {
                current_chi2        = chi2;
                result.cost_initial = chi2;
                if (optimizationOptions.debugOutput) cout << "initial_chi2 = " << 0.5 * current_chi2 << endl;
            }
            result.cost_final = chi2;

            double ltime;
            {
                Saiga::ScopedTimer<double> timer(ltime);
                solveLinearSystem();
            }
            result.linear_solver_time += ltime;

            addDelta();

            double newChi2 = computeCost();

            if (newChi2 < current_chi2)
            {
                // accept
                lambda       = lambda * (1.0 / 3.0);
                v            = 2;
                current_chi2 = newChi2;
            }
            else
            {
                // discard
                lambda = lambda * v;
                v      = 2 * v;
                revertDelta();
                if (optimizationOptions.debugOutput)
                    cerr << i << " warning invalid lm step. lambda: " << lambda << " new/old: " << newChi2 << "/"
                         << current_chi2 << endl;
            }

            if (optimizationOptions.debugOutput) cout << "current_chi2 = " << 0.5 * current_chi2 << endl;
        }
        finalize();

        result.cost_final = current_chi2;
    }
    return result;
}



}  // namespace Saiga
