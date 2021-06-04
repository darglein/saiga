/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Optimizer.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/Thread/omp.h"

#include <iostream>
#include <memory>

namespace Saiga
{
std::ostream& operator<<(std::ostream& strm, const OptimizationResults& op)
{
    strm << "[" << op.name << "] " << op.cost_initial << " -> " << op.cost_final
         << " | Timings (ms): Total=" << op.total_time << " Lin=" << op.linear_solver_time << " JtJ=" << op.jtj_time
         << "";
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
    strm << "[Optimization Options]" << std::endl;
    strm << " Iterations: " << op.maxIterations << std::endl;
    strm << " Initial Lambda: " << op.initialLambda << std::endl;

    if (op.solverType == OptimizationOptions::SolverType::Iterative)
    {
        strm << " solverType: CG Schur" << std::endl;
        strm << " maxIterativeIterations: " << op.maxIterativeIterations << std::endl;
        strm << " iterativeTolerance: " << op.iterativeTolerance << std::endl;
    }
    else
    {
        strm << " solverType: LDLT Schur" << std::endl;
    }
    return strm;
}

OptimizationResults LMOptimizer::solve()
{
    double current_chi2 = std::numeric_limits<double>::max();

    OptimizationResults result;
    result.linear_solver_time = 0;



    Table debug_output_table({8, 15, 15, 15, 15});
    if (optimizationOptions.debugOutput)
    {
        debug_output_table << "iter"
                           << "cost"
                           << "lambda"
                           << "jtj_time (ms)"
                           << "solve_time (ms)";
    }


    for (auto i = 0; i < optimizationOptions.maxIterations; ++i)
    {
        double chi2;
        double jtime = 0;
        {
            Saiga::ScopedTimer<double> timer(jtime);

            chi2 = computeQuadraticForm();
        }
        result.jtj_time += jtime;



        if (optimizationOptions.debugOutput && i == 0)
        {
            //                std::cout << "It fast " << i << ": " << 0.5 * current_chi2 << " -> " << 0.5 * chi2 <<
            //                std::endl;
            debug_output_table << i << chi2 << lambda << 0 << 0;
        }


        if (optimizationOptions.debug)
        {
            // A small sanity check in debug mode to see if compute cost is correct
            double test = computeCost();
            if (chi2 != test)
            {
                std::cerr << "Warning " << chi2 << "!=" << test << std::endl;
                SAIGA_ASSERT(chi2 == test);
            }
        }

        if (optimizationOptions.simple_solver)
        {
            if (chi2 > current_chi2)
            {
                // revert and break
                revertDelta();
                if (optimizationOptions.debugOutput)
                {
                    std::cout << "Early terminate  decrease " << std::endl;
                }
                break;
            }
            else if (chi2 + optimizationOptions.minChi2Delta > current_chi2)
            {
                if (optimizationOptions.debugOutput)
                {
                    std::cout << "Early terminate  " << std::endl;
                }
                break;
            }
            current_chi2 = chi2;
        }



        addLambda(lambda);
        if (i == 0)
        {
            current_chi2        = chi2;
            result.cost_initial = chi2;
        }
        result.cost_final = chi2;

        double ltime;
        {
            Saiga::ScopedTimer<double> timer(ltime);
            solveLinearSystem();
        }
        result.linear_solver_time += ltime;

        addDelta();

        if (optimizationOptions.simple_solver)
        {
            if (optimizationOptions.debugOutput)
            {
                debug_output_table << (i + 1) << chi2 << lambda << jtime << ltime;
            }
            continue;
        }


        double newChi2 = computeCost();

        if (std::isfinite(newChi2) && newChi2 < current_chi2)
        {
            // accept
            lambda       = lambda * (1.0 / 3.0);
            v            = 2;
            current_chi2 = newChi2;
        }
        else
        {
            // discard
            auto oldLambda = lambda;
            lambda         = lambda * v;
            v              = 2 * v;
            revertDelta();
            if (optimizationOptions.debugOutput)
            {
                std::cerr << "It " << (i + 1) << ": Invalid lm step. lambda: " << oldLambda << " -> " << lambda
                          << std::endl;
            }
        }

        if (optimizationOptions.debugOutput)
        {
            debug_output_table << (i + 1) << newChi2 << lambda << jtime << ltime;
        }

        if (std::abs(chi2 - newChi2) < optimizationOptions.minChi2Delta)
        {
            if (optimizationOptions.debugOutput)
            {
                std::cout << "Early terminate because |deltaChi2| < threshold" << std::endl;
            }
            break;
        }
    }
    finalize();

    result.cost_final = current_chi2;
    return result;
}

OptimizationResults LMOptimizer::initAndSolve()
{
    OptimizationResults result;

    lambda = optimizationOptions.initialLambda;

    {
        Saiga::ScopedTimer<double> timer(result.total_time);
        init();

        result = solve();
    }
    return result;
}

OptimizationResults LMOptimizer::solveOMP()
{
    SAIGA_ASSERT(supportOMP());
    double current_chi2 = 0;

    OptimizationResults result;
    result.linear_solver_time = 0;

    Saiga::ScopedTimer<double> timer(result.total_time);

    bool running = true;
    bool revert  = false;

// use this thread block for the complete optimizer
#pragma omp parallel num_threads(optimizationOptions.numThreads)
    {
        int tid = OMP::getThreadNum();
        for (auto i = 0; i < optimizationOptions.maxIterations && running; ++i)
        {
            double chi2;

            chi2 = computeQuadraticForm();

            //            continue;

            if (optimizationOptions.debug)
            {
                // A small sanity check in debug mode to see if compute cost is correct
                double test = computeCost();
                if (chi2 != test)
                {
                    std::cerr << "Warning " << chi2 << "!=" << test << std::endl;
                    SAIGA_ASSERT(chi2 == test);
                }
            }


            addLambda(lambda);

            if (i == 0)
            {
                current_chi2        = chi2;
                result.cost_initial = chi2;
            }
            result.cost_final = chi2;

            double ltime;
            {
                auto timer = (tid == 0) ? std::make_shared<Saiga::ScopedTimer<double>>(ltime) : nullptr;
                solveLinearSystem();
            }
            result.linear_solver_time += ltime;

            addDelta();

            double newChi2 = computeCost();


#pragma omp single
            {
                if (newChi2 < current_chi2)
                {
                    // accept
                    lambda       = lambda * (1.0 / 3.0);
                    v            = 2;
                    current_chi2 = newChi2;
                    revert       = false;
                }
                else
                {
                    // discard
                    auto oldLambda = lambda;
                    lambda         = lambda * v;
                    v              = 2 * v;

                    revert = true;
                    if (optimizationOptions.debugOutput)
                    {
                        std::cerr << "It " << i << ": Invalid lm step. lambda: " << oldLambda << " -> " << lambda
                                  << std::endl;
                    }
                }

                if (optimizationOptions.debugOutput)
                {
                    std::cout << "It " << i << ": " << 0.5 * chi2 << " -> " << 0.5 * newChi2 << std::endl;
                }

                if (std::abs(chi2 - newChi2) < optimizationOptions.minChi2Delta)
                {
                    if (optimizationOptions.debugOutput)
                    {
                        std::cout << "Early terminate because |deltaChi2| < threshold" << std::endl;
                    }
                    running = false;
                }
            }
            if (revert) revertDelta();
        }
        finalize();
    }

    result.cost_final = current_chi2;
    return result;
}

void LMOptimizer::initOMP()
{
    SAIGA_ASSERT(supportOMP());
    setThreadCount(optimizationOptions.numThreads);

    lambda = optimizationOptions.initialLambda;
    //#pragma omp parallel num_threads(optimizationOptions.numThreads)
    {
        init();
    }
}



}  // namespace Saiga
