/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"


namespace Saiga
{
struct SAIGA_VISION_API OptimizationResults
{
    std::string name;

    double cost_initial;
    double cost_final;

    double linear_solver_time;
    double total_time;

    bool success;
};

struct SAIGA_VISION_API OptimizationOptions
{
    int maxIterations = 10;

    enum class SolverType : int
    {
        Iterative = 0,
        Direct    = 1
    };
    SolverType solverType = SolverType::Iterative;

    int maxIterativeIterations = 50;
    double iterativeTolerance  = 1e-5;

    double initialLambda = 1.00e-04;

    bool debugOutput = false;

    void imgui();
};

SAIGA_VISION_API std::ostream& operator<<(std::ostream& strm, const OptimizationOptions& op);

class SAIGA_VISION_API Optimizer
{
   public:
    Optimizer() {}
    virtual OptimizationResults solve() = 0;

    OptimizationOptions optimizationOptions;
};


class SAIGA_VISION_API LMOptimizer : public Optimizer
{
   public:
    LMOptimizer() {}
    virtual OptimizationResults solve() override;

    //    virtual OptimizationResults solve() = 0;
   protected:
    virtual void init()                   = 0;
    virtual double computeQuadraticForm() = 0;
    virtual void addLambda(double lambda) = 0;
    virtual void addDelta()               = 0;
    virtual void revertDelta()            = 0;
    virtual void solveLinearSystem()      = 0;
    virtual double computeCost()          = 0;
    virtual void finalize()               = 0;

    double lambda;
    double v = 2;
};

}  // namespace Saiga
