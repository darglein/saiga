/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <string>
namespace Saiga
{
struct SAIGA_VISION_API OptimizationResults
{
    std::string name = "Optimizer";

    double cost_initial = 0;
    double cost_final   = 0;

    double init_time          = 0;
    double linear_solver_time = 0;
    double jtj_time           = 0;
    double total_time         = 0;

    bool success = false;
};

SAIGA_VISION_API std::ostream& operator<<(std::ostream& strm, const OptimizationResults& op);

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
    bool buildExplizitSchur    = false;

    // early termiante if the chi2 delta is smaller than this value
    double minChi2Delta  = 1e-5;
    double initialLambda = 1.00e-04;
    int numThreads       = 4;

    // Tests if the chi2 has actually decreased.
    // Strongly recommended to set this to true.
    // Otherwise the optimizer might decrease the quality
    bool simple_solver = false;

    bool debugOutput = false;
    bool debug       = false;

    void imgui();
};

SAIGA_VISION_API std::ostream& operator<<(std::ostream& strm, const OptimizationOptions& op);

class SAIGA_VISION_API Optimizer
{
   public:
    Optimizer() {}
    virtual OptimizationResults solve() { return initAndSolve(); }
    virtual OptimizationResults initAndSolve() = 0;

    OptimizationOptions optimizationOptions;
};


class SAIGA_VISION_API LMOptimizer : public Optimizer
{
   public:
    LMOptimizer() {}
    virtual OptimizationResults solve() override;
    virtual OptimizationResults initAndSolve() override;

    // multithreaded version
    // only works if the problem supports omp
    OptimizationResults solveOMP();
    void initOMP();

    //    virtual OptimizationResults solve() = 0;
   protected:
    virtual void init()                   = 0;
    virtual double computeQuadraticForm() = 0;
    virtual void addLambda(double lambda) = 0;
    virtual bool addDelta()               = 0;
    virtual void revertDelta()            = 0;
    virtual void solveLinearSystem()      = 0;
    virtual double computeCost()          = 0;
    virtual void finalize()               = 0;

    virtual void setThreadCount(int n) {}
    virtual bool supportOMP() { return false; }

    double lambda;
    double v = 2;
};

}  // namespace Saiga
