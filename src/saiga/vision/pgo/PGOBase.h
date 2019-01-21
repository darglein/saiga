/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/scene/PoseGraph.h"


namespace Saiga
{
struct SAIGA_GLOBAL PGOOptions
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

    bool debugOutput = false;

    void imgui();
};

/**
 * @brief The BABase class
 *
 * Base class and interface for all BA implementations.
 */
class SAIGA_GLOBAL PGOBase
{
   public:
    PGOBase(const std::string& name) : name(name) {}
    virtual void solve(PoseGraph& scene, const PGOOptions& options) = 0;

    std::string name;
};


}  // namespace Saiga
