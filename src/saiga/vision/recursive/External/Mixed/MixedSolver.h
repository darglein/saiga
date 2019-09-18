/**
 * This file is part of the Eigen Recursive Matrix Extension (ERME).
 *
 * Copyright (c) 2019 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "../Core.h"
#include "MixedMatrix.h"
namespace Eigen::Recursive
{
struct LinearSolverOptions
{
    // Base Options used by almost every solver
    enum class SolverType : int
    {
        Iterative = 0,
        Direct    = 1
    };
    SolverType solverType      = SolverType::Iterative;
    int maxIterativeIterations = 50;
    double iterativeTolerance  = 1e-5;

    // Schur complement options (not used by every solver)
    bool buildExplizitSchur = false;

    // Well the cholmod supernodal ist extremly fast
    // -> Maybe in the future when I have implemented a supernodal recursive factorization
    //      I switch it back to false ;)
    bool cholmod = true;
};

/**
 * A solver for linear systems of equations. Ax=b
 * This class is spezialized for different structures of A.
 */
template <typename AType, typename XType>
struct MixedSymmetricRecursiveSolver
{
};


}  // namespace Eigen::Recursive
