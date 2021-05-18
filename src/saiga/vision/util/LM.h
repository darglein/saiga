/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "../recursive/Recursive.h"

namespace Saiga
{
template <typename T>
void applyLMDiagonalInner(T& diag, double lambda = 1.00e-04, double min_lm_diagonal = 1e-6,
                          double max_lm_diagonal = 1e32)
{
    for (int k = 0; k < diag.rows(); ++k)
    {
        auto& value = diag.diagonal()(k);
        value       = value + lambda * value;
        value       = clamp(value, min_lm_diagonal, max_lm_diagonal);
    }
}


/**
 * Applies the Levenberg Marquarad Diagonal update to a recursive diagonal matrix.
 *
 * U = U + clamp(diag(U) * lambda,min,max)
 */
template <typename T>
void applyLMDiagonal(Eigen::DiagonalMatrix<T, -1>& U, double lambda = 1.00e-04, double min_lm_diagonal = 1e-6,
                     double max_lm_diagonal = 1e32)
{
    for (int i = 0; i < U.rows(); ++i)
    {
        auto& diag = U.diagonal()(i).get();
        applyLMDiagonalInner(diag, lambda, min_lm_diagonal, max_lm_diagonal);
    }
}
template <typename T>
void applyLMDiagonal_omp(Eigen::DiagonalMatrix<T, -1>& U, double lambda = 1.00e-04, double min_lm_diagonal = 1e-6,
                         double max_lm_diagonal = 1e32)
{
#pragma omp for
    for (int i = 0; i < U.rows(); ++i)
    {
        auto& diag = U.diagonal()(i).get();
        applyLMDiagonalInner(diag, lambda, min_lm_diagonal, max_lm_diagonal);
    }
}
/**
 * Simplified LM diagonal update, used by the g2o framwork
 *
 * U = U + ID * lambda
 */
template <typename T>
void applyLMDiagonalG2O(Eigen::DiagonalMatrix<T, -1>& U, double lambda = 1.00e-04)
{
    for (int i = 0; i < U.rows(); ++i)
    {
        auto& diag = U.diagonal()(i).get();

        for (int k = 0; k < diag.RowsAtCompileTime; ++k)
        {
            auto& value = diag.diagonal()(k);
            value       = value + lambda;
        }
    }
}

inline void updateLambda(double& lambda, bool success)
{
    if (success)
    {
        lambda /= 2.0;
    }
    else
    {
        lambda *= 2.0;
    }
}

}  // namespace Saiga
