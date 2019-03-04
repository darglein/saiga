/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/time/Time"
#include "saiga/core/util/assert.h"
#include "saiga/vision/recursiveMatrices/RecursiveMatrices.h"
#include "saiga/vision/recursiveMatrices/RecursiveSimplicialCholesky.h"

#include "Eigen/CholmodSupport"
namespace Saiga
{
/**
 * A solver for sparse matrices. The elements of A should be easily invertible.
 */
template <typename T, int _Options, typename XType>
class MixedSymmetricRecursiveSolver<Eigen::SparseMatrix<Saiga::MatrixScalar<T>, _Options>, XType>
{
   public:
    using AType = typename Eigen::SparseMatrix<Saiga::MatrixScalar<T>, _Options>;
    using LDLT  = Eigen::RecursiveSimplicialLDLT<AType, Eigen::Upper>;

    using ExpandedType = Eigen::SparseMatrix<typename T::Scalar, Eigen::RowMajor>;
    using CholmodLDLT  = Eigen::CholmodSupernodalLLT<ExpandedType, Eigen::Upper>;

    void solve(AType& A, XType& x, XType& b, const LinearSolverOptions& solverOptions = LinearSolverOptions())
    {
        int n = A.rows();
        if (solverOptions.solverType == LinearSolverOptions::SolverType::Direct)
        {
            if (solverOptions.cholmod)
            {
                if (!expandS) expandS = std::make_unique<ExpandedType>();
                sparseBlockToFlatMatrix(A, *expandS);
                auto eb = expand(b);
                if (!cholmodldlt)
                {
                    // Create cholesky solver and do a full compute
                    cholmodldlt = std::make_unique<CholmodLDLT>();
                    cholmodldlt->compute(*expandS);
                }
                else
                {
                    // This line computes the factorization without analyzing the structure again
                    cholmodldlt->factorize(*expandS);
                }
                Eigen::Matrix<double, -1, 1> ex = cholmodldlt->solve(eb);
                // convert back to block x
                for (int i = 0; i < x.rows(); ++i)
                {
                    x(i).get() = ex.segment(i * T::RowsAtCompileTime, T::RowsAtCompileTime);
                }
            }
            else
            {
                if (!ldlt)
                {
                    // Create cholesky solver and do a full compute
                    ldlt = std::make_unique<LDLT>();
                    ldlt->compute(A);
                }
                else
                {
                    // This line computes the factorization without analyzing the structure again
                    ldlt->factorize(A);
                }
                x = ldlt->solve(b);
            }
        }
        else
        {
            x.setZero();
            RecursiveDiagonalPreconditioner<MatrixScalar<T>> P;
            Eigen::Index iters = solverOptions.maxIterativeIterations;
            double tol         = solverOptions.iterativeTolerance;

            P.compute(A);

            XType tmp(n);
            recursive_conjugate_gradient(
                [&](const XType& v) {
                    tmp = A.template selfadjointView<Eigen::Upper>() * v;
                    return tmp;
                },
                b, x, P, iters, tol);
        }
    }

   private:
    std::unique_ptr<LDLT> ldlt;

    // Cholmod stuff
    std::unique_ptr<CholmodLDLT> cholmodldlt;
    std::unique_ptr<ExpandedType> expandS;
};

}  // namespace Saiga
