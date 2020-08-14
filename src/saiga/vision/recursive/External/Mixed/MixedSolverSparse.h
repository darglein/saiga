/**
 * This file is part of the Eigen Recursive Matrix Extension (ERME).
 *
 * Copyright (c) 2019 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "../Cholesky.h"
#include "../Core.h"
#include "MixedSolver.h"

#include <memory>
#include <vector>
#if __has_include("cholmod.h")
#    define SOLVER_USE_CHOLMOD
#    include "Eigen/CholmodSupport"
#endif

namespace Eigen::Recursive
{
/**
 * A solver for sparse matrices. The elements of A should be easily invertible.
 */
template <typename T, int _Options, typename XType>
class MixedSymmetricRecursiveSolver<Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<T>, _Options>, XType>
{
   public:
    using AType = typename Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<T>, _Options>;
    using LDLT  = Eigen::RecursiveSimplicialLDLT<AType, Eigen::Upper>;

    using ExpandedType = Eigen::SparseMatrix<typename T::Scalar, Eigen::RowMajor>;
#ifdef SOLVER_USE_CHOLMOD
    using CholmodLDLT = Eigen::CholmodSupernodalLLT<ExpandedType, Eigen::Upper>;
    //        using CholmodLDLT = Eigen::CholmodSimplicialLDLT<ExpandedType, Eigen::Upper>;
    //        using CholmodLDLT = Eigen::RecursiveSimplicialLDLT<ExpandedType, Eigen::Upper>;
    //    using CholmodLDLT = Eigen::SimplicialLLT<ExpandedType, Eigen::Upper>;
#endif

    void Init()
    {
        ldlt = nullptr;
#ifdef SOLVER_USE_CHOLMOD
        cholmodldlt = nullptr;
#endif
    }

    void solve(AType& A, XType& x, XType& b, const LinearSolverOptions& solverOptions = LinearSolverOptions())
    {
        int n = A.rows();
        if (solverOptions.solverType == LinearSolverOptions::SolverType::Direct)
        {
#ifdef SOLVER_USE_CHOLMOD
            // Use Cholmod's supernodal factorization for very large or very dense matrices.
            //            double density = A.nonZeros() / (double(A.rows()) * A.cols());
            // bool useCholmod = A.rows() > 1000 || density > 0.1;
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
#endif
            {
                if (!ldlt)
                {
#if 0
                    {
                        // Use cholmod to compute the ordering
                        cholmod_common m_cholmod;
                        cholmod_start(&m_cholmod);
                        cholmod_defaults(&m_cholmod);

                        cholmod_sparse cholmod_matrix;

                        const auto& testasdf = reinterpret_cast<Eigen::SparseMatrix<double>&>(A);
                        cholmod_matrix       = Eigen::viewAsCholmod(testasdf.selfadjointView<Eigen::Upper>());

                        m_cholmod.supernodal         = CHOLMOD_SIMPLICIAL;
                        m_cholmod.nmethods           = 1;
                        m_cholmod.method[0].ordering = CHOLMOD_AMD;
                        m_cholmod.postorder          = false;

                        orderingFull.resize(A.rows());
                        cholmod_amd(&cholmod_matrix, 0, 0, orderingFull.data(), &m_cholmod);


                        permFull.resize(A.rows());
                        for (int i = 0; i < A.rows(); ++i)
                        {
                            permFull.indices()[i] = orderingFull[i];
                        }

                        cholmod_finish(&m_cholmod);
                        cholmod_free_work(&m_cholmod);
                    }
#endif

                    // Create cholesky solver and do a full compute
                    ldlt = std::make_unique<LDLT>();
#if 0
                    ldlt->m_Pinv = permFull;
                    ldlt->m_P    = permFull.inverse();
                    ldlt->analyzePattern(A);
                    ldlt->factorize(A);
#else

                    ldlt->compute(A);
#endif
                }
                else
                {
                    // This line computes the factorization without analyzing the structure again
                    ldlt->factorize(A);
                }
                //                std::cout << "ldlt compute" << std::endl;
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
                [&](const XType& v, XType& result) { result = A.template selfadjointView<Eigen::Upper>() * v; }, b, x,
                P, iters, tol);
        }
    }

   private:
    std::unique_ptr<LDLT> ldlt;
    Eigen::PermutationMatrix<-1> permFull;
    std::vector<int> orderingFull;
#ifdef SOLVER_USE_CHOLMOD
    // Cholmod stuff
    std::unique_ptr<CholmodLDLT> cholmodldlt;
    std::unique_ptr<ExpandedType> expandS;
#else
    std::unique_ptr<int> dummy;
    std::unique_ptr<int> dummy2;
#endif
};

}  // namespace Eigen::Recursive
