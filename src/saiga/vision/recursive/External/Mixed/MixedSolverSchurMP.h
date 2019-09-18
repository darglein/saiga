/**
 * This file is part of the Eigen Recursive Matrix Extension (ERME).
 *
 * Copyright (c) 2019 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "../Core.h"
#include "MixedSolver.h"

namespace Eigen::Recursive
{
/**
 * Multi threaded implementation.
 */
template <typename UBlock, typename VBlock, typename WBlock, typename XType>
class MixedSymmetricRecursiveSolver<
    SymmetricMixedMatrix2<Eigen::DiagonalMatrix<UBlock, -1>, Eigen::DiagonalMatrix<VBlock, -1>,
                          Eigen::SparseMatrix<WBlock, Eigen::RowMajor>>,
    XType>
{
   public:
    using AType = SymmetricMixedMatrix2<Eigen::DiagonalMatrix<UBlock, -1>, Eigen::DiagonalMatrix<VBlock, -1>,
                                        Eigen::SparseMatrix<WBlock, Eigen::RowMajor>>;

    using AUType = typename AType::UType;
    using AVType = typename AType::VType;
    using AWType = typename AType::WType;

    using AWTType = typename TransposeType<AWType>::Type;

    using XUType = typename XType::UType;
    using XVType = typename XType::VType;

    using S1Type = Eigen::SparseMatrix<UBlock, Eigen::RowMajor>;
    using S2Type = Eigen::SparseMatrix<VBlock, Eigen::RowMajor>;

    void analyzePattern(const AType& A, const LinearSolverOptions& solverOptions)
    {
#pragma omp single
        {
            n = A.u.rows();
            m = A.v.rows();

            Vinv.resize(m);
            Y.resize(n, m);
            Sdiag.resize(n);
            ej.resize(n);
            q.resize(m);
            S1.resize(n, n);
            P.resize(n);
            tmp.resize(n);


            if (solverOptions.solverType == LinearSolverOptions::SolverType::Direct)
            {
                std::terminate();
                hasWT         = true;
                explizitSchur = true;
            }
            else
            {
                // TODO: add heurisitc here
                hasWT         = true;
                explizitSchur = true;
            }

            if (hasWT)
            {
                transposeStructureOnly_omp(A.w, WT, transposeTargets);
            }

            patternAnalyzed = true;
        }
    }


    void solve(AType& A, XType& x, XType& b, const LinearSolverOptions& solverOptions = LinearSolverOptions())
    {
        // Some references for easier access
        const AUType& U  = A.u;
        const AVType& V  = A.v;
        const AWType& W  = A.w;
        XUType& da       = x.u;
        XVType& db       = x.v;
        const XUType& ea = b.u;
        const XVType& eb = b.v;



        if (!patternAnalyzed) analyzePattern(A, solverOptions);


        transposeValueOnly_omp(A.w, WT, transposeTargets);
        // U schur (S1)
#pragma omp for
        for (int i = 0; i < m; ++i) Vinv.diagonal()(i) = V.diagonal()(i).get().inverse();


        multSparseDiag_omp(W, Vinv, Y);
        diagInnerProductTransposed_omp(Y, W, Sdiag);

#pragma omp for
        for (int i = 0; i < n; ++i) Sdiag.diagonal()(i).get() = U.diagonal()(i).get() - Sdiag.diagonal()(i).get();


        sparse_mv_omp(Y, eb, ej);

#pragma omp for
        for (int i = 0; i < n; ++i)
        {
            ej(i).get() = ea(i).get() - ej(i).get();
            da(i).get().setZero();
        }


        {
            // A special implicit schur solver.
            // We cannot use the recursive inner solver here.
            // (Maybe a todo for the future)
            //            da.setZero();
        }

        Eigen::Index iters = solverOptions.maxIterativeIterations;
        double tol         = solverOptions.iterativeTolerance;
        P.compute(Sdiag);



        recursive_conjugate_gradient_OMP(
            [&](const XUType& v, XUType& result) {
                // x = U * p - Y * WT * p
                sparse_mv_omp(WT, v, q);
                sparse_mv_omp(Y, q, tmp);
#pragma omp for
                for (int i = 0; i < v.rows(); ++i)
                {
                    result(i).get() = (U.diagonal()(i).get() * v(i).get()) - tmp(i).get();
                }
            },
            ej, da, P, iters, tol);


        sparse_mv_omp(WT, da, q);

        {
#pragma omp for
            for (int i = 0; i < m; ++i)
            {
                q(i).get() = eb(i).get() - q(i).get();
            }
        }
        multDiagVector_omp(Vinv, q, db);
    }

   private:
    int n, m;

    // ==== Solver tmps ====
    XVType q;
    AVType Vinv;
    AWType Y;
    S1Type S1;
    Eigen::DiagonalMatrix<UBlock, -1> Sdiag;
    XUType ej;
    XUType tmp;

    std::vector<int> transposeTargets;
    AWTType WT;

    RecursiveDiagonalPreconditioner<UBlock> P;

    bool patternAnalyzed = false;
    bool hasWT           = true;
    bool explizitSchur   = true;
};


}  // namespace Eigen::Recursive
