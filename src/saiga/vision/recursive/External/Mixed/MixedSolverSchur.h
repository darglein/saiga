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

namespace Eigen::Recursive
{
/**
 * A spezialized solver for BundleAjustment-like problems.
 * The structure is a follows:
 *
 * | U  W |
 * | WT V |
 *
 * Where,
 * U : Diagonalmatrix
 * V : Diagonalmatrix
 * W : Sparsematrix
 *
 * This solver computes the schur complement on U and solves the reduced system with CG.
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

    //    using S1Type = Eigen::Matrix<UBlock, -1,-1,Eigen::RowMajor>;
    using S1Type = Eigen::SparseMatrix<UBlock, Eigen::RowMajor>;

    //    using LDLT = Eigen::RecursiveSimplicialLDLT<S1Type, Eigen::Upper>;


    using InnerSolver1 = MixedSymmetricRecursiveSolver<S1Type, XUType>;

    void resize(int n, int m)
    {
        this->n = n;
        this->m = m;

        Vinv.resize(m);
        Y.resize(n, m);
        Sdiag.resize(n);
        ej.resize(n);
        q.resize(m);
        S1.resize(n, n);
        P.resize(n);
        tmp.resize(n);
    }

    void analyzePattern(const AType& A, const LinearSolverOptions& solverOptions)
    {
        resize(A.u.rows(), A.v.rows());

        if (solverOptions.solverType == LinearSolverOptions::SolverType::Direct)
        {
            hasWT         = true;
            explizitSchur = true;
        }
        else
        {
            // TODO: add heurisitc here
            hasWT = true;
            if (solverOptions.buildExplizitSchur)
                explizitSchur = true;
            else
                explizitSchur = false;
        }

        if (hasWT)
        {
            transposeStructureOnly(A.w, WT);
        }

        patternAnalyzed = true;
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

        if (hasWT)
        {
            transposeValueOnly(A.w, WT);
        }

#if 1
        // U schur (S1)
        for (int i = 0; i < m; ++i) Vinv.diagonal()(i) = V.diagonal()(i).get().inverse();
        multSparseDiag(W, Vinv, Y);

        if (explizitSchur)
        {
            eigen_assert(hasWT);
            S1            = (Y * WT).template triangularView<Eigen::Upper>();
            S1            = -S1;
            S1.diagonal() = U.diagonal() + S1.diagonal();

            //            S2            = (Y * WT).template triangularView<Eigen::Upper>();
            //            S2            = -S2;
            //            S2.diagonal() = U.diagonal() + S2.diagonal();

            //            std::cout << expand(S1) << std::endl << std::endl;
            //            std::cout << expand(S2) << std::endl << std::endl;
        }
        else
        {
            diagInnerProductTransposed(Y, W, Sdiag);
            Sdiag.diagonal() = U.diagonal() - Sdiag.diagonal();
        }

        //        std::cout << expand(Sdiag.toDenseMatrix()) << std::endl << std::endl;
        //        exit(0);
        ej = ea + -(Y * eb);
        da.setZero();

        //        exit(0);

        // A special implicit schur solver.
        // We cannot use the recursive inner solver here.
        // (Maybe a todo for the future)
        Eigen::Index iters = solverOptions.maxIterativeIterations;
        double tol         = solverOptions.iterativeTolerance;


        if (explizitSchur)
        {
            P.compute(S1);
            //            std::cout << expand(P.m_invdiag) << std::endl << std::endl;
        }
        else
        {
            P.compute(Sdiag);
            //            std::cout << expand(P.m_invdiag) << std::endl << std::endl;
        }

        XUType tmp(n);


        recursive_conjugate_gradient(
            [&](const XUType& v, XUType& result) {
                // x = U * p - Y * WT * p
                if (explizitSchur)
                {
                    //                    if constexpr (denseSchur)
                    //                        denseMV(S1, v, result);
                    //                    else
                    result = S1.template selfadjointView<Eigen::Upper>() * v;
                    //                    std::cout << expand(result) << std::endl << std::endl;
                }
                else
                {
                    if (hasWT)
                    {
                        tmp = Y * (WT * v);
                    }
                    else
                    {
                        multSparseRowTransposedVector(W, v, q);
                        tmp = Y * q;
                    }
                    result = (U.diagonal().array() * v.array()) - tmp.array();
                    //                    std::cout << expand(result) << std::endl << std::endl;
                }
            },
            ej, da, P, iters, tol);



        // finalize
        if (hasWT)
        {
            q = WT * da;
        }
        else
        {
            multSparseRowTransposedVector(W, da, q);
        }
        q  = eb - q;
        db = multDiagVector(Vinv, q);
#else
        // V schur (S1)
        for (int i = 0; i < m; ++i) Vinv.diagonal()(i) = V.diagonal()(i).get().inverse();
        Y = multSparseDiag(W, Vinv);
#endif
    }

   private:
    int n, m;

    // ==== Solver tmps ====
    XVType q;
    AVType Vinv;
    AWType Y;
    Eigen::DiagonalMatrix<UBlock, -1> Sdiag;
    XUType ej;
    XUType tmp;

    std::vector<int> transposeTargets;
    AWTType WT;

    RecursiveDiagonalPreconditioner<UBlock> P;

    S1Type S1;
    InnerSolver1 solver1;
    //    InnerSolver2 solver2;

    bool patternAnalyzed = false;
    bool hasWT           = true;
    bool explizitSchur   = true;
};


}  // namespace Eigen::Recursive
