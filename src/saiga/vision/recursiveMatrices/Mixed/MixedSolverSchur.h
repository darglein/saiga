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

namespace Saiga
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

    using SType = Eigen::SparseMatrix<UBlock, Eigen::RowMajor>;

    using LDLT = Eigen::RecursiveSimplicialLDLT<SType, Eigen::Upper>;


    using InnerSolver = MixedSymmetricRecursiveSolver<SType, XUType>;

    void analyzePattern(const AType& A, const LinearSolverOptions& solverOptions)
    {
        n = A.u.rows();
        m = A.v.rows();

        Vinv.resize(m);
        Y.resize(n, m);
        Sdiag.resize(n);
        ej.resize(n);
        q.resize(m);
        S.resize(n, n);


        if (solverOptions.solverType == LinearSolverOptions::SolverType::Direct)
        {
            hasWT         = true;
            explizitSchur = true;
        }
        else
        {
            // TODO: add heurisitc here
            hasWT         = true;
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
            //            transpose(A.w, WT);
            //            cout << expand(A.w) << endl << endl;
            //            cout << expand(WT) << endl << endl;
            //            cout << A.w.rows() << "x" << A.w.cols() << endl;
            //            cout << WT.rows() << "x" << WT.cols() << endl;
        }



        // compute schur
        for (int i = 0; i < m; ++i) Vinv.diagonal()(i) = V.diagonal()(i).get().inverse();
        Y = multSparseDiag(W, Vinv);

        //        cout << "Vinv" << endl << expand(Vinv) << endl << endl;
        //        cout << "Y" << endl << expand(Y) << endl << endl;

        if (explizitSchur)
        {
            SAIGA_ASSERT(hasWT);
            // (S is symmetric)
            S = (Y * WT).template triangularView<Eigen::Upper>();
            //            S            = (Y * WT);
            S            = -S;
            S.diagonal() = U.diagonal() + S.diagonal();
            //            cout << "S" << endl << expand(S) << endl << endl;

            //            double S_density = S.nonZeros() / double(S.rows() * S.cols());
            //            cout << "S density: " << S_density << endl;
        }
        else
        {
            diagInnerProductTransposed(Y, W, Sdiag);
            Sdiag.diagonal() = U.diagonal() - Sdiag.diagonal();
        }
        ej = ea + -(Y * eb);
        //        cout << "ej" << endl << expand(ej) << endl << endl;

        if (solverOptions.solverType == LinearSolverOptions::SolverType::Iterative && !explizitSchur)
        {
            // A special implicit schur solver.
            // We cannot use the recursive inner solver here.
            // (Maybe a todo for the future)
            da.setZero();
            RecursiveDiagonalPreconditioner<UBlock> P;
            Eigen::Index iters = solverOptions.maxIterativeIterations;
            double tol         = solverOptions.iterativeTolerance;

            P.compute(Sdiag);
            XUType tmp(n);
            recursive_conjugate_gradient(
                [&](const XUType& v) {
                    // x = U * p - Y * WT * p
                    if (hasWT)
                    {
                        tmp = Y * (WT * v);
                    }
                    else
                    {
                        multSparseRowTransposedVector(W, v, q);
                        tmp = Y * q;
                    }
                    tmp = (U.diagonal().array() * v.array()) - tmp.array();
                    return tmp;
                },
                ej, da, P, iters, tol);
        }
        else
        {
            solver.solve(S, da, ej, solverOptions);
        }

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
    }

   private:
    int n, m;

    // ==== Solver tmps ====
    XVType q;
    AVType Vinv;
    AWType Y;
    SType S;
    Eigen::DiagonalMatrix<UBlock, -1> Sdiag;
    XUType ej;

    AWTType WT;

    InnerSolver solver;

    bool patternAnalyzed = false;
    bool hasWT           = true;
    bool explizitSchur   = true;
};


}  // namespace Saiga
