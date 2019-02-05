/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/time/Time"
#include "saiga/core/util/assert.h"
#include "saiga/vision/recursiveMatrices/RecursiveMatrices.h"

namespace Saiga
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
};

/**
 * A solver for linear systems of equations. Ax=b
 * This class is spezialized for different structures of A.
 */
template <typename AType, typename XType, typename BType>
struct MixedRecursiveSolver
{
};


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
template <typename UBlock, typename VBlock, typename WType, typename WTType, typename XType>
class MixedRecursiveSolver<
    SymmetricMixedMatrix22<Eigen::DiagonalMatrix<UBlock, -1>, Eigen::DiagonalMatrix<VBlock, -1>, WType, WTType>, XType,
    XType>
{
   public:
    using AType =
        SymmetricMixedMatrix22<Eigen::DiagonalMatrix<UBlock, -1>, Eigen::DiagonalMatrix<VBlock, -1>, WType, WTType>;

    using AUType  = typename AType::UType;
    using AVType  = typename AType::VType;
    using AWType  = typename AType::WType;
    using AWTType = typename AType::WTType;

    using XUType = typename XType::UType;
    using XVType = typename XType::VType;

    using SType = Eigen::SparseMatrix<UBlock, Eigen::RowMajor>;

    void analyzePattern(AType& A)
    {
        n = A.u.rows();
        m = A.v.rows();

        Vinv.resize(m);
        Y.resize(n, m);
        Sdiag.resize(n);
        ej.resize(n);
        q.resize(m);
        S.resize(n, n);

        patternAnalyzed = true;
    }

    void solve(AType& A, XType& x, XType& b, const LinearSolverOptions& solverOptions)
    {
        // Some references for easier access
        const AUType& U   = A.u;
        const AVType& V   = A.v;
        const AWType& W   = A.w;
        const AWTType& WT = A.wt;
        XUType& da        = x.u;
        XVType& db        = x.v;
        const XUType& ea  = b.u;
        const XVType& eb  = b.v;

        if (!patternAnalyzed) analyzePattern(A);



        // compute schur
        for (int i = 0; i < m; ++i) Vinv.diagonal()(i) = V.diagonal()(i).get().inverse();
        Y = multSparseDiag(W, Vinv);

        if (solverOptions.buildExplizitSchur)
        {
            SAIGA_ASSERT(hasWT);
            S            = Y * WT;
            S            = -S;
            S.diagonal() = U.diagonal() + S.diagonal();
        }
        else
        {
            diagInnerProductTransposed(Y, W, Sdiag);
            Sdiag.diagonal() = U.diagonal() - Sdiag.diagonal();
        }
        ej = ea + -(Y * eb);

        if (solverOptions.solverType == LinearSolverOptions::SolverType::Direct)
        {
            SAIGA_ASSERT(solverOptions.buildExplizitSchur);
            Eigen::SparseMatrix<double> ssparse(n * UBlock::M::RowsAtCompileTime, n * UBlock::M::RowsAtCompileTime);
            {
                // Step 5
                // Solve the schur system for da
                // ~ 5.04%

                auto triplets = sparseBlockToTriplets(S);
                ssparse.setFromTriplets(triplets.begin(), triplets.end());
            }
            {
                //~61%
                Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
                solver.compute(ssparse);

                auto b                              = expand(ej);
                Eigen::Matrix<double, -1, 1> deltaA = solver.solve(b);
                // copy back into da
                for (int i = 0; i < n; ++i)
                {
                    da(i) = deltaA.segment<UBlock::M::RowsAtCompileTime>(i * UBlock::M::RowsAtCompileTime);
                }
            }
        }
        else
        {
            // solve
            da.setZero();
            RecursiveDiagonalPreconditioner<UBlock> P;
            Eigen::Index iters = solverOptions.maxIterativeIterations;
            double tol         = solverOptions.iterativeTolerance;


            if (solverOptions.buildExplizitSchur)
            {
                P.compute(S);
                XUType tmp(n);
                recursive_conjugate_gradient(
                    [&](const XUType& v) {
                        tmp = S * v;
                        return tmp;
                    },
                    ej, da, P, iters, tol);
            }
            else
            {
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
                //            cout << "error " << tol << " iterations " << iters << endl;
            }
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
    WType Y;
    SType S;
    Eigen::DiagonalMatrix<UBlock, -1> Sdiag;
    XUType ej;

    bool patternAnalyzed = false;
    bool hasWT           = true;
};


}  // namespace Saiga
