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
template <typename AType, typename XType>
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
template <typename UBlock, typename VBlock, typename WBlock, typename XType>
class MixedRecursiveSolver<SymmetricMixedMatrix2<Eigen::DiagonalMatrix<UBlock, -1>, Eigen::DiagonalMatrix<VBlock, -1>,
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
            S            = Y * WT;
            S            = -S;
            S.diagonal() = U.diagonal() + S.diagonal();
            //            cout << "S" << endl << expand(S) << endl << endl;
        }
        else
        {
            diagInnerProductTransposed(Y, W, Sdiag);
            Sdiag.diagonal() = U.diagonal() - Sdiag.diagonal();
        }
        ej = ea + -(Y * eb);
        //        cout << "ej" << endl << expand(ej) << endl << endl;

        if (solverOptions.solverType == LinearSolverOptions::SolverType::Direct)
        {
            // Direct solver using cholesky factorization
            SAIGA_ASSERT(explizitSchur);

            if (!ldlt)
            {
                // Create cholesky solver and do a full compute
                ldlt = std::make_unique<LDLT>();
                ldlt->compute(S);
            }
            else
            {
                // This line computes the factorization without analyzing the structure again
                ldlt->factorize(S);
            }

            da = ldlt->solve(ej);
        }
        else
        {
            // solve
            da.setZero();
            RecursiveDiagonalPreconditioner<UBlock> P;
            Eigen::Index iters = solverOptions.maxIterativeIterations;
            double tol         = solverOptions.iterativeTolerance;


            if (explizitSchur)
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
    AWType Y;
    SType S;
    Eigen::DiagonalMatrix<UBlock, -1> Sdiag;
    XUType ej;

    AWTType WT;

    std::unique_ptr<LDLT> ldlt;

    bool patternAnalyzed = false;
    bool hasWT           = true;
    bool explizitSchur   = true;
};


}  // namespace Saiga
