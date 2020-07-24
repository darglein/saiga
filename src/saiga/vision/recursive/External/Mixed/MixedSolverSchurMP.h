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
namespace Eigen::Recursive
{
// Schur Solver for the following format:
//
//   | U   W  |  |x1|    |a|
//   | WT  V  |  |x2| =  |b|
//
// U: Sparse Block
// V: Block Diagonal
// W: Sparse Block
//
// Solution: (https://en.wikipedia.org/wiki/Schur_complement)
//
// Schur Matrix
//    S = U - W * V^-1 * WT
// Right Hand Side of Schur System
//   r1 = a - W * V^-1 * b
// Solve for x1
//    S * x1 = r1
// Solve for x2
//   r2 = b - WT * x1
//   x2 = V^-1 * r
//
//
template <typename UBlock, typename VBlock, typename WBlock, typename XType>
class MixedSymmetricRecursiveSolver<
    SymmetricMixedMatrix2<Eigen::SparseMatrix<UBlock, Eigen::RowMajor>, Eigen::DiagonalMatrix<VBlock, -1>,
                          Eigen::SparseMatrix<WBlock, Eigen::RowMajor>>,
    XType>
{
   public:
    using AType = SymmetricMixedMatrix2<Eigen::SparseMatrix<UBlock, Eigen::RowMajor>, Eigen::DiagonalMatrix<VBlock, -1>,
                                        Eigen::SparseMatrix<WBlock, Eigen::RowMajor>>;

    using AUType = typename AType::UType;
    using AVType = typename AType::VType;
    using AWType = typename AType::WType;

    using AWTType = typename TransposeType<AWType>::Type;

    using XUType = typename XType::UType;
    using XVType = typename XType::VType;

    using S1Type = Eigen::SparseMatrix<UBlock, Eigen::RowMajor>;
    using S2Type = Eigen::SparseMatrix<VBlock, Eigen::RowMajor>;

    using LDLT         = Eigen::RecursiveSimplicialLDLT<S1Type, Eigen::Upper>;
    using InnerSolver1 = MixedSymmetricRecursiveSolver<S1Type, XUType>;


    void resize(int n, int m)
    {
        this->n = n;
        this->m = m;

        Vinv.resize(m);
        Y.resize(n, m);
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
            ldlt          = nullptr;
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

        // Invert V
        for (int i = 0; i < m; ++i)
        {
            //            Vinv.diagonal()(i) = V.diagonal()(i).get().inverse();

            Vinv.diagonal()(i) = Recursive::inverseCholesky(V.diagonal()(i));
        }

        // Y = W * V^-1
        multSparseDiag(W, Vinv, Y);


        eigen_assert(hasWT);
        // S = U - W * V^-1 * WT
        S1 = (Y * WT).template triangularView<Eigen::Upper>();
        S1 = U - S1;

        // r = a - W * V^-1 * b
        ej = ea + -(Y * eb);


        if (solverOptions.solverType == LinearSolverOptions::SolverType::Direct)
        {
            // Direct recusive ldlt solver
            if (!ldlt)
            {
                ldlt = std::make_unique<LDLT>();
                ldlt->compute(S1);
            }
            else
            {
                ldlt->factorize(S1);
            }
            da = ldlt->solve(ej);
        }
        else
        {
            P.compute(S1);

            da.setZero();

            // Iterative CG solver
            Eigen::Index iters = solverOptions.maxIterativeIterations;
            double tol         = solverOptions.iterativeTolerance;
            //            XUType tmp(n);

            recursive_conjugate_gradient(
                [&](const XUType& v, XUType& result) {
                    // x = U * p - Y * WT * p
                    result = S1.template selfadjointView<Eigen::Upper>() * v;
                },
                ej, da, P, iters, tol);
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
    XUType ej;
    XUType tmp;

    std::vector<int> transposeTargets;
    AWTType WT;

    RecursiveDiagonalPreconditioner<UBlock> P;
    S1Type S1;
    //    InnerSolver1 solver1;

    std::unique_ptr<LDLT> ldlt;

    bool patternAnalyzed = false;
    bool hasWT           = true;
    bool explizitSchur   = true;
};



// Schur Solver for the following format:
//
//   | U   W  |  |x1|    |a|
//   | WT  V  |  |x2| =  |b|
//
// U: Block Diagonal
// V: Block Diagonal
// W: Sparse Block
//
// Solution: (https://en.wikipedia.org/wiki/Schur_complement)
//
// Schur Matrix
//    S = U - W * V^-1 * WT
// Right Hand Side of Schur System
//   r1 = a - W * V^-1 * b
// Solve for x1
//    S * x1 = r1
// Solve for x2
//   r2 = b - WT * x1
//   x2 = V^-1 * r
//
//
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

    using LDLT         = Eigen::RecursiveSimplicialLDLT<S1Type, Eigen::Upper>;
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
            ldlt          = nullptr;
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

        // Invert V
        for (int i = 0; i < m; ++i) Vinv.diagonal()(i) = V.diagonal()(i).get().inverse();

        // Y = W * V^-1
        multSparseDiag(W, Vinv, Y);

        if (explizitSchur)
        {
            eigen_assert(hasWT);
            // S = U - W * V^-1 * WT
            S1            = (Y * WT).template triangularView<Eigen::Upper>();
            S1            = -S1;
            S1.diagonal() = U.diagonal() + S1.diagonal();
        }
        else
        {
            diagInnerProductTransposed(Y, W, Sdiag);
            Sdiag.diagonal() = U.diagonal() - Sdiag.diagonal();
        }

        // r = a - W * V^-1 * b
        ej = ea + -(Y * eb);


        if (solverOptions.solverType == LinearSolverOptions::SolverType::Direct)
        {
            // Direct recusive ldlt solver
            if (!ldlt)
            {
                ldlt = std::make_unique<LDLT>();
                ldlt->compute(S1);
            }
            else
            {
                ldlt->factorize(S1);
            }
            da = ldlt->solve(ej);
        }
        else
        {
            if (explizitSchur)
            {
                P.compute(S1);
            }
            else
            {
                P.compute(Sdiag);
            }
            da.setZero();

            // Iterative CG solver
            Eigen::Index iters = solverOptions.maxIterativeIterations;
            double tol         = solverOptions.iterativeTolerance;
            //            XUType tmp(n);

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


    void analyzePattern_omp(const AType& A, const LinearSolverOptions& solverOptions)
    {
#pragma omp single
        {
            resize(A.u.rows(), A.v.rows());


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

    void solve_omp(AType& A, XType& x, XType& b, const LinearSolverOptions& solverOptions = LinearSolverOptions())
    {
        // Some references for easier access
        const AUType& U  = A.u;
        const AVType& V  = A.v;
        const AWType& W  = A.w;
        XUType& da       = x.u;
        XVType& db       = x.v;
        const XUType& ea = b.u;
        const XVType& eb = b.v;



        if (!patternAnalyzed) analyzePattern_omp(A, solverOptions);


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
    Eigen::DiagonalMatrix<UBlock, -1> Sdiag;
    XUType ej;
    XUType tmp;

    std::vector<int> transposeTargets;
    AWTType WT;

    RecursiveDiagonalPreconditioner<UBlock> P;
    S1Type S1;
    //    InnerSolver1 solver1;

    std::unique_ptr<LDLT> ldlt;

    bool patternAnalyzed = false;
    bool hasWT           = true;
    bool explizitSchur   = true;
};



}  // namespace Eigen::Recursive
