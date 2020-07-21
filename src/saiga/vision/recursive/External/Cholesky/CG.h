/**
 * This file contains (modified) code from the Eigen library.
 * Eigen License:
 *
 * Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
 * Copyright (C) 2007-2011 Benoit Jacob <jacob.benoit.1@gmail.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla
 * Public License v. 2.0. If a copy of the MPL was not distributed
 * with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * ======================
 *
 * The modifications are part of the Eigen Recursive Matrix Extension (ERME).
 * ERME License:
 *
 * Copyright (c) 2019 Darius Rückert
 * Licensed under the MIT License.
 */


#pragma once
#include "../Core.h"
#include "../Core/ParallelHelper.h"
#include "Cholesky.h"
namespace Eigen::Recursive
{
template <typename _Scalar>
class RecursiveDiagonalPreconditioner
{
    typedef _Scalar Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

   public:
    typedef typename Vector::StorageIndex StorageIndex;
    enum
    {
        ColsAtCompileTime    = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic
    };

    RecursiveDiagonalPreconditioner() : m_isInitialized(false) {}

    void resize(int N) { m_invdiag.resize(N); }
    template <typename MatType>
    explicit RecursiveDiagonalPreconditioner(const MatType& mat) : m_invdiag(mat.cols())
    {
        compute(mat);
    }

    Eigen::Index rows() const { return m_invdiag.size(); }
    Eigen::Index cols() const { return m_invdiag.size(); }

    template <typename MatType>
    RecursiveDiagonalPreconditioner& analyzePattern(const MatType&)
    {
        return *this;
    }

    // Sparse Matrix Initialization
    template <typename Scalar, int options>
    //    RecursiveDiagonalPreconditioner& factorize(const MatType& mat)
    RecursiveDiagonalPreconditioner& factorize(const SparseMatrix<Scalar, options>& mat)
    {
        using MatType = SparseMatrix<Scalar, options>;
        m_invdiag.resize(mat.cols());
        for (int j = 0; j < mat.outerSize(); ++j)
        {
            typename MatType::InnerIterator it(mat, j);
            while (it && it.index() != j) ++it;
            if (it && it.index() == j)
                //          m_invdiag(j) = Scalar(1)/it.value();
                removeMatrixScalar(m_invdiag(j)) = removeMatrixScalar(inverseCholesky(it.value()));
            else
                //                m_invdiag(j) = Scalar(1);
                removeMatrixScalar(m_invdiag(j)) = removeMatrixScalar(MultiplicativeNeutral<Scalar>::get());
        }
        m_isInitialized = true;
        return *this;
    }

    // Dense Matrix Initialization
    template <typename MatType>
    RecursiveDiagonalPreconditioner& factorize(const MatType& mat)
    {
        m_invdiag.resize(mat.cols());
        for (int j = 0; j < mat.outerSize(); ++j)
        {
            removeMatrixScalar(m_invdiag(j)) = removeMatrixScalar(inverseCholesky(mat(j, j)));
        }
        m_isInitialized = true;
        return *this;
    }

    template <typename T>
    RecursiveDiagonalPreconditioner& factorize(const Eigen::DiagonalMatrix<T, -1>& mat)
    {
        auto N = mat.rows();
        if (m_invdiag.rows() != N)
        {
            std::terminate();
            m_invdiag.resize(N);
        }

        //#pragma omp for
        for (int j = 0; j < N; ++j)
        {
            m_invdiag(j) = inverseCholesky(mat.diagonal()(j));
        }
        m_isInitialized = true;
        return *this;
    }

    template <typename MatType>
    RecursiveDiagonalPreconditioner& compute(const MatType& mat)
    {
        return factorize(mat);
    }

    /** \internal */
    template <typename Rhs, typename Dest>
    void _solve_impl(const Rhs& b, Dest& x) const
    {
        //        x = m_invdiag.array() * b.array();
        //#pragma omp for
        for (int i = 0; i < b.rows(); ++i)
        {
            x(i) = m_invdiag(i) * b(i);
        }
    }

    template <typename Rhs>
    inline const Eigen::Solve<RecursiveDiagonalPreconditioner, Rhs> solve(const Eigen::MatrixBase<Rhs>& b) const
    {
        eigen_assert(m_isInitialized && "DiagonalPreconditioner is not initialized.");
        eigen_assert(m_invdiag.size() == b.rows() &&
                     "DiagonalPreconditioner::solve(): invalid number of rows of the right hand side matrix b");
        return Eigen::Solve<RecursiveDiagonalPreconditioner, Rhs>(*this, b.derived());
    }

    Eigen::ComputationInfo info() { return Eigen::Success; }

    const auto& getDiagElement(int i) const { return m_invdiag(i); }

    Vector m_invdiag;

   protected:
    bool m_isInitialized;
};

//#define RM_CG_DEBUG_OUTPUT

/**
 * A conjugate gradient solver, which works for recursives matrices.
 * Solve:
 *          A * x = b   for x
 *
 * The matrix A is given as function (for example a lambda function).
 * This way we can implement an implicit cg solver, which does not construct the full matrix A.
 *
 * Example call:
 *
 * // Build preconditioner
 * RecursiveDiagonalPreconditioner<MatrixScalar<Block>> P;
 * Eigen::Index iters = 50;
 * Scalar tol         = 1e-50;
 * P.compute(S);
 *
 * // Solve with explicit matrix S
 * DAType tmp(n);
 * recursive_conjugate_gradient(
 *     [&](const DAType& v) {
 *         tmp = S * v;
 *         return tmp;
 *     },
 *     ej, da, P, iters, tol);
 *
 */
template <typename MultFunction, typename Rhs, typename Dest, typename Preconditioner, typename SuperScalar>
EIGEN_DONT_INLINE void recursive_conjugate_gradient(const MultFunction& applyA, const Rhs& rhs, Dest& x,
                                                    const Preconditioner& precond, Eigen::Index& iters,
                                                    SuperScalar& tol_error)
{
    // Typedefs
    using namespace Eigen;
    using std::abs;
    using std::sqrt;
    typedef SuperScalar RealScalar;
    typedef SuperScalar Scalar;
    typedef Rhs VectorType;

    // Temp Vector variables
    Index n = rhs.rows();

#ifdef RM_CG_DEBUG_OUTPUT
    std::cout << "Starting recursive CG" << std::endl;
    std::cout << "Iterations: " << iters << std::endl;
    std::cout << "Tolerance: " << tol_error << std::endl;
    std::cout << "N: " << n << std::endl;
#endif

#if 0
    // Create them locally
    VectorType z(n);
    VectorType p(n);
#else
    // Use static variables so a repeated call with the same size doesn't allocate memory
    static thread_local VectorType z;
    static thread_local VectorType p;
    static thread_local VectorType residual;
    z.resize(n);
    p.resize(n);
    residual.resize(n);
#endif



    RealScalar tol = tol_error;
    Index maxIters = iters;

    applyA(x, residual);
    residual = rhs - residual;

    RealScalar rhsNorm2 = squaredNorm(rhs);
    if (rhsNorm2 == 0)
    {
        x.setZero();
        iters     = 0;
        tol_error = 0;
        return;
    }
    RealScalar threshold     = tol * tol * rhsNorm2;
    RealScalar residualNorm2 = squaredNorm(residual);
#ifdef RM_CG_DEBUG_OUTPUT
    std::cout << "Initial residual: " << residualNorm2 << std::endl;
#endif
    if (residualNorm2 < threshold)
    {
        iters     = 0;
        tol_error = sqrt(residualNorm2 / rhsNorm2);
        return;
    }

    p = precond.solve(residual);  // initial search direction
    // the square of the absolute value of r scaled by invM
    RealScalar absNew = dot(residual, p);
#ifdef RM_CG_DEBUG_OUTPUT
    std::cout << "dot(r,p): " << absNew << std::endl;
#endif

    Index i = 0;
    while (i < maxIters)
    {
        //        std::cout << "CG Residual " << i << ": " << residualNorm2 << std::endl;
        applyA(p, z);

        // the amount we travel on dir
        Scalar alpha = absNew / dot(p, z);
        // update solution
        x += scalarMult(p, alpha);
        // update residual
        residual -= scalarMult(z, alpha);

        residualNorm2 = squaredNorm(residual);
#ifdef RM_CG_DEBUG_OUTPUT
        std::cout << "Iteration: " << i << " Residual: " << residualNorm2 << " Alpha: " << alpha << std::endl;
#endif
        if (residualNorm2 < threshold) break;

        z = precond.solve(residual);  // approximately solve for "A z = residual"
                                      //        std::cout << expand(p).transpose() << std::endl;

        RealScalar absOld = absNew;
        absNew            = dot(residual, z);  // update the absolute value of r
        RealScalar beta = absNew / absOld;  // calculate the Gram-Schmidt value used to create the new search direction
                                            //        std::cout << "absnew " << absNew << " beta " << beta << std::endl;
        p = z + scalarMult(p, beta);        // update search direction


        i++;
    }
    tol_error = sqrt(residualNorm2 / rhsNorm2);
    iters     = i;
}

#if defined(_OPENMP)

template <typename T>
struct alignas(64) CacheAlignedValues
{
    T data;
};

template <typename T>
inline double accumulate(const T& v)
{
    double d = 0;
    for (auto& v : v)
    {
        d += v.data;
    }
    return d;
}


// Multi threaded implementation
template <typename MultFunction, typename Rhs, typename Dest, typename Preconditioner, typename SuperScalar>
EIGEN_DONT_INLINE void recursive_conjugate_gradient_OMP(const MultFunction& applyA, const Rhs& rhs, Dest& x,
                                                        const Preconditioner& precond, Eigen::Index& iters,
                                                        SuperScalar& tol_error)
{
    // Typedefs
    using namespace Eigen;
    using std::abs;
    using std::sqrt;
    typedef SuperScalar RealScalar;
    typedef SuperScalar Scalar;
    typedef Rhs VectorType;

    // Temp Vector variables
    Index n = rhs.rows();


    // Use static variables so a repeated call with the same size doesn't allocate memory
    static VectorType z;
    static VectorType p;
    static VectorType residual;
    static std::vector<CacheAlignedValues<Scalar>> tmpResults1, tmpResults;

#    pragma omp single
    {
        z.resize(n);
        p.resize(n);
        residual.resize(n);
        tmpResults1.resize(omp_get_num_threads());
        tmpResults.resize(omp_get_num_threads());
    }

    int tid        = omp_get_thread_num();
    RealScalar tol = tol_error;
    Index maxIters = iters;


    applyA(x, residual);

#    pragma omp for
    for (int i = 0; i < n; ++i)
    {
        residual(i) = rhs(i) - residual(i);
    }

    //    tmpResults[tid]     = squaredNorm_omp(rhs);
    squaredNorm_omp_local(rhs, tmpResults[tid].data);
    RealScalar rhsNorm2 = accumulate(tmpResults);



    if (rhsNorm2 == 0)
    {
//        x.setZero();
#    pragma omp for
        for (int i = 0; i < n; ++i)
        {
            x(i).get().setZero();
        }
        iters     = 0;
        tol_error = 0;
        maxIters  = 0;
    }


    RealScalar threshold = tol * tol * rhsNorm2;

    squaredNorm_omp_local(residual, tmpResults1[tid].data);
    RealScalar residualNorm2 = accumulate(tmpResults1);
    //    RealScalar residualNorm2 = squaredNorm(residual);
    if (residualNorm2 < threshold)
    {
        iters     = 0;
        tol_error = sqrt(residualNorm2 / rhsNorm2);
        maxIters  = 0;
    }

    p = precond.solve(residual);  // initial search direction

    dot_omp_local(residual, p, tmpResults[tid].data);
    RealScalar absNew = accumulate(tmpResults);

    Index i = 0;
    while (i < maxIters)
    {
        //        std::cout << "CG Residual " << i << ": " << residualNorm2 << std::endl;
        applyA(p, z);
        dot_omp_local(p, z, tmpResults1[tid].data);
        Scalar dotpz = accumulate(tmpResults1);
        Scalar alpha = absNew / dotpz;

#    pragma omp for
        for (int i = 0; i < n; ++i)
        {
            // the amount we travel on dir
            // update solution
            x(i) += p(i) * alpha;
            // update residual
            residual(i) -= z(i) * alpha;
        }

        squaredNorm_omp_local(residual, tmpResults[tid].data);
        residualNorm2 = accumulate(tmpResults);

        if (residualNorm2 < threshold) break;
        z = precond.solve(residual);  // approximately solve for "A z = residual"

        RealScalar absOld = absNew;
        dot_omp_local(residual, z, tmpResults[tid].data);
        absNew          = accumulate(tmpResults);
        RealScalar beta = absNew / absOld;  // calculate the Gram-Schmidt value used to create the new search direction
                                            //        std::cout << "absnew " << absNew << " beta " << beta << std::endl;
#    pragma omp for
        for (int i = 0; i < n; ++i)
        {
            p(i) = z(i) + p(i) * beta;  // update search direction
        }

        i++;
    }


    tol_error = sqrt(residualNorm2 / rhsNorm2);
    iters     = i;
}
#endif

}  // namespace Eigen::Recursive
