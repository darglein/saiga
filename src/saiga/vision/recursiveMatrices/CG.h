/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/assert.h"
#include "saiga/vision/MatrixScalar.h"
#include "saiga/vision/recursiveMatrices/Cholesky.h"
#include "saiga/vision/recursiveMatrices/Dot.h"
#include "saiga/vision/recursiveMatrices/Inverse.h"
#include "saiga/vision/recursiveMatrices/NeutralElements.h"
#include "saiga/vision/recursiveMatrices/Norm.h"
#include "saiga/vision/recursiveMatrices/ScalarMult.h"


#ifndef NO_CG_TYPES
using Scalar = float;
const int bn = 4;
const int bm = 4;
using Block  = Eigen::Matrix<Scalar, bn, bm>;
using Vector = Eigen::Matrix<Scalar, bn, 1>;
#endif

template <typename BinaryOp>
struct Eigen::ScalarBinaryOpTraits<Saiga::MatrixScalar<Block>, Saiga::MatrixScalar<Vector>, BinaryOp>
{
    typedef Saiga::MatrixScalar<Vector> ReturnType;
};


#ifndef NO_CG_SPEZIALIZATIONS
template <typename SparseLhsType, typename DenseRhsType>
struct Eigen::internal::sparse_time_dense_product_impl<SparseLhsType, DenseRhsType,
                                                       Eigen::Matrix<Saiga::MatrixScalar<Vector>, -1, 1>,
                                                       Saiga::MatrixScalar<Vector>, Eigen::RowMajor, true>
{
    typedef typename internal::remove_all<SparseLhsType>::type Lhs;
    typedef typename internal::remove_all<DenseRhsType>::type Rhs;
    using DenseResType = DenseRhsType;
    typedef typename internal::remove_all<DenseResType>::type Res;
    typedef typename evaluator<Lhs>::InnerIterator LhsInnerIterator;
    typedef evaluator<Lhs> LhsEval;
    static void run(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res, const typename Res::Scalar&)
    {
        LhsEval lhsEval(lhs);

        Index n = lhs.outerSize();


        for (Index c = 0; c < rhs.cols(); ++c)
        {
            {
                for (Index i = 0; i < n; ++i) processRow(lhsEval, rhs, res, i, c);
            }
        }
    }

    static void processRow(const LhsEval& lhsEval, const DenseRhsType& rhs, DenseResType& res, Index i, Index col)
    {
        typename Res::Scalar tmp(0);
        for (LhsInnerIterator it(lhsEval, i); it; ++it) tmp += it.value() * rhs.coeff(it.index(), col);
        res.coeffRef(i, col) += tmp;
    }
};

#endif

namespace Saiga
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
    template <typename MatType>
    RecursiveDiagonalPreconditioner& factorize(const MatType& mat)
    {
        m_invdiag.resize(mat.cols());
        for (int j = 0; j < mat.outerSize(); ++j)
        {
            typename MatType::InnerIterator it(mat, j);
            while (it && it.index() != j) ++it;
            if (it && it.index() == j)
                //          m_invdiag(j) = Scalar(1)/it.value();
                m_invdiag(j) = inverseCholesky(it.value());
            else
                //                m_invdiag(j) = Scalar(1);
                m_invdiag(j) = MultiplicativeNeutral<Scalar>::get();
        }
        m_isInitialized = true;
        return *this;
    }

    template <typename T>
    RecursiveDiagonalPreconditioner& factorize(const Eigen::DiagonalMatrix<T, -1>& mat)
    {
        auto N = mat.rows();
        m_invdiag.resize(N);

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
        x = m_invdiag.array() * b.array();
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

   protected:
    Vector m_invdiag;
    bool m_isInitialized;
};

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
    using namespace Eigen;

    using std::abs;
    using std::sqrt;
    typedef SuperScalar RealScalar;
    typedef SuperScalar Scalar;
    typedef Rhs VectorType;

    RealScalar tol = tol_error;
    Index maxIters = iters;

    Index n = rhs.rows();


    VectorType z(n), tmp(n);


    VectorType residual = rhs - applyA(x);

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
    if (residualNorm2 < threshold)
    {
        iters     = 0;
        tol_error = sqrt(residualNorm2 / rhsNorm2);
        return;
    }

    VectorType p(n);
    p = precond.solve(residual);  // initial search direction

    // the square of the absolute value of r scaled by invM
    RealScalar absNew = dot(residual, p);
    Index i           = 0;
    while (i < maxIters)
    {
        //        cout << "CG Residual " << i << ": " << residualNorm2 << endl;
        tmp = applyA(p);
        // the amount we travel on dir
        Scalar alpha = absNew / dot(p, tmp);
        // update solution
        x += scalarMult(p, alpha);
        // update residual
        residual -= scalarMult(tmp, alpha);

        residualNorm2 = squaredNorm(residual);
        if (residualNorm2 < threshold) break;

        z = precond.solve(residual);  // approximately solve for "A z = residual"

        RealScalar absOld = absNew;
        absNew            = dot(residual, z);  // update the absolute value of r
        RealScalar beta = absNew / absOld;  // calculate the Gram-Schmidt value used to create the new search direction
        p               = z + scalarMult(p, beta);  // update search direction
        i++;
    }
    tol_error = sqrt(residualNorm2 / rhsNorm2);
    iters     = i;
}

}  // namespace Saiga
