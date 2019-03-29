/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "Expand.h"
#include "MatrixScalar.h"



namespace Eigen::Recursive
{
template <typename LHS, typename RHS, typename DiagType>
EIGEN_ALWAYS_INLINE void diagInnerProductTransposed(const LHS& lhs, const RHS& rhsTransposed, DiagType& res)
{
    eigen_assert(lhs.IsRowMajor && rhsTransposed.IsRowMajor);
    eigen_assert(lhs.rows() == rhsTransposed.rows());
    eigen_assert(lhs.cols() == rhsTransposed.cols());
    eigen_assert(res.rows() == lhs.rows());

    for (int i = 0; i < lhs.rows(); ++i)
    {
        typename DiagType::Scalar value;
        setZero(value);
        typename LHS::InnerIterator lhsit(lhs, i);
        typename RHS::InnerIterator rhsit(rhsTransposed, i);

        for (; lhsit; ++lhsit, ++rhsit)
        {
            value.get() += lhsit.value().get() * transpose(rhsit.value().get());
        }
        res.diagonal()(i) = value;
    }
}

// Compute res = lhs^T * rhs
// lhs is a sparse matrix in row major storage order!
template <typename LHS, typename RHS, typename RES>
EIGEN_ALWAYS_INLINE void multSparseRowTransposedVector(const LHS& lhsTransposed, const RHS& rhs, RES& res)
{
    eigen_assert(lhsTransposed.IsRowMajor);
    eigen_assert(lhsTransposed.rows() == rhs.rows());

    setZero(res);
    for (int i = 0; i < lhsTransposed.outerSize(); ++i)
    {
        auto value = rhs(i).get();

        typename LHS::InnerIterator lhsit(lhsTransposed, i);

        for (; lhsit; ++lhsit)
        {
            res(lhsit.index()).get() += lhsit.value().get().transpose() * value;
        }
    }
}



}  // namespace Eigen::Recursive



// Computes R = M * D  with
// M : Sparse Matrix in either row or column major format
// D : Diagonal (dense) matrix
// R : Result same format and sparsity pattern as M
template <typename S, typename DiagType>
EIGEN_ALWAYS_INLINE S multSparseDiag(const S& M, const DiagType& D)
{
    eigen_assert(M.cols() == D.rows());

    S result(M.rows(), M.cols());
    result.reserve(M.nonZeros());
    result.markAsRValue();

    // Copy the structure
    for (int k = 0; k < M.outerSize() + 1; ++k)
    {
        result.outerIndexPtr()[k] = M.outerIndexPtr()[k];
    }
    for (int k = 0; k < M.nonZeros(); ++k)
    {
        result.innerIndexPtr()[k] = M.innerIndexPtr()[k];
    }

    // Copmpute result
    for (int k = 0; k < M.outerSize(); ++k)
    {
        typename S::InnerIterator itM(M, k);
        typename S::InnerIterator itRes(result, k);

        for (; itM; ++itM, ++itRes)
        {
            itRes.valueRef() = itM.value() * D.diagonal()(itM.col());
        }
    }

    return result;
}

template <typename Diag, typename Vec>
EIGEN_ALWAYS_INLINE Vec multDiagVector(const Diag& D, const Vec& v)
{
    eigen_assert(D.cols() == v.rows());

    Vec result;
    result.resize(v.rows(), v.cols());
    //    Vec result = v;

    for (int k = 0; k < D.rows(); ++k)
    {
        result(k) = D.diagonal()(k) * v(k);
    }

    return result;
}


// v = D * v
template <typename Diag, typename Vec>
EIGEN_ALWAYS_INLINE void multDiagVector2(const Diag& D, Vec& v)
{
    //    cout << D.rows() << " " << D.cols() << " " << D.size() << endl;
    eigen_assert(D.cols() == v.rows());
    for (int k = 0; k < D.rows(); ++k)
    {
        v(k) = D.diagonal()(k) * v(k);
    }
}

template <typename Diag, typename Vec>
EIGEN_ALWAYS_INLINE Vec multDiagVectorMulti(const Diag& D, const Vec& v)
{
    eigen_assert(D.cols() == v.rows());

    Vec result;
    result.resize(v.rows(), v.cols());
    //    Vec result = v;

    for (int k = 0; k < D.rows(); ++k)
    {
        result.row(k) = D.diagonal()(k) * v.row(k);
    }

    return result;
}
