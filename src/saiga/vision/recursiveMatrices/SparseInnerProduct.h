/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/assert.h"
#include "saiga/vision/recursiveMatrices/Expand.h"
#include "saiga/vision/recursiveMatrices/MatrixScalar.h"



namespace Saiga
{
template <typename LHS, typename RHS, typename DiagType>
EIGEN_ALWAYS_INLINE void diagInnerProductTransposed(const LHS& lhs, const RHS& rhsTransposed, DiagType& res)
{
    SAIGA_ASSERT(lhs.IsRowMajor && rhsTransposed.IsRowMajor);
    SAIGA_ASSERT(lhs.rows() == rhsTransposed.rows());
    SAIGA_ASSERT(lhs.cols() == rhsTransposed.cols());
    SAIGA_ASSERT(res.rows() == lhs.rows());

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
    SAIGA_ASSERT(lhsTransposed.IsRowMajor);
    SAIGA_ASSERT(lhsTransposed.rows() == rhs.rows());

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



}  // namespace Saiga



// Computes R = M * D  with
// M : Sparse Matrix in either row or column major format
// D : Diagonal (dense) matrix
// R : Result same format and sparsity pattern as M
template <typename S, typename DiagType>
EIGEN_ALWAYS_INLINE S multSparseDiag(const S& M, const DiagType& D)
{
    SAIGA_ASSERT(M.cols() == D.rows());

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
    SAIGA_ASSERT(D.cols() == v.rows());

    Vec result;
    result.resize(v.rows(), v.cols());
    //    Vec result = v;

    for (int k = 0; k < D.rows(); ++k)
    {
        result(k) = D.diagonal()(k) * v(k);
    }

    return result;
}


template <typename Diag, typename Vec>
EIGEN_ALWAYS_INLINE Vec multDiagVectorMulti(const Diag& D, const Vec& v)
{
    SAIGA_ASSERT(D.cols() == v.rows());

    Vec result;
    result.resize(v.rows(), v.cols());
    //    Vec result = v;

    for (int k = 0; k < D.rows(); ++k)
    {
        result.row(k) = D.diagonal()(k) * v.row(k);
    }

    return result;
}
