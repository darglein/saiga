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


template <typename LHS, typename RHS, typename DiagType>
EIGEN_ALWAYS_INLINE void diagInnerProductTransposed_omp(const LHS& lhs, const RHS& rhsTransposed, DiagType& res)
{
    eigen_assert(lhs.IsRowMajor && rhsTransposed.IsRowMajor);
    eigen_assert(lhs.rows() == rhsTransposed.rows());
    eigen_assert(lhs.cols() == rhsTransposed.cols());
    eigen_assert(res.rows() == lhs.rows());

#pragma omp for
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
EIGEN_ALWAYS_INLINE void multSparseDiag(const S& M, const DiagType& D, S& result)
{
    eigen_assert(M.cols() == D.rows());


    result.resize(M.rows(), M.cols());
    result.reserve(M.nonZeros());



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
}
template <typename S, typename DiagType>
EIGEN_ALWAYS_INLINE void multSparseDiag_omp(const S& M, const DiagType& D, S& result)
{
    eigen_assert(M.cols() == D.rows());

#pragma omp single
    {
        result.resize(M.rows(), M.cols());
        result.reserve(M.nonZeros());
    }


    // Copy the structure
#pragma omp for nowait
    for (int k = 0; k < M.outerSize() + 1; ++k)
    {
        result.outerIndexPtr()[k] = M.outerIndexPtr()[k];
    }
#pragma omp for
    for (int k = 0; k < M.nonZeros(); ++k)
    {
        result.innerIndexPtr()[k] = M.innerIndexPtr()[k];
    }

// Copmpute result
#pragma omp for
    for (int k = 0; k < M.outerSize(); ++k)
    {
        typename S::InnerIterator itM(M, k);
        typename S::InnerIterator itRes(result, k);

        for (; itM; ++itM, ++itRes)
        {
            itRes.valueRef() = itM.value() * D.diagonal()(itM.col());
        }
    }
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

template <typename Diag, typename Vec>
EIGEN_ALWAYS_INLINE void multDiagVector_omp(const Diag& D, const Vec& v, Vec& result)
{
    eigen_assert(D.cols() == v.rows());

#pragma omp for
    for (int k = 0; k < D.rows(); ++k)
    {
        result(k) = D.diagonal()(k) * v(k);
    }
}



// v = D * v
template <typename Diag, typename Vec>
EIGEN_ALWAYS_INLINE void multDiagVector2(const Diag& D, Vec& v)
{
    //    std::cout << D.rows() << " " << D.cols() << " " << D.size() << std::endl;
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


template <typename Mat, typename Vec>
EIGEN_ALWAYS_INLINE void denseMV(const Mat& A, const Vec& v, Vec& result)
{
    eigen_assert(A.cols() == v.rows());
    eigen_assert(A.IsRowMajor);

    for (int k = 0; k < A.rows(); ++k)
    {
        result(k).get().setZero();
        for (int j = 0; j < A.cols(); ++j)
        {
            result(k).get() += A(k, j).get() * v(j).get();
        }
    }
}

template <typename Mat, typename Vec>
EIGEN_ALWAYS_INLINE void denseMV_omp(const Mat& A, const Vec& v, Vec& result)
{
    eigen_assert(A.cols() == v.rows());
    eigen_assert(A.IsRowMajor);

#pragma omp for
    for (int k = 0; k < A.rows(); ++k)
    {
        result(k).get().setZero();
        for (int j = 0; j < A.cols(); ++j)
        {
            result(k).get() += A(k, j).get() * v(j).get();
        }
    }
}
