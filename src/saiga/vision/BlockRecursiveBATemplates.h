/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/MatrixScalar.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/recursiveMatrices/Expand.h"

#include "Eigen/Sparse"

namespace Saiga
{
// ======================== Types ========================

// Block size
const int asize = 6;
const int bsize = 3;

using T = double;

// block types
using ADiag  = Eigen::Matrix<T, asize, asize>;
using BDiag  = Eigen::Matrix<T, bsize, bsize>;
using WElem  = Eigen::Matrix<T, asize, bsize>;
using WTElem = Eigen::Matrix<T, bsize, asize>;
using ARes   = Eigen::Matrix<T, asize, 1>;
using BRes   = Eigen::Matrix<T, bsize, 1>;

// Block structured diagonal matrices
using UType = Eigen::DiagonalMatrix<MatrixScalar<ADiag>, -1>;
using VType = Eigen::DiagonalMatrix<MatrixScalar<BDiag>, -1>;

// Block structured vectors
using DAType = Eigen::Matrix<MatrixScalar<ARes>, -1, 1>;
using DBType = Eigen::Matrix<MatrixScalar<BRes>, -1, 1>;

// Block structured sparse matrix
using WType  = Eigen::SparseMatrix<MatrixScalar<WElem>>;
using WTType = Eigen::SparseMatrix<MatrixScalar<WTElem>>;
using SType  = Eigen::SparseMatrix<MatrixScalar<ADiag>, Eigen::RowMajor>;



}  // namespace Saiga

namespace Eigen
{
// ======================== Eigen Magic ========================

template <typename BinaryOp>
struct ScalarBinaryOpTraits<Saiga::MatrixScalar<Saiga::WElem>, Saiga::MatrixScalar<Saiga::WTElem>, BinaryOp>
{
    typedef Saiga::MatrixScalar<Saiga::ADiag> ReturnType;
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<Saiga::MatrixScalar<Saiga::WElem>, Saiga::MatrixScalar<Saiga::BDiag>, BinaryOp>
{
    typedef Saiga::MatrixScalar<Saiga::WElem> ReturnType;
};



template <typename BinaryOp>
struct ScalarBinaryOpTraits<Saiga::MatrixScalar<Saiga::WElem>, Saiga::MatrixScalar<Saiga::BRes>, BinaryOp>
{
    typedef Saiga::MatrixScalar<Saiga::ARes> ReturnType;
};


template <typename SparseLhsType, typename DenseRhsType, typename DenseResType>
struct internal::sparse_time_dense_product_impl<SparseLhsType, DenseRhsType, DenseResType,
                                                Saiga::MatrixScalar<Saiga::ARes>, Eigen::ColMajor, true>
{
    typedef typename internal::remove_all<SparseLhsType>::type Lhs;
    typedef typename internal::remove_all<DenseRhsType>::type Rhs;
    typedef typename internal::remove_all<DenseResType>::type Res;
    typedef typename evaluator<Lhs>::InnerIterator LhsInnerIterator;
    using AlphaType = Saiga::MatrixScalar<Saiga::ARes>;
    static void run(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res, const AlphaType& alpha)
    {
        evaluator<Lhs> lhsEval(lhs);
        for (Index c = 0; c < rhs.cols(); ++c)
        {
            for (Index j = 0; j < lhs.outerSize(); ++j)
            {
                for (LhsInnerIterator it(lhsEval, j); it; ++it)
                {
                    res.coeffRef(it.index(), c) += (it.value() * rhs.coeff(j, c));
                }
            }
        }
    }
};

template <typename BinaryOp>
struct ScalarBinaryOpTraits<Saiga::MatrixScalar<Saiga::WTElem>, Saiga::MatrixScalar<Saiga::ARes>, BinaryOp>
{
    typedef Saiga::MatrixScalar<Saiga::BRes> ReturnType;
};

template <typename SparseLhsType, typename DenseRhsType, typename DenseResType>
struct internal::sparse_time_dense_product_impl<SparseLhsType, DenseRhsType, DenseResType,
                                                Saiga::MatrixScalar<Saiga::BRes>, Eigen::ColMajor, true>
{
    typedef typename internal::remove_all<SparseLhsType>::type Lhs;
    typedef typename internal::remove_all<DenseRhsType>::type Rhs;
    typedef typename internal::remove_all<DenseResType>::type Res;
    typedef typename evaluator<Lhs>::InnerIterator LhsInnerIterator;
    using AlphaType = Saiga::MatrixScalar<Saiga::BRes>;
    static void run(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res, const AlphaType& alpha)
    {
        evaluator<Lhs> lhsEval(lhs);
        for (Index c = 0; c < rhs.cols(); ++c)
        {
            for (Index j = 0; j < lhs.outerSize(); ++j)
            {
                for (LhsInnerIterator it(lhsEval, j); it; ++it)
                {
                    res.coeffRef(it.index(), c) += (it.value() * rhs.coeff(j, c));
                }
            }
        }
    }
};

}  // namespace Eigen


// Computes R = M * D  with
// M : Sparse Matrix in either row or column major format
// D : Diagonal (dense) matrix
// R : Result same format and sparsity pattern as M
template <typename S, typename DiagType>
S multSparseDiag(const S& M, const DiagType& D)
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
Vec multDiagVector(const Diag& D, const Vec& v)
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
Vec multDiagVectorMulti(const Diag& D, const Vec& v)
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
