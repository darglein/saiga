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

#include <iostream>



namespace Eigen
{
namespace internal
{
// exact the same impl as normal sparse x sparse product but the scalar multiplication is swapped to y * x
template <typename Lhs, typename Rhs, typename ResultType>
static void conservative_sparse_sparse_product_impl_yx(const Lhs& lhs, const Rhs& rhs, ResultType& res)
{
    typedef typename remove_all<Lhs>::type::Scalar LhsScalar;
    typedef typename remove_all<Rhs>::type::Scalar RhsScalar;
    typedef typename remove_all<ResultType>::type::Scalar ResScalar;

    // make sure to call innerSize/outerSize since we fake the storage order.
    Index rows = lhs.innerSize();
    Index cols = rhs.outerSize();
    eigen_assert(lhs.outerSize() == rhs.innerSize());

    ei_declare_aligned_stack_constructed_variable(bool, mask, rows, 0);
    ei_declare_aligned_stack_constructed_variable(ResScalar, values, rows, 0);
    ei_declare_aligned_stack_constructed_variable(Index, indices, rows, 0);

    std::memset(mask, 0, sizeof(bool) * rows);

    evaluator<Lhs> lhsEval(lhs);
    evaluator<Rhs> rhsEval(rhs);

    // estimate the number of non zero entries
    // given a rhs column containing Y non zeros, we assume that the respective Y columns
    // of the lhs differs in average of one non zeros, thus the number of non zeros for
    // the product of a rhs column with the lhs is X+Y where X is the average number of non zero
    // per column of the lhs.
    // Therefore, we have nnz(lhs*rhs) = nnz(lhs) + nnz(rhs)
    Index estimated_nnz_prod = lhsEval.nonZerosEstimate() + rhsEval.nonZerosEstimate();

    res.setZero();
    res.reserve(Index(estimated_nnz_prod));
    // we compute each column of the result, one after the other
    for (Index j = 0; j < cols; ++j)
    {
        res.startVec(j);
        Index nnz = 0;
        for (typename evaluator<Rhs>::InnerIterator rhsIt(rhsEval, j); rhsIt; ++rhsIt)
        {
            RhsScalar y = rhsIt.value();
            Index k     = rhsIt.index();
            for (typename evaluator<Lhs>::InnerIterator lhsIt(lhsEval, k); lhsIt; ++lhsIt)
            {
                Index i     = lhsIt.index();
                LhsScalar x = lhsIt.value();
                if (!mask[i])
                {
                    mask[i]      = true;
                    values[i]    = y * x;
                    indices[nnz] = i;
                    ++nnz;
                }
                else
                    values[i] += y * x;
            }
        }
        {
            // unordered insertion
            for (Index k = 0; k < nnz; ++k)
            {
                Index i                                   = indices[k];
                res.insertBackByOuterInnerUnordered(j, i) = values[i];
                mask[i]                                   = false;
            }
        }
    }
    res.finalize();
}
}  // namespace internal
}  // namespace Eigen


template <typename SparseLhsType, typename T, typename DenseResType>
struct Eigen::internal::sparse_time_dense_product_impl<
    SparseLhsType, Eigen::Matrix<Eigen::Recursive::MatrixScalar<T>, -1, 1>, DenseResType, typename DenseResType::Scalar,
    Eigen::RowMajor, true>
{
    using DenseRhsType = Eigen::Matrix<Eigen::Recursive::MatrixScalar<T>, -1, 1>;

    typedef typename internal::remove_all<SparseLhsType>::type Lhs;
    typedef typename internal::remove_all<DenseRhsType>::type Rhs;
    typedef typename internal::remove_all<DenseResType>::type Res;
    typedef typename evaluator<Lhs>::InnerIterator LhsInnerIterator;
    typedef evaluator<Lhs> LhsEval;
    static void run(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res,
                    const typename Res::Scalar& alpha)
    {
        LhsEval lhsEval(lhs);
        Index n = lhs.outerSize();


        for (Index c = 0; c < rhs.cols(); ++c)
        {
            { /* This loop can be parallelized without syncronization */
                for (Index i = 0; i < n; ++i) processRow(lhsEval, rhs, res, alpha, i, c);
            }
        }
    }

    static void processRow(const LhsEval& lhsEval, const DenseRhsType& rhs, DenseResType& res,
                           const typename Res::Scalar&, Index i, Index col)
    {
#ifndef SAIGA_GENEREAL_MV
        for (LhsInnerIterator it(lhsEval, i); it; ++it)
        {
            auto& vlhs = it.value().get();
            auto& vrhs = rhs.coeff(it.index(), col).get();
            res.coeffRef(i, col).get() += vlhs * vrhs;
        }
#else
        using SuperScalar = typename Res::Scalar::Scalar;
        typedef const_blas_data_mapper<SuperScalar, Index, RowMajor> LhsMapper;
        typedef const_blas_data_mapper<SuperScalar, Index, ColMajor> RhsMapper;
        general_matrix_vector_product<Index, SuperScalar, LhsMapper, RowMajor, false, double, RhsMapper, false, 0> prod;

        for (LhsInnerIterator it(lhsEval, i); it; ++it)
        {
            auto vlhs = it.value().get();
            auto vrhs = rhs.coeff(it.index(), col).get();
            //            res.coeffRef(i, col) += vlhs * vrhs;


            typename Res::Scalar::M tmp2;
            tmp2.setZero();
            prod.run(vlhs.rows(), vlhs.cols(), LhsMapper(vlhs.data(), vlhs.outerStride()), RhsMapper(vrhs.data(), 1),
                     tmp2.data(), tmp2.col(0).innerStride(), 1);

            res.coeffRef(i, col).get() += tmp2;

            //            res.coeffRef(i, col) += it.value() * rhs.coeff(it.index(), col);
        }
#endif
    }
};

template <typename SparseLhsType, typename T, typename DenseResType>
struct Eigen::internal::sparse_time_dense_product_impl<
    SparseLhsType, Eigen::Matrix<Eigen::Recursive::MatrixScalar<T>, -1, 1>, DenseResType, typename DenseResType::Scalar,
    Eigen::ColMajor, true>
{
    using DenseRhsType = Eigen::Matrix<Eigen::Recursive::MatrixScalar<T>, -1, 1>;

    typedef typename internal::remove_all<SparseLhsType>::type Lhs;
    typedef typename internal::remove_all<DenseRhsType>::type Rhs;
    typedef typename internal::remove_all<DenseResType>::type Res;
    typedef typename evaluator<Lhs>::InnerIterator LhsInnerIterator;

    static void run(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res, const typename Res::Scalar&)
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



namespace Eigen
{
namespace internal
{
template <typename T, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector<
    Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<T>, Eigen::RowMajor>, Rhs, ResultType, RowMajor, RowMajor,
    ColMajor>
{
    using Lhs = Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<T>, Eigen::RowMajor>;
    static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res)
    {
        typedef SparseMatrix<typename ResultType::Scalar, RowMajor, typename ResultType::StorageIndex> RowMajorMatrix;
        RowMajorMatrix resRow(lhs.rows(), rhs.cols());
        conservative_sparse_sparse_product_impl_yx<Rhs, Lhs, RowMajorMatrix>(rhs, lhs, resRow);
        res = resRow;
    }
};


template <typename T, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector<
    Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<T>, Eigen::RowMajor>, Rhs, ResultType, RowMajor, ColMajor,
    ColMajor>
{
    using Lhs = Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<T>, Eigen::RowMajor>;
    static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res)
    {
        typedef SparseMatrix<typename Rhs::Scalar, RowMajor, typename ResultType::StorageIndex> RowMajorRhs;
        typedef SparseMatrix<typename ResultType::Scalar, RowMajor, typename ResultType::StorageIndex> RowMajorRes;
        RowMajorRhs rhsRow = rhs;
        RowMajorRes resRow(lhs.rows(), rhs.cols());
        conservative_sparse_sparse_product_impl_yx<RowMajorRhs, Lhs, RowMajorRes>(rhsRow, lhs, resRow);
        res = resRow;
    }
};

template <typename T, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector<
    Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<T>, Eigen::RowMajor>, Rhs, ResultType, RowMajor, ColMajor,
    RowMajor>
{
    using Lhs = Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<T>, Eigen::RowMajor>;
    static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res)
    {
        typedef SparseMatrix<typename Lhs::Scalar, ColMajor, typename ResultType::StorageIndex> ColMajorLhs;
        typedef SparseMatrix<typename ResultType::Scalar, ColMajor, typename ResultType::StorageIndex> ColMajorRes;
        ColMajorLhs lhsCol = lhs;
        ColMajorRes resCol(lhs.rows(), rhs.cols());
        internal::conservative_sparse_sparse_product_impl<ColMajorLhs, Rhs, ColMajorRes>(lhsCol, rhs, resCol);
        res = resCol;
    }
};

}  // namespace internal
}  // namespace Eigen
