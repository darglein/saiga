/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/recursiveMatrices/EigenTemplateDefines.h"



#define SAIGA_RM_CREATE_SMV_COL_MAJOR(_RHS_SCALAR)                                                                 \
    template <typename SparseLhsType, typename DenseRhsType, typename DenseResType>                                \
    struct Eigen::internal::sparse_time_dense_product_impl<SparseLhsType, DenseRhsType, DenseResType, _RHS_SCALAR, \
                                                           Eigen::ColMajor, true>                                  \
    {                                                                                                              \
        typedef typename internal::remove_all<SparseLhsType>::type Lhs;                                            \
        typedef typename internal::remove_all<DenseRhsType>::type Rhs;                                             \
        typedef typename internal::remove_all<DenseResType>::type Res;                                             \
        typedef typename evaluator<Lhs>::InnerIterator LhsInnerIterator;                                           \
        using AlphaType = _RHS_SCALAR;                                                                             \
        static void run(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res, const AlphaType&)    \
        {                                                                                                          \
            evaluator<Lhs> lhsEval(lhs);                                                                           \
            for (Index c = 0; c < rhs.cols(); ++c)                                                                 \
            {                                                                                                      \
                for (Index j = 0; j < lhs.outerSize(); ++j)                                                        \
                {                                                                                                  \
                    for (LhsInnerIterator it(lhsEval, j); it; ++it)                                                \
                    {                                                                                              \
                        res.coeffRef(it.index(), c) += (it.value() * rhs.coeff(j, c));                             \
                    }                                                                                              \
                }                                                                                                  \
            }                                                                                                      \
        }                                                                                                          \
    };



#define SAIGA_RM_CREATE_SMV_ROW_MAJOR(_RHS_TYPE)                                                                 \
    template <typename SparseLhsType, typename DenseResType>                                                     \
    struct Eigen::internal::sparse_time_dense_product_impl<SparseLhsType, _RHS_TYPE, DenseResType,               \
                                                           typename DenseResType::Scalar, Eigen::RowMajor, true> \
    {                                                                                                            \
        typedef typename internal::remove_all<SparseLhsType>::type Lhs;                                          \
        typedef typename internal::remove_all<_RHS_TYPE>::type Rhs;                                              \
        typedef typename internal::remove_all<DenseResType>::type Res;                                           \
        typedef typename evaluator<Lhs>::InnerIterator LhsInnerIterator;                                         \
        typedef evaluator<Lhs> LhsEval;                                                                          \
        using DenseRhsType = _RHS_TYPE;                                                                          \
        static void run(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res,                    \
                        const typename Res::Scalar& alpha)                                                       \
        {                                                                                                        \
            LhsEval lhsEval(lhs);                                                                                \
                                                                                                                 \
            Index n = lhs.outerSize();                                                                           \
                                                                                                                 \
                                                                                                                 \
            for (Index c = 0; c < rhs.cols(); ++c)                                                               \
            {                                                                                                    \
                { /* This loop can be parallelized without syncronization */                                     \
                    for (Index i = 0; i < n; ++i) processRow(lhsEval, rhs, res, alpha, i, c);                    \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
                                                                                                                 \
        static void processRow(const LhsEval& lhsEval, const DenseRhsType& rhs, DenseResType& res,               \
                               const typename Res::Scalar&, Index i, Index col)                                  \
        {                                                                                                        \
            typename Res::Scalar tmp(0);                                                                         \
            for (LhsInnerIterator it(lhsEval, i); it; ++it)                                                      \
                res.coeffRef(i, col) += it.value() * rhs.coeff(it.index(), col);                                 \
            res.coeffRef(i, col) += tmp;                                                                         \
        }                                                                                                        \
    };


#define SAIGA_RM_CREATE_SMM_YX_RRC(_LHS_TYPE)                                                                    \
    namespace Eigen                                                                                              \
    {                                                                                                            \
    namespace internal                                                                                           \
    {                                                                                                            \
    template <typename Rhs, typename ResultType>                                                                 \
    struct conservative_sparse_sparse_product_selector<_LHS_TYPE, Rhs, ResultType, RowMajor, RowMajor, ColMajor> \
    {                                                                                                            \
        using Lhs = _LHS_TYPE;                                                                                   \
        static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res)                                         \
        {                                                                                                        \
            typedef SparseMatrix<typename ResultType::Scalar, RowMajor, typename ResultType::StorageIndex>       \
                RowMajorMatrix;                                                                                  \
            RowMajorMatrix resRow(lhs.rows(), rhs.cols());                                                       \
            conservative_sparse_sparse_product_impl_yx<Rhs, Lhs, RowMajorMatrix>(rhs, lhs, resRow);              \
            res = resRow;                                                                                        \
        }                                                                                                        \
    };                                                                                                           \
    }                                                                                                            \
    }

#define SAIGA_RM_CREATE_SMM_YX_RCC(_LHS_TYPE)                                                                    \
    namespace Eigen                                                                                              \
    {                                                                                                            \
    namespace internal                                                                                           \
    {                                                                                                            \
    template <typename Rhs, typename ResultType>                                                                 \
    struct conservative_sparse_sparse_product_selector<_LHS_TYPE, Rhs, ResultType, RowMajor, ColMajor, ColMajor> \
    {                                                                                                            \
        using Lhs = _LHS_TYPE;                                                                                   \
        static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res)                                         \
        {                                                                                                        \
            typedef SparseMatrix<typename Rhs::Scalar, RowMajor, typename ResultType::StorageIndex> RowMajorRhs; \
            typedef SparseMatrix<typename ResultType::Scalar, RowMajor, typename ResultType::StorageIndex>       \
                RowMajorRes;                                                                                     \
            RowMajorRhs rhsRow = rhs;                                                                            \
            RowMajorRes resRow(lhs.rows(), rhs.cols());                                                          \
            conservative_sparse_sparse_product_impl_yx<RowMajorRhs, Lhs, RowMajorRes>(rhsRow, lhs, resRow);      \
            res = resRow;                                                                                        \
        }                                                                                                        \
    };                                                                                                           \
    }                                                                                                            \
    }

#define SAIGA_RM_CREATE_SMM_YX_RCR(_LHS_TYPE)                                                                      \
    namespace Eigen                                                                                                \
    {                                                                                                              \
    namespace internal                                                                                             \
    {                                                                                                              \
    template <typename Rhs, typename ResultType>                                                                   \
    struct conservative_sparse_sparse_product_selector<_LHS_TYPE, Rhs, ResultType, RowMajor, ColMajor, RowMajor>   \
    {                                                                                                              \
        using Lhs = _LHS_TYPE;                                                                                     \
        static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res)                                           \
        {                                                                                                          \
            typedef SparseMatrix<typename Lhs::Scalar, ColMajor, typename ResultType::StorageIndex> ColMajorLhs;   \
            typedef SparseMatrix<typename ResultType::Scalar, ColMajor, typename ResultType::StorageIndex>         \
                ColMajorRes;                                                                                       \
            ColMajorLhs lhsCol = lhs;                                                                              \
            ColMajorRes resCol(lhs.rows(), rhs.cols());                                                            \
            internal::conservative_sparse_sparse_product_impl<ColMajorLhs, Rhs, ColMajorRes>(lhsCol, rhs, resCol); \
            res = resCol;                                                                                          \
        }                                                                                                          \
    };                                                                                                             \
    }                                                                                                              \
    }



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
