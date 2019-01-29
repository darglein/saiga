/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/assert.h"
#include "saiga/vision/recursiveMatrices/MatrixScalar.h"


#define SAIGA_RM_CREATE_RETURN(_LHS, _RHS, _RETURN)          \
    template <typename BinaryOp>                             \
    struct Eigen::ScalarBinaryOpTraits<_LHS, _RHS, BinaryOp> \
    {                                                        \
        typedef _RETURN ReturnType;                          \
    };


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
    namespace Eigen                                                                                              \
    {                                                                                                            \
    namespace internal                                                                                           \
    {                                                                                                            \
    template <typename SparseLhsType, typename DenseResType>                                                     \
    struct sparse_time_dense_product_impl<SparseLhsType, _RHS_TYPE, DenseResType, typename DenseResType::Scalar, \
                                          RowMajor, true>                                                        \
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
    };                                                                                                           \
    }                                                                                                            \
    }
