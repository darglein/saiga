/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/assert.h"
#include "saiga/vision/MatrixScalar.h"
#include "saiga/vision/recursiveMatrices/Expand.h"



namespace Saiga
{
template <typename LHS, typename RHS, typename DiagType>
void diagInnerProductTransposed(const LHS& lhs, const RHS& rhsTransposed, DiagType& res)
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
void multSparseRowTransposedVector(const LHS& lhsTransposed, const RHS& rhs, RES& res)
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
