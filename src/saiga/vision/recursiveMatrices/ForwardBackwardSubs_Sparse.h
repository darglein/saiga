/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/assert.h"
#include "saiga/vision/recursiveMatrices/Expand.h"
#include "saiga/vision/recursiveMatrices/NeutralElements.h"
#include "saiga/vision/recursiveMatrices/Transpose.h"

namespace Saiga
{
template <typename MatrixType, typename VectorType>
VectorType forwardSubstituteDiagOne2(const MatrixType& A, const VectorType& b)
{
    static_assert((MatrixType::Options & Eigen::RowMajorBit) == true, "The matrix must be row-major!");
    using Scalar = typename VectorType::Scalar;
    // solve Ax=b
    // with A triangular block matrix where diagonal elements are 1.
    VectorType x;  //(b.rows());
    x.resize(b.rows());
    for (int i = 0; i < A.outerSize(); ++i)
    {
        Scalar sum = AdditiveNeutral<Scalar>::get();

        for (typename MatrixType::InnerIterator it(A, i); it; ++it)
        {
            if (it.col() >= i) break;
            sum += it.value() * x(it.col());
        }
        x(i) = b(i) - sum;
    }

#if 0
    // Test if (Ax-b)==0
    double test = (expand(A.toDense()) * expand(x) - expand(b)).squaredNorm();
    cout << "error forwardSubstituteDiagOne: " << test << endl;
#endif
    return x;
}



// Works both for sparse and dense matrices
template <typename MatrixType, typename VectorType>
VectorType backwardSubstituteDiagOneTranspose2(const MatrixType& A, const VectorType& b)
{
    static_assert((MatrixType::Options & Eigen::RowMajorBit) == true, "The matrix must be row-major!");
    using Scalar = typename VectorType::Scalar;
    // solve Ax=b
    // with A triangular block matrix where diagonal elements are 1.
    VectorType x(b.rows());
    x.setZero();

    for (int i = A.outerSize() - 1; i >= 0; --i)
    {
        Scalar value = b(i) - x(i);
        typename MatrixType::ReverseInnerIterator it(A, i);

        SAIGA_ASSERT(it.col() == i);
        //        SAIGA_ASSERT(it.value() == 1);
        x(i) = value;
        --it;

        // subtract the current x from all x(j) with j < i
        for (; it; --it)
        {
            x(it.col()) += transpose(it.value()) * value;
        }
    }

#if 0
    // Test if (Ax-b)==0
    double test = (expand(A.toDense()).transpose() * expand(x) - expand(b)).squaredNorm();
    //    cout << A.transpose() << endl << endl;
    //    cout << x << endl << endl;
    //    cout << b << endl << endl;

    cout << "error backwardSubstituteDiagOneTranspose2: " << test << endl;
#endif
    return x;
}


}  // namespace Saiga
