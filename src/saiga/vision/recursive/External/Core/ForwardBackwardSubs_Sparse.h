/**
 * This file is part of the Eigen Recursive Matrix Extension (ERME).
 *
 * Copyright (c) 2019 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "Expand.h"
#include "NeutralElements.h"
#include "Transpose.h"

namespace Eigen::Recursive
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
    std::cout << "error forwardSubstituteDiagOne: " << test << std::endl;
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

        eigen_assert(it.col() == i);
        //        eigen_assert(it.value() == 1);
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
    //    std::cout << A.transpose() << std::endl << std::endl;
    //    std::cout << x << std::endl << std::endl;
    //    std::cout << b << std::endl << std::endl;

    std::cout << "error backwardSubstituteDiagOneTranspose2: " << test << std::endl;
#endif
    return x;
}


}  // namespace Eigen::Recursive
