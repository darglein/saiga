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
// template <typename _Scalar, typename _Scalar2, int _Rows, int _Cols>
template <typename MatrixType, typename VectorType>
VectorType forwardSubstituteDiagOne(const MatrixType& A, const VectorType& b)
{
    using Scalar = typename VectorType::Scalar;

    // solve Ax=b
    // with A triangular LOWER block matrix where diagonal elements are 1.
    VectorType x;
    x.resize(b.rows(), b.cols());

    for (int i = 0; i < A.rows(); ++i)
    {
        //        Scalar sum = AdditiveNeutral<Scalar>::get();
        Scalar sum = b(i);
        for (int j = 0; j < i; ++j)
        {
            sum -= A(i, j) * x(j);
        }
        x(i) = sum;
    }

#if 0
    // Test if (Ax-b)==0
    double test = (expand(A).template triangularView<Lower>() * expand(x) - expand(b)).squaredNorm();
    std::cout << "error forwardSubstituteDiagOne: " << test << std::endl;
    eigen_assert(test < 1e-10);
#endif
    return x;
}


// template <typename _Scalar, typename _Scalar2, int _Rows, int _Cols>
template <typename MatrixType>
MatrixType forwardSubstituteDiagOneMulti(const MatrixType& A, const MatrixType& b)
{
    // solve Ax=b
    // with A triangular block matrix where diagonal elements are 1.
    MatrixType x;
    x.resize(b.rows(), b.cols());


    using RowType = Matrix<typename MatrixType::Scalar, 1, MatrixType::ColsAtCompileTime>;
    RowType row;
    row.resize(b.cols());


    for (int i = 0; i < A.rows(); ++i)
    {
        row = AdditiveNeutral<RowType>::get();
        for (int j = 0; j < i; ++j)
        {
            row += A(i, j) * x.row(j);
        }
        x.row(i) = b.row(i) - row;
    }

#if 0
    // Test if (Ax-b)==0
    double test = (expand(A).template triangularView<Lower>() * expand(x) - expand(b)).squaredNorm();
    std::cout << "error forwardSubstituteDiagOneMulti: " << test << std::endl;
    eigen_assert(test < 1e-10);
#endif
    return x;
}


// template <typename _Scalar, typename _Scalar2, int _Rows, int _Cols>
template <typename MatrixType, typename VectorType>
VectorType backwardSubstituteDiagOneTranspose(const MatrixType& A, const VectorType& b)
{
    using Scalar = typename VectorType::Scalar;
    // solve Ax=b
    // with A triangular block matrix where diagonal elements are 1.
    VectorType x;
    x.resize(b.rows(), b.cols());

    for (int i = A.rows() - 1; i >= 0; --i)
    {
        Scalar sum = AdditiveNeutral<Scalar>::get();
        for (int j = i + 1; j < A.rows(); ++j)
        {
            sum += transpose(A(j, i)) * x(j);
        }
        x(i) = b(i) - sum;
    }

#if 0
    // Test if (Ax-b)==0
    double test = (expand(A).transpose().template triangularView<Upper>() * expand(x) - expand(b)).squaredNorm();
    std::cout << "error backwardSubstituteDiagOneTranspose: " << test << std::endl;
    eigen_assert(test < 1e-10);
#endif
    return x;
}


// template <typename _Scalar, typename _Scalar2, int _Rows, int _Cols>
template <typename MatrixType>
MatrixType backwardSubstituteDiagOneTransposeMulti(const MatrixType& A, const MatrixType& b)
{
    // solve Ax=b
    // with A triangular block matrix where diagonal elements are 1.
    MatrixType x;
    x.resize(b.rows(), b.cols());


    using RowType = Matrix<typename MatrixType::Scalar, 1, MatrixType::ColsAtCompileTime>;
    RowType row;
    row.resize(b.cols());

    for (int i = A.rows() - 1; i >= 0; --i)
    {
        row = AdditiveNeutral<RowType>::get();
        for (int j = i + 1; j < A.rows(); ++j)
        {
            row += transpose(A(j, i)) * x.row(j);
        }
        x.row(i) = b.row(i) - row;
    }

#if 0
    std::cout << fixedBlockMatrixToMatrix(x) << std::endl << std::endl;
    // Test if (Ax-b)==0
    double test = (fixedBlockMatrixToMatrix(A).transpose() * fixedBlockMatrixToMatrix(x) - fixedBlockMatrixToMatrix(b))
                      .squaredNorm();
    std::cout << "error backwardSubstituteDiagOneTranspose: " << test << std::endl;
#endif
    return x;
}



}  // namespace Eigen::Recursive
