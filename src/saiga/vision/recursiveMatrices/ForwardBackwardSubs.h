/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/assert.h"
#include "saiga/vision/recursiveMatrices/NeutralElements.h"

namespace Saiga
{
template <typename _Scalar, typename _Scalar2, int _Rows, int _Cols>
Eigen::Matrix<_Scalar2, _Rows, 1> forwardSubstituteDiagOne(const Eigen::Matrix<_Scalar, _Rows, _Cols>& A,
                                                           const Eigen::Matrix<_Scalar2, _Rows, 1>& b)
{
    // solve Ax=b
    // with A triangular block matrix where diagonal elements are 1.
    Eigen::Matrix<_Scalar2, _Rows, 1> x;
    for (int i = 0; i < _Rows; ++i)
    {
        _Scalar2 sum = AdditiveNeutral<_Scalar2>::get();
        for (int j = 0; j < i; ++j)
        {
            sum += A(i, j) * x(j);
        }
        x(i) = b(i) - sum;
    }

#if 0
    // Test if (Ax-b)==0
    double test =
        (fixedBlockMatrixToMatrix(A) * fixedBlockMatrixToMatrix(x) - fixedBlockMatrixToMatrix(b)).squaredNorm();
    cout << "error forwardSubstituteDiagOne: " << test << endl;
#endif
    return x;
}


template <typename MatrixType, typename VectorType>
VectorType forwardSubstituteDiagOne2(const MatrixType& A, const VectorType& b)
{
    static_assert((MatrixType::Options | Eigen::RowMajorBit) == true, "The matrix must be row-major!");
    using Scalar = typename VectorType::Scalar;
    // solve Ax=b
    // with A triangular block matrix where diagonal elements are 1.
    VectorType x(b.rows());
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
    double test =
        (fixedBlockMatrixToMatrix(A) * fixedBlockMatrixToMatrix(x) - fixedBlockMatrixToMatrix(b)).squaredNorm();
    cout << "error forwardSubstituteDiagOne: " << test << endl;
#endif
    return x;
}


template <typename _Scalar, typename _Scalar2, int _Rows, int _Cols>
Eigen::Matrix<_Scalar2, _Rows, 1> backwardSubstituteDiagOneTranspose(const Eigen::Matrix<_Scalar, _Rows, _Cols>& A,
                                                                     const Eigen::Matrix<_Scalar2, _Rows, 1>& b)
{
    // solve Ax=b
    // with A triangular block matrix where diagonal elements are 1.
    Eigen::Matrix<_Scalar2, _Rows, 1> x;
    for (int i = _Rows - 1; i >= 0; --i)
    {
        _Scalar2 sum = AdditiveNeutral<_Scalar2>::get();
        for (int j = i + 1; j < _Rows; ++j)
        {
            sum += A(j, i).transpose() * x(j);
        }
        x(i) = b(i) - sum;
    }

#if 0
    cout << fixedBlockMatrixToMatrix(x) << endl << endl;
    // Test if (Ax-b)==0
    double test = (fixedBlockMatrixToMatrix(A).transpose() * fixedBlockMatrixToMatrix(x) - fixedBlockMatrixToMatrix(b))
                      .squaredNorm();
    cout << "error backwardSubstituteDiagOneTranspose: " << test << endl;
#endif
    return x;
}


// Works both for sparse and dense matrices
template <typename MatrixType, typename VectorType>
VectorType backwardSubstituteDiagOneTranspose2(const MatrixType& A, const VectorType& b)
{
    static_assert((MatrixType::Options | Eigen::RowMajorBit) == true, "The matrix must be row-major!");
    using Scalar = typename VectorType::Scalar;
    // solve Ax=b
    // with A triangular block matrix where diagonal elements are 1.
    VectorType x(b.rows());
    for (int i = A.outerSize() - 1; i >= 0; ++i)
    {
        Scalar sum = AdditiveNeutral<Scalar>::get();

        for (typename MatrixType::InnerIterator it(A, i); it; ++it)
        {
            if (it.col() >= i + 1)
            {
                sum += it.value() * x(it.col());
            }
        }
        x(i) = b(i) - sum;
    }

#if 0
    // Test if (Ax-b)==0
    double test =
        (fixedBlockMatrixToMatrix(A) * fixedBlockMatrixToMatrix(x) - fixedBlockMatrixToMatrix(b)).squaredNorm();
    cout << "error forwardSubstituteDiagOne: " << test << endl;
#endif
    return x;
}


}  // namespace Saiga
