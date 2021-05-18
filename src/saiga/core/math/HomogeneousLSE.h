/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "math.h"

#include <Eigen/QR>
#include <Eigen/SVD>

/**
 * Solves a homogeneous system of linear equations of the form
 *          Ax = 0
 * , where A is a singular matrix (rank defficient).
 *
 * Recommendation:
 * Just use the QR decomposition for all problems. (It's fast and precise)
 */
namespace Saiga
{
// Fastest one but also least accurate.
// Not stable if A is close to full rank
template <typename Matrix, typename Vector, bool normalize = true>
void solveHomogeneousLU(const Matrix& A, Vector& x)
{
    Eigen::FullPivLU<Matrix> lu(A);
    x = lu.kernel();
    if constexpr (normalize) x.normalize();
}

// Slightly slower than LU but also more precise
// -> Use this method if you are not sure
template <typename Matrix, typename Vector, bool normalize = true>
void solveHomogeneousQR(const Matrix& A, Vector& x)
{
    Eigen::ColPivHouseholderQR<Matrix> qr(A.transpose());
    Matrix Q = qr.householderQ();
    x        = Q.col(A.rows() - 1);
    if constexpr (normalize) x.normalize();
}

// roughly identical to QR (slightly slower than QR in my tests)
// Not stable if A is close to full rank
template <typename Matrix, typename Vector, bool normalize = true>
void solveHomogeneousCOD(const Matrix& A, Vector& x)
{
    Eigen::CompleteOrthogonalDecomposition<Matrix> cod(A);
    Matrix Z = cod.matrixZ().transpose();
    x        = cod.colsPermutation() * Z.col(A.rows() - 1);
    if constexpr (normalize) x.normalize();
}

// A lot slower than the methods above.
// Accuracy is similar to QR
template <typename Matrix, typename Vector, bool normalize = true>
void solveHomogeneousJacobiSVD(const Matrix& A, Vector& x)
{
    Eigen::JacobiSVD<Matrix> svd(A, Eigen::ComputeFullV);
    x = svd.matrixV().col(A.cols() - 1);
    if constexpr (normalize) x.normalize();
}

// Even slower than jacobi SVD. (for small/medium matrices)
template <typename Matrix, typename Vector, bool normalize = true>
void solveHomogeneousBDCSVD(const Matrix& A, Vector& x)
{
    Eigen::BDCSVD<Matrix> svd(A, Eigen::ComputeFullV);
    x = svd.matrixV().col(A.rows() - 1);
    if constexpr (normalize) x.normalize();
}

}  // namespace Saiga
