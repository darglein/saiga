/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/assert.h"
#include "saiga/vision/BlockRecursiveBATemplates.h"
#include "saiga/vision/MatrixScalar.h"
#include "saiga/vision/recursiveMatrices/Cholesky.h"
#include "saiga/vision/recursiveMatrices/Expand.h"
#include "saiga/vision/recursiveMatrices/ForwardBackwardSubs.h"
#include "saiga/vision/recursiveMatrices/ForwardBackwardSubs_Sparse.h"
#include "saiga/vision/recursiveMatrices/Inverse.h"
#include "saiga/vision/recursiveMatrices/NeutralElements.h"
#include "saiga/vision/recursiveMatrices/Transpose.h"

namespace Saiga
{
template <typename MatrixType, typename VectorType>
struct SparseRecursiveLDLT
{
    using MatrixScalar = typename MatrixType::Scalar;
    using VectorScalar = typename VectorType::Scalar;

   public:
    // Computes L, D and Dinv
    void compute(const MatrixType& A);
    VectorType solve(const VectorType& b);

    MatrixType L;
    Eigen::DiagonalMatrix<MatrixScalar, MatrixType::RowsAtCompileTime> D;
    Eigen::DiagonalMatrix<MatrixScalar, MatrixType::RowsAtCompileTime> Dinv;
};

template <typename MatrixType, typename VectorType>
void SparseRecursiveLDLT<MatrixType, VectorType>::compute(const MatrixType& A)
{
    SAIGA_ASSERT(A.rows() == A.cols());
    L.resize(A.rows(), A.cols());
    D.resize(A.rows());
    Dinv.resize(A.rows());


    for (int i = 0; i < A.outerSize(); ++i)
    {
        // compute Dj
        MatrixScalar sumd = AdditiveNeutral<MatrixScalar>::get();

        typename Eigen::SparseMatrix<MatrixScalar, Eigen::RowMajor>::InnerIterator it(A, i);


        for (int j = 0; j < i; ++j)

        {
            MatrixScalar sum = AdditiveNeutral<MatrixScalar>::get();

            // dot product in L of row i with row j
            // but only until column j
            typename Eigen::SparseMatrix<MatrixScalar, Eigen::RowMajor>::InnerIterator Li(L, i);
            typename Eigen::SparseMatrix<MatrixScalar, Eigen::RowMajor>::InnerIterator Lj(L, j);
            while (Li && Lj && Li.col() < j && Lj.col() < j)
            {
                if (Li.col() == Lj.col())
                {
                    //                    sum += Li.value() * D.diagonal()(Li.col()) * transpose(Lj.value());
                    removeMatrixScalar(sum) += removeMatrixScalar(Li.value()) *
                                               removeMatrixScalar(D.diagonal()(Li.col())) *
                                               removeMatrixScalar(transpose(Lj.value()));
                    ++Li;
                    ++Lj;
                }
                else if (Li.col() < Lj.col())
                {
                    ++Li;
                }
                else
                {
                    ++Lj;
                }
            }
            if (it.col() == j)
            {
                sum = it.value() - sum;
                ++it;
            }
            else
            {
                sum = -sum;
            }

            sum            = sum * Dinv.diagonal()(j);
            L.insert(i, j) = sum;
            removeMatrixScalar(sumd) +=
                removeMatrixScalar(sum) * removeMatrixScalar(D.diagonal()(j)) * removeMatrixScalar(transpose(sum));
        }
        SAIGA_ASSERT(it.col() == i);
        L.insert(i, i)     = MultiplicativeNeutral<MatrixScalar>::get();
        D.diagonal()(i)    = it.value() - sumd;
        Dinv.diagonal()(i) = inverseCholesky(D.diagonal()(i));
    }

#if 0
    // compute the product LDLT and compare it to the original matrix
    double factorizationError = (expand(L).template triangularView<Eigen::Lower>() * expand(D) *
                                     expand(L).template triangularView<Eigen::Lower>().transpose() -
                                 expand(A))
                                    .norm();
    cout << "sparse LDLT factorizationError " << factorizationError << endl;
    SAIGA_ASSERT(factorizationError < 1e-10);
#endif
    //    cout << expand(Dinv) << endl << endl;
    //    cout << expand(L) << endl << endl;
}

template <typename MatrixType, typename VectorType>
VectorType SparseRecursiveLDLT<MatrixType, VectorType>::solve(const VectorType& b)
{
    SAIGA_ASSERT(L.rows() == b.rows());
    VectorType x, y;
    x.resize(b.rows());
    y.resize(b.rows());

    x = forwardSubstituteDiagOne2(L, b);
    y = multDiagVector(Dinv, x);
    x = backwardSubstituteDiagOneTranspose2(L, y);

    return x;
}


}  // namespace Saiga
