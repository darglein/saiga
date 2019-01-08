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
struct SparseLDLT
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
void SparseLDLT<MatrixType, VectorType>::compute(const MatrixType& A)
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
                    sum += Li.value() * D.diagonal()(Li.col()) * transpose(Lj.value());
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
            sumd += sum * D.diagonal()(j) * transpose(sum);
        }
        SAIGA_ASSERT(it.col() == i);
        L.insert(i, i)     = MultiplicativeNeutral<MatrixScalar>::get();
        D.diagonal()(i)    = it.value() - sumd;
        Dinv.diagonal()(i) = inverseCholesky(D.diagonal()(i));
    }
}

template <typename MatrixType, typename VectorType>
VectorType SparseLDLT<MatrixType, VectorType>::solve(const VectorType& b)
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
