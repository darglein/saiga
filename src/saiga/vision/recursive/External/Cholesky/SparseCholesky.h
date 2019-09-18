/**
 * This file contains (modified) code from the Eigen library.
 * Eigen License:
 *
 * Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
 * Copyright (C) 2007-2011 Benoit Jacob <jacob.benoit.1@gmail.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla
 * Public License v. 2.0. If a copy of the MPL was not distributed
 * with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * ======================
 *
 * The modifications are part of the Eigen Recursive Matrix Extension (ERME).
 * ERME License:
 *
 * Copyright (c) 2019 Darius Rückert
 * Licensed under the MIT License.
 */

#pragma once

#include "../Core.h"


namespace Eigen::Recursive
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
    eigen_assert(A.rows() == A.cols());
    L.resize(A.rows(), A.cols());
    D.resize(A.rows());
    Dinv.resize(A.rows());


    for (int k = 0; k < A.outerSize(); ++k)
    {
        // compute Dj
        MatrixScalar sumd = AdditiveNeutral<MatrixScalar>::get();

        typename Eigen::SparseMatrix<MatrixScalar, Eigen::RowMajor>::InnerIterator it(A, k);


        for (int j = 0; j < k; ++j)

        {
            MatrixScalar sum = AdditiveNeutral<MatrixScalar>::get();

            // dot product in L of row i with row j
            // but only until column j
            typename Eigen::SparseMatrix<MatrixScalar, Eigen::RowMajor>::InnerIterator Li(L, k);
            typename Eigen::SparseMatrix<MatrixScalar, Eigen::RowMajor>::InnerIterator Lj(L, j);
            while (Li && Lj && Li.col() < j && Lj.col() < j)
            {
                if (Li.col() == Lj.col())
                {
                    //                    sum += Li.value() * D.diagonal()(Li.col()) * transpose(Lj.value());
                    removeMatrixScalar(sum) += removeMatrixScalar(Li.value()) *
                                               removeMatrixScalar(D.diagonal()(Li.col())) *
                                               removeMatrixScalar(transpose(Lj.value()));
                    //                    std::cout << "li" << std::endl << expand(Li.value()) << std::endl;
                    //                    std::cout << "lj" << std::endl << expand(Lj.value()) << std::endl;
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
                //                std::cout << "itcol==j it " << std::endl << expand(it.value()) << std::endl;
                //                std::cout << "itcol==j sum " << std::endl << expand(sum) << std::endl;
                sum = it.value() - sum;
                ++it;
            }
            else
            {
                //                std::cout << "itcol!=j sum " << std::endl << expand(sum) << std::endl;
                sum = -sum;
            }

            sum = sum * Dinv.diagonal()(j);
            //            std::cout << "test " << i << " " << j << std::endl << expand(sum) << std::endl << std::endl;
            L.insert(k, j) = sum;
            removeMatrixScalar(sumd) +=
                removeMatrixScalar(sum) * removeMatrixScalar(D.diagonal()(j)) * removeMatrixScalar(transpose(sum));
        }
        eigen_assert(it.col() == k);
        L.insert(k, k)     = MultiplicativeNeutral<MatrixScalar>::get();
        D.diagonal()(k)    = it.value() - sumd;
        Dinv.diagonal()(k) = inverseCholesky(D.diagonal()(k));

        //        std::cout << "computed " << k << "ths diagonal element: " << std::endl << expand(D.diagonal()(k)) << std::endl;
    }

#if 0
    // compute the product LDLT and compare it to the original matrix
    double factorizationError = (expand(L).template triangularView<Eigen::Lower>() * expand(D) *
                                     expand(L).template triangularView<Eigen::Lower>().transpose() -
                                 expand(A))
                                    .norm();
    std::cout << "sparse LDLT factorizationError " << factorizationError << std::endl;
    eigen_assert(factorizationError < 1e-10);
#endif
    //    std::cout << expand(Dinv) << std::endl << std::endl;
    //    std::cout << expand(L) << std::endl << std::endl;
}

template <typename MatrixType, typename VectorType>
VectorType SparseRecursiveLDLT<MatrixType, VectorType>::solve(const VectorType& b)
{
    eigen_assert(L.rows() == b.rows());
    VectorType x, y;
    x.resize(b.rows());
    y.resize(b.rows());

    x = forwardSubstituteDiagOne2(L, b);
    y = multDiagVector(Dinv, x);
    x = backwardSubstituteDiagOneTranspose2(L, y);

    return x;
}


}  // namespace Eigen::Recursive
