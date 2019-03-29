/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
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
                    //                    cout << "li" << endl << expand(Li.value()) << endl;
                    //                    cout << "lj" << endl << expand(Lj.value()) << endl;
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
                //                cout << "itcol==j it " << endl << expand(it.value()) << endl;
                //                cout << "itcol==j sum " << endl << expand(sum) << endl;
                sum = it.value() - sum;
                ++it;
            }
            else
            {
                //                cout << "itcol!=j sum " << endl << expand(sum) << endl;
                sum = -sum;
            }

            sum = sum * Dinv.diagonal()(j);
            //            cout << "test " << i << " " << j << endl << expand(sum) << endl << endl;
            L.insert(k, j) = sum;
            removeMatrixScalar(sumd) +=
                removeMatrixScalar(sum) * removeMatrixScalar(D.diagonal()(j)) * removeMatrixScalar(transpose(sum));
        }
        SAIGA_ASSERT(it.col() == k);
        L.insert(k, k)     = MultiplicativeNeutral<MatrixScalar>::get();
        D.diagonal()(k)    = it.value() - sumd;
        Dinv.diagonal()(k) = inverseCholesky(D.diagonal()(k));

        //        cout << "computed " << k << "ths diagonal element: " << endl << expand(D.diagonal()(k)) << endl;
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


}  // namespace Eigen::Recursive
