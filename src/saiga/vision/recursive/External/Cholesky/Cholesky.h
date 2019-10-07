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
template <typename T>
auto inverseCholesky(const T& v);


template <typename MatrixType>
struct DenseLDLT
{
    using MatrixScalar = typename MatrixType::Scalar;

   public:
    // Computes L, D and Dinv
    void compute(const MatrixType& A);

    template <typename VectorType>
    VectorType solve(const VectorType& b);

    MatrixType solve(const MatrixType& b);
    MatrixType invert();

    MatrixType L;
    Eigen::DiagonalMatrix<MatrixScalar, MatrixType::RowsAtCompileTime> D;
    Eigen::DiagonalMatrix<MatrixScalar, MatrixType::RowsAtCompileTime> Dinv;
};

template <typename MatrixType>
void DenseLDLT<MatrixType>::compute(const MatrixType& A)
{
    eigen_assert(A.rows() == A.cols());
    L.resize(A.rows(), A.cols());
    D.resize(A.rows());
    Dinv.resize(A.rows());

    // compute L
    for (int i = 0; i < A.rows(); i++)
    {
        // compute Dj
        MatrixScalar sumd = AdditiveNeutral<MatrixScalar>::get();

        for (int j = 0; j < i; ++j)
        {
            // compute all l's for this row
            MatrixScalar sum = AdditiveNeutral<MatrixScalar>::get();

            // dot product of row i with row j
            // but only until column j
            for (int k = 0; k < j; ++k)
            {
                sum += L(i, k) * D.diagonal()(k) * transpose(L(j, k));
            }

            sum     = A(i, j) - sum;
            sum     = sum * Dinv.diagonal()(j);
            L(i, j) = sum;
            L(j, i) = sum;
            sumd += sum * D.diagonal()(j) * transpose(sum);
        }
        L(i, i)            = MultiplicativeNeutral<MatrixScalar>::get();
        D.diagonal()(i)    = A(i, i) - sumd;
        Dinv.diagonal()(i) = inverseCholesky(D.diagonal()(i));
    }


#if 0
    // compute the product LDLT and compare it to the original matrix
    double factorizationError = (expand(L).template triangularView<Eigen::Lower>() * expand(D) *
                                     expand(L).template triangularView<Eigen::Lower>().transpose() -
                                 expand(A))
                                    .norm();
    std::cout << "dense LDLT factorizationError " << factorizationError << std::endl;
    eigen_assert(factorizationError < 1e-10);
#endif
    //    std::cout << expand(Dinv) << std::endl << std::endl;
    //    std::cout << expand(L) << std::endl << std::endl;
}


template <typename MatrixType>
template <typename VectorType>
VectorType DenseLDLT<MatrixType>::solve(const VectorType& b)
{
    eigen_assert(L.rows() == b.rows());
    VectorType x, y;
    x.resize(b.rows());
    y.resize(b.rows());

    x = forwardSubstituteDiagOne(L, b);
    y = multDiagVector(Dinv, x);
    x = backwardSubstituteDiagOneTranspose(L, y);

    return x;
}

template <typename MatrixType>
MatrixType DenseLDLT<MatrixType>::solve(const MatrixType& b)
{
    eigen_assert(L.rows() == b.rows());
    MatrixType x, y;
    x.resize(L.rows(), L.cols());
    y.resize(L.rows(), L.cols());

    x = forwardSubstituteDiagOneMulti<MatrixType>(L, b);
    y = multDiagVectorMulti(Dinv, x);
    x = backwardSubstituteDiagOneTransposeMulti(L, y);

    return x;
}


template <typename MatrixType>
MatrixType DenseLDLT<MatrixType>::invert()
{
    MatrixType id = MultiplicativeNeutral<MatrixType>::get(L.rows(), L.cols());
    return solve(id);
}



template <typename T>
struct InverseCholeskyImpl
{
    static T get(const T& m)
    {
        static_assert(T::RowsAtCompileTime == T::ColsAtCompileTime,
                      "The Symmetric Inverse is only defined for square matrices!");
        //        using Scalar = typename T::Scalar;
        //        using VectorType = Eigen::Matrix<Scalar, T::RowsAtCompileTime, 1>;
        DenseLDLT<T> ldlt;
        ldlt.compute(m);
        return ldlt.invert();
    }
};

template <>
struct InverseCholeskyImpl<double>
{
    static double get(double d) { return 1.0 / d; }
};


template <>
struct InverseCholeskyImpl<float>
{
    static float get(float d) { return 1.0f / d; }
};

#if 1
template <typename Derived>
struct InverseCholeskyImpl<Eigen::MatrixBase<Derived>>
{
    using MatrixType = Eigen::MatrixBase<Derived>;
    static MatrixType get(const MatrixType& m) { return m.ldlt().solve(MatrixType::Identity()); }
};

// spezialization for matrices of float and double
// use the eigen inverse here instead of our recursive implementation
// template <int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
// struct InverseCholeskyImpl<Eigen::Matrix<double, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>
//{
//    using MatrixType = Eigen::Matrix<double, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;
//    static MatrixType get(const MatrixType& m) { return m.ldlt().solve(MatrixType::Identity()); }
//};
// template <int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
// struct InverseCholeskyImpl<Eigen::Matrix<float, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>
//{
//    using MatrixType = Eigen::Matrix<float, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;
//    static MatrixType get(const MatrixType& m) { return m.ldlt().solve(MatrixType::Identity()); }
//};
#endif

template <typename G>
struct InverseCholeskyImpl<MatrixScalar<G>>
{
    static MatrixScalar<G> get(const MatrixScalar<G>& m)
    {
        return MatrixScalar<G>(InverseCholeskyImpl<G>::get(m.get()));
    }
};



template <typename T>
auto inverseCholesky(const T& v)
{
    return InverseCholeskyImpl<T>::get(v);
}


}  // namespace Eigen::Recursive
