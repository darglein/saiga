/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/assert.h"
#include "saiga/vision/BlockRecursiveBATemplates.h"
#include "saiga/vision/MatrixScalar.h"
#include "saiga/vision/recursiveMatrices/Expand.h"
#include "saiga/vision/recursiveMatrices/ForwardBackwardSubs.h"
#include "saiga/vision/recursiveMatrices/Inverse.h"
#include "saiga/vision/recursiveMatrices/NeutralElements.h"
#include "saiga/vision/recursiveMatrices/Transpose.h"

namespace Saiga
{
template <typename T>
auto inverseCholesky(const T& v);


template <typename MatrixType, typename VectorType>
struct DenseLDLT
{
    using MatrixScalar = typename MatrixType::Scalar;
    using VectorScalar = typename VectorType::Scalar;

   public:
    // Computes L, D and Dinv
    void compute(const MatrixType& A);
    VectorType solve(const VectorType& b);
    MatrixType solve(const MatrixType& b);
    MatrixType invert();

    MatrixType L;
    Eigen::DiagonalMatrix<MatrixScalar, MatrixType::RowsAtCompileTime> D;
    Eigen::DiagonalMatrix<MatrixScalar, MatrixType::RowsAtCompileTime> Dinv;
};

template <typename MatrixType, typename VectorType>
void DenseLDLT<MatrixType, VectorType>::compute(const MatrixType& A)
{
    SAIGA_ASSERT(A.rows() == A.cols());
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

            L(i, j) = (A(i, j) - sum) * Dinv.diagonal()(j);
            //            L(j, i) = AdditiveNeutral<MatrixScalar>::get();
            sumd += L(i, j) * D.diagonal()(j) * transpose(L(i, j));
        }
        //        L(i, i)            = MultiplicativeNeutral<MatrixScalar>::get();
        D.diagonal()(i)    = A(i, i) - sumd;
        Dinv.diagonal()(i) = inverseCholesky(D.diagonal()(i));
    }
}

template <typename MatrixType, typename VectorType>
VectorType DenseLDLT<MatrixType, VectorType>::solve(const VectorType& b)
{
    SAIGA_ASSERT(L.rows() == b.rows());
    VectorType x, y;
    x.resize(b.rows());
    y.resize(b.rows());

    x = forwardSubstituteDiagOne(L, b);
    y = multDiagVector(Dinv, x);
    x = backwardSubstituteDiagOneTranspose(L, y);

    return x;
}

template <typename MatrixType, typename VectorType>
MatrixType DenseLDLT<MatrixType, VectorType>::solve(const MatrixType& b)
{
    SAIGA_ASSERT(L.rows() == b.rows());
    MatrixType x, y;
    x.resize(L.rows(), L.cols());
    y.resize(L.rows(), L.cols());

    x = forwardSubstituteDiagOne<MatrixType>(L, b);
    y = multDiagVectorMulti(Dinv, x);
    x = backwardSubstituteDiagOneTranspose(L, y);

    return x;
}


template <typename MatrixType, typename VectorType>
MatrixType DenseLDLT<MatrixType, VectorType>::invert()
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
        using Scalar     = typename T::Scalar;
        using VectorType = Eigen::Matrix<Scalar, T::RowsAtCompileTime, 1>;
        DenseLDLT<T, VectorType> ldlt;
        ldlt.compute(m);
        return ldlt.invert();
    }
};

template <>
struct InverseCholeskyImpl<double>
{
    static double get(double d) { return 1.0 / d; }
};

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


}  // namespace Saiga
