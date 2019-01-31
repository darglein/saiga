/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/vision/recursiveMatrices/Expand.h"
#include "saiga/vision/recursiveMatrices/ForwardBackwardSubs.h"
#include "saiga/vision/recursiveMatrices/Inverse.h"
#include "saiga/vision/recursiveMatrices/MatrixScalar.h"
#include "saiga/vision/recursiveMatrices/NeutralElements.h"
#include "saiga/vision/recursiveMatrices/SparseInnerProduct.h"
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
    cout << "dense LDLT factorizationError " << factorizationError << endl;
    SAIGA_ASSERT(factorizationError < 1e-10);
#endif
    //    cout << expand(Dinv) << endl << endl;
    //    cout << expand(L) << endl << endl;
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

    x = forwardSubstituteDiagOneMulti<MatrixType>(L, b);
    y = multDiagVectorMulti(Dinv, x);
    x = backwardSubstituteDiagOneTransposeMulti(L, y);

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


template <>
struct InverseCholeskyImpl<float>
{
    static float get(float d) { return 1.0 / d; }
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
