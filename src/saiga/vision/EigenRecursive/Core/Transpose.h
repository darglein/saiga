/**
 * This file is part of the Eigen Recursive Matrix Extension (ERME).
 *
 * Copyright (c) 2019 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "MatrixScalar.h"


namespace Eigen::Recursive
{
// ==================================================================================================
/**
 * Computes the transposed matrix type.
 *  - Same storage order as original type
 *  - Switched Rows-Cols
 */
template <typename T>
struct TransposeType
{
    using Type = T;
};


template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct TransposeType<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>
{
    // Recursive Typedef with switch row/cols
    using Type = Matrix<typename TransposeType<_Scalar>::Type, _Cols, _Rows, _Options>;
};

template <typename _Scalar, int _Options>
struct TransposeType<SparseMatrix<_Scalar, _Options>>
{
    using Type = SparseMatrix<typename TransposeType<_Scalar>::Type, _Options>;
};

template <typename G>
struct TransposeType<MatrixScalar<G>>
{
    using Type = MatrixScalar<typename TransposeType<G>::Type>;
};


// ==================================================================================================

template <typename T>
struct MSTranspose
{
    using ReturnType = typename TransposeType<T>::Type;
    static ReturnType get(const T& m) { return (m.transpose().eval()); }
};

template <>
struct MSTranspose<double>
{
    static double get(double d) { return d; }
};

template <>
struct MSTranspose<float>
{
    static float get(float d) { return d; }
};

template <typename G>
struct MSTranspose<MatrixScalar<G>>
{
    using ReturnType = typename TransposeType<MatrixScalar<G>>::Type;
    static ReturnType get(const MatrixScalar<G>& m) { return makeMatrixScalar(MSTranspose<G>::get(m.get())); }
};


template <typename T>
auto transpose(const T& v)
{
    return MSTranspose<T>::get(v);
}

}  // namespace Eigen::Recursive
