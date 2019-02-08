/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/vision/recursiveMatrices/MatrixScalar.h"


namespace Saiga
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
struct TransposeType<Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>
{
    // Recursive Typedef with switch row/cols
    using Type = Eigen::Matrix<typename TransposeType<_Scalar>::Type, _Cols, _Rows, _Options, _MaxCols, _MaxRows>;
};

template <typename _Scalar, int _Options>
struct TransposeType<Eigen::SparseMatrix<_Scalar, _Options>>
{
    using Type = Eigen::SparseMatrix<typename TransposeType<_Scalar>::Type, _Options>;
};

template <typename G>
struct TransposeType<MatrixScalar<G>>
{
    using Type = MatrixScalar<typename TransposeType<G>::Type>;
};


// ==================================================================================================

template <typename T>
struct Transpose
{
    using ReturnType = typename TransposeType<T>::Type;
    static ReturnType get(const T& m) { return (m.transpose().eval()); }
};

template <>
struct Transpose<double>
{
    static double get(double d) { return d; }
};

template <>
struct Transpose<float>
{
    static float get(float d) { return d; }
};

template <typename G>
struct Transpose<MatrixScalar<G>>
{
    using ReturnType = typename TransposeType<MatrixScalar<G>>::Type;
    static ReturnType get(const MatrixScalar<G>& m) { return makeMatrixScalar(Transpose<G>::get(m.get())); }
};


template <typename T>
auto transpose(const T& v)
{
    return Transpose<T>::get(v);
}

}  // namespace Saiga
