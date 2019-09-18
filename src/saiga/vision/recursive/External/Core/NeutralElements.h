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
/**
 * Additive neutral element e
 *
 * A + e = A
 */
template <typename T>
struct AdditiveNeutral
{
    static T get() { return T::Zero(); }
    static T get(int rows, int cols) { return T::Zero(rows, cols); }
    static void setZero(T& d) { d.setZero(); }
};

template <>
struct AdditiveNeutral<double>
{
    static double get() { return 0.0; }
    static double get(int rows, int cols) { return 0.0; }
    static void setZero(double& d) { d = 0; }
};

template <>
struct AdditiveNeutral<float>
{
    static float get() { return 0.0f; }
    static float get(int rows, int cols) { return 0.0f; }
    static void setZero(float& d) { d = 0.f; }
};

template <typename G>
struct AdditiveNeutral<MatrixScalar<G>>
{
    static MatrixScalar<G> get() { return MatrixScalar<G>(AdditiveNeutral<G>::get()); }
    static MatrixScalar<G> get(int rows, int cols) { return MatrixScalar<G>(AdditiveNeutral<G>::get(rows, cols)); }
    static void setZero(MatrixScalar<G>& d) { AdditiveNeutral<G>::setZero(d.get()); }
};

template <typename T>
void setZero(T& v)
{
    AdditiveNeutral<T>::setZero(v);
}


/// =================================================================================================


/**
 * Multiplicative neutral element e
 *
 * A * e = A
 */
template <typename T>
struct MultiplicativeNeutral
{
};

template <>
struct MultiplicativeNeutral<double>
{
    static double get() { return 1.0; }
};

template <>
struct MultiplicativeNeutral<float>
{
    static float get() { return 1.0; }
};

template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct MultiplicativeNeutral<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>
{
    using MatrixType = Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;

    static MatrixType get()
    {
        static_assert(_Rows >= 0 && _Cols >= 0, "The matrix must be fixed size.");
        static_assert(_Rows == _Cols, "The matrix must be square");


        MatrixType A = AdditiveNeutral<MatrixType>::get();
        for (int i = 0; i < A.rows(); ++i)
        {
            A(i, i) = MultiplicativeNeutral<_Scalar>::get();
        }
        return A;
    }

    static MatrixType get(int rows, int cols)
    {
        eigen_assert(rows == cols);

        MatrixType A = AdditiveNeutral<MatrixType>::get(rows, cols);
        for (int i = 0; i < A.rows(); ++i)
        {
            A(i, i) = MultiplicativeNeutral<_Scalar>::get();
        }
        return A;
    }
};



template <typename G>
struct MultiplicativeNeutral<MatrixScalar<G>>
{
    static MatrixScalar<G> get() { return MatrixScalar<G>(MultiplicativeNeutral<G>::get()); }

    static MatrixScalar<G> get(int rows, int cols)
    {
        return MatrixScalar<G>(MultiplicativeNeutral<G>::get(rows, cols));
    }
};


}  // namespace Eigen::Recursive
