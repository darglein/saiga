/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "config.h"

namespace Eigen
{


enum StorageOptions {
    /** Storage order is column major (see \ref TopicStorageOrders). */
    ColMajor = 0,
    /** Storage order is row major (see \ref TopicStorageOrders). */
    RowMajor = 0x1,  // it is only a coincidence that this is equal to RowMajorBit -- don't rely on that
    /** Align the matrix itself if it is vectorizable fixed-size */
    AutoAlign = 0,
    /** Don't require alignment for the matrix itself (the array of coefficients, if dynamically allocated, may still be requested to be aligned) */ // FIXME --- clarify the situation
    DontAlign = 0x2
};



template <typename _Scalar, int _Rows, int _Cols, int _Options>
class Matrix;

template <typename T>
struct traits;


template <typename _Scalar, int _Rows, int _Cols, int _Options>
struct traits<Matrix<_Scalar, _Rows, _Cols, _Options>>
{
   public:
    typedef _Scalar Scalar;
};


template <typename Derived>
class MatrixBase
{
   public:
    using Scalar = typename traits<Derived>::Scalar;

    const Derived& derived() const { return *static_cast<const Derived*>(this); }
    Derived& derived() { return *static_cast<Derived*>(this); }

    Scalar& operator()(int i, int j) { return derived(i, j); }
    const Scalar& operator()(int i, int j) const { return derived(i, j); }
};

template <typename _Scalar, int _Rows, int _Cols, int _Options>
class Matrix : public MatrixBase<Matrix<_Scalar, _Rows, _Cols, _Options>>
{
   public:
    using Scalar              = _Scalar;
    using SameMatrix          = Matrix<_Scalar, _Rows, _Cols, _Options>;
    static constexpr int Size = _Rows * _Cols;
    static_assert(_Rows > 0, "Rows must be positive");
    static_assert(_Options == ColMajor, "Only colmajor supportet");

    Matrix() {}

    // Vector constructors
    Matrix(_Scalar x0, _Scalar x1)
    {
        static_assert(_Rows == 2 && _Cols == 1, "Constructor only valid for vectors.");
        _data[0] = x0;
        _data[1] = x1;
    }
    Matrix(_Scalar x0, _Scalar x1, _Scalar x2)
    {
        static_assert(_Rows == 2 && _Cols == 1, "Constructor only valid for vectors.");
        _data[0] = x0;
        _data[1] = x1;
        _data[2] = x2;
    }
    Matrix(_Scalar x0, _Scalar x1, _Scalar x2, _Scalar x3)
    {
        static_assert(_Rows == 2 && _Cols == 1, "Constructor only valid for vectors.");
        _data[0] = x0;
        _data[1] = x1;
        _data[2] = x2;
        _data[3] = x3;
    }

    _Scalar* data() { return data; }

    _Scalar& operator()(int i, int j) { return _data[j * _Rows + i]; }
    const _Scalar& operator()(int i, int j) const { return _data[j * _Rows + i]; }

    int Rows() { return _Rows; }
    int Cols() { return _Cols; }

    Scalar norm() const
    {
        Scalar result = 0;
        for (int i = 0; i < Size; ++i)
        {
            result += _data[i] * _data[i];
        }
        return sqrt(result);
    }

    SameMatrix normalized() const { return *this / norm(); }

    Scalar dot(const SameMatrix& other) const
    {
        static_assert(_Cols == 1, "Only valid for vectors.");
        Scalar result = 0;
        for (int i = 0; i < Size; ++i)
        {
            result += _data[i] * other._data[i];
        }
        return result;
    }

   private:
    _Scalar _data[Size];
};

template <typename _Scalar, int _Rows0, int _Cols0, int _Options0, int _Rows1, int _Cols1, int _Options1>
Matrix<_Scalar, _Rows0, _Cols1, _Options0> operator*(const Matrix<_Scalar, _Rows0, _Cols0, _Options0>& m1,
                                                     Matrix<_Scalar, _Rows1, _Cols1, _Options1> m2)
{
    Matrix<_Scalar, _Rows0, _Cols1, _Options0> result;
    return result;
}

template <typename _Scalar, int _Rows0, int _Cols0, int _Options0>
Matrix<_Scalar, _Rows0, _Cols0, _Options0> operator/(const Matrix<_Scalar, _Rows0, _Cols0, _Options0>& m1, _Scalar v)
{
    Matrix<_Scalar, _Rows0, _Cols0, _Options0> result;
    return result;
}

using Quaternionf = Quaternion<float>;
using Quaterniond = Quaternion<double>;

}  // namespace Eigen