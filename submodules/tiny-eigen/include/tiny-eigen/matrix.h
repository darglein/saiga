/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "config.h"

#include <algorithm>
#include <cmath>
#include <iostream>

// don't need to do anything here because we don't use aligned vectors
#define EIGEN_MAKE_ALIGNED_OPERATOR_NEW

namespace Eigen
{
template <typename _Scalar, int _Rows, int _Cols, int _Options>
class Matrix;
template <typename _Scalar, int _Rows, int _Cols, int _Options>
class MatrixView;
template <typename _Scalar, int _Rows, int _Cols, int _Options>
class Array;
namespace internal
{

template <typename T>
struct traits;


template <typename _Scalar, int _Rows, int _Cols, int _Options>
struct traits<Matrix<_Scalar, _Rows, _Cols, _Options>>
{
   public:
    typedef _Scalar Scalar;
    using DenseReturnType = Matrix<_Scalar, _Rows, _Cols, _Options>;
};

template <typename _Scalar, int _Rows, int _Cols, int _Options>
struct traits<MatrixView<_Scalar, _Rows, _Cols, _Options>>
{
   public:
    typedef _Scalar Scalar;
    using DenseReturnType = Matrix<_Scalar, _Rows, _Cols, _Options>;
};

template <typename _Scalar, int _Rows, int _Cols, int _Options>
struct traits<Array<_Scalar, _Rows, _Cols, _Options>>
{
   public:
    typedef _Scalar Scalar;
    using DenseReturnType = Array<_Scalar, _Rows, _Cols, _Options>;
};

}  // namespace internal

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

template <typename Derived>
class MatrixBase
{
   public:
    using Scalar          = typename internal::traits<Derived>::Scalar;
    using DenseReturnType = typename internal::traits<Derived>::DenseReturnType;
    using SameMatrix      = Derived;
    using PlainObject     = Derived;

    HD const Derived& derived() const { return *static_cast<const Derived*>(this); }
    HD Derived& derived() { return *static_cast<Derived*>(this); }

    HD Scalar& operator()(int i, int j) { return derived()(i, j); }
    HD const Scalar& operator()(int i, int j) const { return derived()(i, j); }

    HD Scalar& operator()(int i) { return derived()(i); }
    HD const Scalar& operator()(int i) const { return derived()(i); }

    HD Scalar& at(int i) { return derived().at(i); }
    HD const Scalar& at(int i) const { return derived().at(i); }


    HD int rows() const { return derived().rows(); }
    HD int cols() const { return derived().cols(); }
    HD int size() const { return derived().size(); }

    HD Scalar minCoeff() const
    {
        Scalar result = derived().at(0);
        for (int i = 1; i < derived().size(); ++i)
        {
            result = std::min(result, derived().at(i));
        }
        return result;
    }

    HD Scalar sum() const
    {
        Scalar result = 0;
        for (int i = 0; i < derived().size(); ++i)
        {
            result += derived().at(i);
        }
        return result;
    }

    HD Scalar prod() const
    {
        Scalar result = 1;
        for (int i = 0; i < derived().size(); ++i)
        {
            result *= derived().at(i);
        }
        return result;
    }


    HD Scalar maxCoeff() const
    {
        Scalar result = derived().at(0);
        for (int i = 1; i < derived().size(); ++i)
        {
            result = std::max(result, derived().at(i));
        }
        return result;
    }


    HD Scalar norm() const
    {
        Scalar result = 0;
        for (int i = 0; i < derived().size(); ++i)
        {
            result += derived().at(i) * derived().at(i);
        }
        return sqrt(result);
    }

    HD Scalar squaredNorm() const
    {
        Scalar result = 0;
        for (int i = 0; i < derived().size(); ++i)
        {
            result += derived().at(i) * derived().at(i);
        }
        return result;
    }

    HD DenseReturnType eval() const { return derived(); }

    HD DenseReturnType normalized() const { return derived() / norm(); }


    HD SameMatrix& normalize()
    {
        derived() = derived() / norm();
        return derived();
    }

    HD Scalar dot(const SameMatrix& other) const
    {
        Scalar result = 0;
        for (int i = 0; i < derived().size(); ++i)
        {
            result += derived().at(i) * other.derived().at(i);
        }
        return result;
    }

    HD DenseReturnType cross(const SameMatrix& other) const
    {
        DenseReturnType result;
        result.at(0) = at(1) * other.at(2) - at(2) * other.at(1);
        result.at(1) = at(2) * other.at(0) - at(0) * other.at(2);
        result.at(2) = at(0) * other.at(1) - at(1) * other.at(0);
        return result;
    }



    HD DenseReturnType operator-() const
    {
        DenseReturnType result;
        for (int i = 0; i < rows(); ++i)
        {
            for (int j = 0; j < cols(); ++j)
            {
                result(i, j) = -derived()(i, j);
            }
        }
        return result;
    }


    HD void setZero()
    {
        for (int i = 0; i < rows(); ++i)
        {
            for (int j = 0; j < cols(); ++j)
            {
                derived()(i, j) = 0;
            }
        }
    }


    HD void setOnes()
    {
        for (int i = 0; i < rows(); ++i)
        {
            for (int j = 0; j < cols(); ++j)
            {
                derived()(i, j) = 1;
            }
        }
    }

    HD DenseReturnType inverse() const
    {
        int N               = cols();
        DenseReturnType mat = DenseReturnType::Identity();
        // mat.setZero();

        DenseReturnType m = derived();

        // code from
        // https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/matrix-inverse
        for (unsigned column = 0; column < N; ++column)
        {
            // Swap row in case our pivot point is not working
            if (m(column, column) == 0)
            {
                unsigned big = column;
                for (unsigned row = 0; row < N; ++row)
                    if (fabs(m(row, column)) > fabs(m(big, column))) big = row;
                if (big == column)
                {
                    // Print this is a singular matrix, return identity ?
#ifdef SAIGA_HOST
                    printf("inverse error");
#endif
                    return mat;
                }
                else
                {
                    for (unsigned j = 0; j < N; ++j)
                    {
                        // std::swap(m(column, j), m(big, j));
                        // std::swap(mat(column, j), mat(big, j));

                        auto tmp1    = m(column, j);
                        m(column, j) = m(big, j);
                        m(big, j)    = tmp1;

                        auto tmp2      = mat(column, j);
                        mat(column, j) = mat(big, j);
                        mat(big, j)    = tmp2;
                    }
                }
            }
            // Set each row in the column to 0
            for (unsigned row = 0; row < N; ++row)
            {
                if (row != column)
                {
                    Scalar coeff = m(row, column) / m(column, column);
                    if (coeff != 0)
                    {
                        for (unsigned j = 0; j < N; ++j)
                        {
                            m(row, j) -= coeff * m(column, j);
                            mat(row, j) -= coeff * mat(column, j);
                        }
                        // Set the element to 0 for safety
                        m(row, column) = 0;
                    }
                }
            }
        }
        // Set each element of the diagonal to 1
        for (unsigned row = 0; row < N; ++row)
        {
            for (unsigned column = 0; column < N; ++column)
            {
                mat(row, column) /= m(row, row);
            }
        }

        return mat;
    }

    HD DenseReturnType round() const
    {
        DenseReturnType result;
        for (int i = 0; i < derived().size(); ++i)
        {
            result.at(i) = std::round(derived().at(i));
        }
        return result;
    }

    HD bool allFinite() const
    {
        for (int i = 0; i < rows(); ++i)
        {
            for (int j = 0; j < cols(); ++j)
            {
                if (!std::isfinite(derived()(i, j)))
                {
                    return false;
                }
            }
        }
        return true;
    }

    HD bool isZero() const
    {
        for (int i = 0; i < rows(); ++i)
        {
            for (int j = 0; j < cols(); ++j)
            {
                if (derived()(i, j) != 0)
                {
                    return false;
                }
            }
        }
        return true;
    }

    HD bool operator!=(SameMatrix other) const { return !((*this) == other); }

    HD bool operator==(SameMatrix other) const
    {
        bool result = true;
        for (int i = 0; i < rows(); ++i)
        {
            for (int j = 0; j < cols(); ++j)
            {
                result &= derived()(i, j) == other.derived()(i, j);
            }
        }
        return result;
    }

    HD DenseReturnType operator&&(SameMatrix other) const
    {
        DenseReturnType result;
        for (int i = 0; i < derived().size(); ++i)
        {
            result.at(i) = derived().at(i) && other.derived().at(i);
        }
        return result;
    }

    HD DenseReturnType operator<=(SameMatrix other) const
    {
        DenseReturnType result;
        for (int i = 0; i < derived().size(); ++i)
        {
            result.at(i) = derived().at(i) <= other.derived().at(i);
        }
        return result;
    }

    HD DenseReturnType operator<(SameMatrix other) const
    {
        DenseReturnType result;
        for (int i = 0; i < derived().size(); ++i)
        {
            result.at(i) = derived().at(i) < other.derived().at(i);
        }
        return result;
    }

    HD Scalar all() const
    {
        Scalar result = 0;
        for (int i = 0; i < derived().size(); ++i)
        {
            result &= derived().at(i);
        }
        return result;
    }


    HD DenseReturnType operator>=(SameMatrix other) const
    {
        DenseReturnType result;
        for (int i = 0; i < derived().size(); ++i)
        {
            result.at(i) = derived().at(i) >= other.derived().at(i);
        }
        return result;
    }


    HD SameMatrix& operator+=(Scalar value)
    {
        derived() = derived() + value;
        return derived();
    }

    HD SameMatrix& operator+=(const SameMatrix& other)
    {
        derived() = derived() + other;
        return derived();
    }

    HD SameMatrix& operator-=(const SameMatrix& other)
    {
        derived() = derived() - other;
        return derived();
    }


    HD SameMatrix& operator*=(Scalar value)
    {
        derived() = derived() * value;
        return derived();
    }

    HD SameMatrix& operator/=(Scalar other)
    {
        derived() = derived() / other;
        return derived();
    }
};

template <typename _Scalar, int _Rows, int _Cols, int _Options = ColMajor>
class Array : public MatrixBase<Array<_Scalar, _Rows, _Cols, _Options>>
{
   public:
    static constexpr int Rows = _Rows;
    static constexpr int Cols = _Cols;
    using Scalar              = _Scalar;
    using SameMatrix          = Array<_Scalar, _Rows, _Cols, _Options>;
    static constexpr int Size = _Rows * _Cols;
    static_assert(_Rows > 0, "Rows must be positive");
    static_assert((_Options & RowMajor) == 0, "Only colmajor supportet");

    HD Array() {}

    template <typename OtherType>
    HD Array(const MatrixBase<OtherType>& other)
    {
        for (int i = 0; i < rows(); ++i)
        {
            for (int j = 0; j < cols(); ++j)
            {
                (*this)(i, j) = other(i, j);
            }
        }
    }

    HD _Scalar* data() { return data; }
    HD const _Scalar* data() const { return data; }

    HD _Scalar& operator()(int i, int j) { return _data[j * _Rows + i]; }
    HD const _Scalar& operator()(int i, int j) const { return _data[j * _Rows + i]; }

    template <typename T>
    HD Array<T, _Rows, _Cols, _Options> cast() const
    {
        Array<T, _Rows, _Cols, _Options> result;
        for (int i = 0; i < result.size(); ++i)
        {
            result.at(i) = (T)at(i);
        }

        return result;
    }


    HD SameMatrix min(SameMatrix other) const
    {
        SameMatrix result;
        for (int i = 0; i < size(); ++i)
        {
            result.at(i) = std::min(at(i), other.at(i));
        }
        return result;
    }

    HD SameMatrix max(SameMatrix other) const
    {
        SameMatrix result;
        for (int i = 0; i < size(); ++i)
        {
            result.at(i) = std::max(at(i), other.at(i));
        }
        return result;
    }

    HD SameMatrix abs() const
    {
        SameMatrix result;
        for (int i = 0; i < size(); ++i)
        {
            result.at(i) = std::abs(at(i));
        }
        return result;
    }


    HD SameMatrix floor() const
    {
        SameMatrix result;
        for (int i = 0; i < size(); ++i)
        {
            result.at(i) = std::floor(at(i));
        }
        return result;
    }

    HD SameMatrix ceil() const
    {
        SameMatrix result;
        for (int i = 0; i < size(); ++i)
        {
            result.at(i) = std::ceil(at(i));
        }
        return result;
    }

    HD int rows() const { return _Rows; }
    HD int cols() const { return _Cols; }
    HD int size() const { return Size; }

    HD _Scalar& at(int index) { return _data[index]; }
    HD const _Scalar& at(int index) const { return _data[index]; }



   private:
    _Scalar _data[Size];
};

template <typename _Scalar, int _Rows, int _Cols, int _Options = ColMajor>
class MatrixView : public MatrixBase<MatrixView<_Scalar, _Rows, _Cols, _Options>>
{
   public:
    static constexpr int Rows = _Rows;
    static constexpr int Cols = _Cols;
    using Scalar              = _Scalar;
    using SameMatrix          = MatrixView<_Scalar, _Rows, _Cols, _Options>;
    static constexpr int Size = _Rows * _Cols;
    static_assert(_Rows > 0, "Rows must be positive");
    static_assert((_Options & RowMajor) == 0, "Only colmajor supportet");

    HD MatrixView(Scalar* data, int row_stride, int col_stride)
        : _data(data), _row_stride(row_stride), _col_stride(col_stride)
    {
    }


    // assignment of two views should copy the data from to the other
    HD SameMatrix& operator=(const SameMatrix& other)
    {
        for (int i = 0; i < rows(); ++i)
        {
            for (int j = 0; j < cols(); ++j)
            {
                (*this)(i, j) = other(i, j);
            }
        }
        return *this;
    }

    template <typename OtherType>
    HD SameMatrix& operator=(const MatrixBase<OtherType>& other)
    {
        static_assert(Rows == OtherType::Rows && Cols == OtherType::Cols,
                      "Assignment is only allowed with the same dimensions.");
        for (int i = 0; i < rows(); ++i)
        {
            for (int j = 0; j < cols(); ++j)
            {
                (*this)(i, j) = other(i, j);
            }
        }
        return *this;
    }


    HD _Scalar& operator()(int i, int j) { return _data[i * _row_stride + j * _col_stride]; }
    HD const _Scalar& operator()(int i, int j) const { return _data[i * _row_stride + j * _col_stride]; }


    HD Scalar& at(int i) { return _data[i * _row_stride]; }
    HD const Scalar& at(int i) const { return _data[i * _row_stride]; }


    HD int rows() const { return _Rows; }
    HD int cols() const { return _Cols; }
    HD int size() const { return Size; }

   private:
    _Scalar* _data = nullptr;
    // distance between neighboring rows/cols
    int _row_stride, _col_stride;
};

template <typename _Scalar, int _Rows, int _Cols, int _Options = ColMajor>
class Matrix : public MatrixBase<Matrix<_Scalar, _Rows, _Cols, _Options>>
{
   public:
    static constexpr int Rows = _Rows;
    static constexpr int Cols = _Cols;
    using Scalar              = _Scalar;
    using SameMatrix          = Matrix<_Scalar, _Rows, _Cols, _Options>;
    static constexpr int Size = _Rows * _Cols;
    static_assert(_Rows > 0, "Rows must be positive");
    static_assert((_Options & RowMajor) == 0, "Only colmajor supportet");

    HD static SameMatrix Zero()
    {
        SameMatrix result;
        for (int i = 0; i < result.size(); ++i)
        {
            result._data[i] = 0;
        }
        return result;
    }

    HD static SameMatrix Ones()
    {
        SameMatrix result;
        for (int i = 0; i < result.size(); ++i)
        {
            result._data[i] = 1;
        }
        return result;
    }

    HD static SameMatrix Identity()
    {
        static_assert(_Rows == _Cols, "Only valid for square matrices.");
        SameMatrix result = Zero();
        for (int i = 0; i < _Rows; ++i)
        {
            result(i, i) = 1;
        }
        return result;
    }

    HD Matrix() {}

    template <typename OtherType>
    HD Matrix(const MatrixBase<OtherType>& other)
    {
        static_assert(Rows == OtherType::Rows && Cols == OtherType::Cols,
                      "Assignment is only allowed with the same dimensions.");
        for (int i = 0; i < rows(); ++i)
        {
            for (int j = 0; j < cols(); ++j)
            {
                (*this)(i, j) = other(i, j);
            }
        }
    }

    HD Matrix(const Array<_Scalar, _Rows, _Cols, _Options> other) { this->array() = other; }

    // Vector constructors
    HD Matrix(_Scalar x0, _Scalar x1)
    {
        static_assert(_Rows == 2 && _Cols == 1, "Constructor only valid for vectors.");
        _data[0] = x0;
        _data[1] = x1;
    }
    HD Matrix(_Scalar x0, _Scalar x1, _Scalar x2)
    {
        static_assert(_Rows == 3 && _Cols == 1, "Constructor only valid for vectors.");
        _data[0] = x0;
        _data[1] = x1;
        _data[2] = x2;
    }
    HD Matrix(_Scalar x0, _Scalar x1, _Scalar x2, _Scalar x3)
    {
        static_assert(_Rows == 4 && _Cols == 1, "Constructor only valid for vectors.");
        _data[0] = x0;
        _data[1] = x1;
        _data[2] = x2;
        _data[3] = x3;
    }

    HD Matrix(_Scalar x0, _Scalar x1, _Scalar x2, _Scalar x3, _Scalar x4)
    {
        static_assert(_Rows == 5 && _Cols == 1, "Constructor only valid for vectors.");
        _data[0] = x0;
        _data[1] = x1;
        _data[2] = x2;
        _data[3] = x3;
        _data[4] = x4;
    }

    HD Matrix(_Scalar x0, _Scalar x1, _Scalar x2, _Scalar x3, _Scalar x4, _Scalar x5)
    {
        static_assert(_Rows == 6 && _Cols == 1, "Constructor only valid for vectors.");
        _data[0] = x0;
        _data[1] = x1;
        _data[2] = x2;
        _data[3] = x3;
        _data[4] = x4;
        _data[5] = x5;
    }
    HD Matrix(_Scalar x0, _Scalar x1, _Scalar x2, _Scalar x3, _Scalar x4, _Scalar x5, _Scalar x6)
    {
        static_assert(_Rows == 7 && _Cols == 1, "Constructor only valid for vectors.");
        _data[0] = x0;
        _data[1] = x1;
        _data[2] = x2;
        _data[3] = x3;
        _data[4] = x4;
        _data[5] = x5;
        _data[6] = x6;
    }
    HD Matrix(_Scalar x0, _Scalar x1, _Scalar x2, _Scalar x3, _Scalar x4, _Scalar x5, _Scalar x6, _Scalar x7)
    {
        static_assert(_Rows == 8 && _Cols == 1, "Constructor only valid for vectors.");
        _data[0] = x0;
        _data[1] = x1;
        _data[2] = x2;
        _data[3] = x3;
        _data[4] = x4;
        _data[5] = x5;
        _data[6] = x6;
        _data[7] = x7;
    }

    HD _Scalar* data() { return _data; }
    HD const _Scalar* data() const { return _data; }

    HD _Scalar& operator[](int i)
    {
        static_assert(_Rows == 1 || _Cols == 1, "Constructor only valid for vectors.");
        return _data[i];
    }
    HD const _Scalar& operator[](int i) const
    {
        static_assert(_Rows == 1 || _Cols == 1, "Constructor only valid for vectors.");
        return _data[i];
    }

    HD _Scalar& operator()(int i)
    {
        static_assert(_Rows == 1 || _Cols == 1, "Constructor only valid for vectors.");
        return _data[i];
    }
    HD const _Scalar& operator()(int i) const
    {
        static_assert(_Rows == 1 || _Cols == 1, "Constructor only valid for vectors.");
        return _data[i];
    }

    HD _Scalar& at(int i)
    {
        static_assert(_Rows == 1 || _Cols == 1, "Constructor only valid for vectors.");
        return _data[i];
    }
    HD const _Scalar& at(int i) const
    {
        static_assert(_Rows == 1 || _Cols == 1, "Constructor only valid for vectors.");
        return _data[i];
    }

    HD _Scalar& operator()(int i, int j) { return _data[j * _Rows + i]; }
    HD const _Scalar& operator()(int i, int j) const { return _data[j * _Rows + i]; }

    HD Scalar& x() { return _data[0]; }
    HD Scalar& y() { return _data[1]; }
    HD Scalar& z() { return _data[2]; }
    HD Scalar& w() { return _data[3]; }
    HD const Scalar& x() const { return _data[0]; }
    HD const Scalar& y() const { return _data[1]; }
    HD const Scalar& z() const { return _data[2]; }
    HD const Scalar& w() const { return _data[3]; }

    constexpr int rows() const { return _Rows; }
    constexpr int cols() const { return _Cols; }
    constexpr int size() const { return Size; }

    template <typename T>
    HD Matrix<T, _Rows, _Cols, _Options> cast() const
    {
        Matrix<T, _Rows, _Cols, _Options> result;
        for (int i = 0; i < rows(); ++i)
        {
            for (int j = 0; j < cols(); ++j)
            {
                result(i, j) = (T)(*this)(i, j);
            }
        }

        return result;
    }

    HD Matrix<Scalar, _Cols, _Rows, _Options> transpose() const
    {
        Matrix<Scalar, _Cols, _Rows, _Options> result;
        for (int i = 0; i < rows(); ++i)
        {
            for (int j = 0; j < cols(); ++j)
            {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    HD MatrixView<Scalar, 1, _Cols, _Options> row(int id)
    {
        return MatrixView<Scalar, 1, _Cols, _Options>(&((*this)(id, 0)), 1, _Rows);
    }

    HD Matrix<Scalar, 1, _Cols, _Options> row(int id) const
    {
        return MatrixView<const Scalar, 1, _Cols, _Options>(&((*this)(id, 0)), 1, _Rows);
    }


    HD MatrixView<Scalar, _Rows, 1, _Options> col(int id)
    {
        return MatrixView<Scalar, _Rows, 1, _Options>(&((*this)(0, id)), 1, 1);
    }

    HD Matrix<Scalar, _Rows, 1, _Options> col(int id) const
    {
        return MatrixView<const Scalar, _Rows, 1, _Options>(&((*this)(0, id)), 1, 1);
    }

    template <int NewRows, int NewCols>
    HD MatrixView<Scalar, NewRows, NewCols, _Options> block(int i, int j)
    {
        return MatrixView<Scalar, NewRows, NewCols, _Options>(&((*this)(i, j)), 1, _Rows);
    }

    template <int NewRows, int NewCols>
    HD Matrix<Scalar, NewRows, NewCols, _Options> block(int i, int j) const
    {
        return MatrixView<const Scalar, NewRows, NewCols, _Options>(&((*this)(i, j)), 1, _Rows);
    }


    template <int NewRows>
    HD MatrixView<Scalar, NewRows, 1, _Options> head()
    {
        return MatrixView<Scalar, NewRows, 1, _Options>(&((*this)(0, 0)), 1, 1);
    }

    template <int NewRows>
    HD Matrix<Scalar, NewRows, 1, _Options> head() const
    {
        return MatrixView<const Scalar, NewRows, 1, _Options>(&((*this)(0, 0)), 1, 1);
    }

    template <int NewRows>
    HD MatrixView<Scalar, NewRows, 1, _Options> tail()
    {
        return MatrixView<Scalar, NewRows, 1, _Options>(&((*this)(rows() - NewRows, 0)), 1, 1);
    }

    template <int NewRows>
    HD Matrix<Scalar, NewRows, 1, _Options> tail() const
    {
        return MatrixView<Scalar, NewRows, 1, _Options>(&((*this)(rows() - NewRows, 0)), 1, 1);
    }

    HD const Array<_Scalar, _Rows, _Cols, _Options>& array() const
    {
        return *reinterpret_cast<const Array<_Scalar, _Rows, _Cols, _Options>*>(this);
    }

    HD Array<_Scalar, _Rows, _Cols, _Options>& array()
    {
        return *reinterpret_cast<Array<_Scalar, _Rows, _Cols, _Options>*>(this);
    }


   private:
    _Scalar _data[Size];
};


template <typename Derived1, typename Derived2>
HD typename Derived1::DenseReturnType operator-(const MatrixBase<Derived1>& m1, const MatrixBase<Derived2>& m2)
{
    typename Derived1::DenseReturnType result;
    for (int i = 0; i < result.rows(); ++i)
    {
        for (int j = 0; j < result.cols(); ++j)
        {
            result(i, j) = m1(i, j) - m2(i, j);
        }
    }
    return result;
}

template <typename Derived1, typename Derived2>
HD typename Derived1::DenseReturnType operator+(const MatrixBase<Derived1>& m1, const MatrixBase<Derived2>& m2)
{
    typename Derived1::DenseReturnType result;
    for (int i = 0; i < result.rows(); ++i)
    {
        for (int j = 0; j < result.cols(); ++j)
        {
            result(i, j) = m1(i, j) + m2(i, j);
        }
    }
    return result;
}


template <typename Derived>
HD typename Derived::DenseReturnType operator+(const MatrixBase<Derived>& m1, typename Derived::Scalar v)
{
    return v + m1;
}

template <typename Derived>
HD typename Derived::DenseReturnType operator+(typename Derived::Scalar v, const MatrixBase<Derived>& m1)
{
    typename Derived::DenseReturnType result;
    for (int i = 0; i < result.rows(); ++i)
    {
        for (int j = 0; j < result.cols(); ++j)
        {
            result(i, j) = v + m1(i, j);
        }
    }
    return result;
}


template <typename Derived>
HD typename Derived::DenseReturnType operator*(const MatrixBase<Derived>& m1, typename Derived::Scalar v)
{
    return v * m1;
}

template <typename Derived>
HD typename Derived::DenseReturnType operator*(typename Derived::Scalar v, const MatrixBase<Derived>& m1)
{
    typename Derived::DenseReturnType result;
    for (int i = 0; i < result.rows(); ++i)
    {
        for (int j = 0; j < result.cols(); ++j)
        {
            result(i, j) = v * m1(i, j);
        }
    }
    return result;
}

template <typename Derived>
HD typename Derived::DenseReturnType operator/(const MatrixBase<Derived>& m1, typename Derived::Scalar v)
{
    typename Derived::DenseReturnType result;
    for (int i = 0; i < result.rows(); ++i)
    {
        for (int j = 0; j < result.cols(); ++j)
        {
            result(i, j) = m1(i, j) / v;
        }
    }
    return result;
}

template <typename Derived>
HD typename Derived::PlainObject operator/(typename Derived::Scalar v, const MatrixBase<Derived>& m1)
{
    typename Derived::PlainObject result;
    for (int i = 0; i < result.rows(); ++i)
    {
        for (int j = 0; j < result.cols(); ++j)
        {
            result(i, j) = v / m1(i, j);
        }
    }
    return result;
}

template <typename _Scalar, int _Rows0, int _Cols0, int _Options0, int _Rows1, int _Cols1, int _Options1>
HD Matrix<_Scalar, _Rows0, _Cols1, _Options0> operator*(const Matrix<_Scalar, _Rows0, _Cols0, _Options0>& m1,
                                                        Matrix<_Scalar, _Rows1, _Cols1, _Options1> m2)
{
    static_assert(_Rows1 == _Cols0, "Invalid matrix shapes");
    Matrix<_Scalar, _Rows0, _Cols1, _Options0> result;
    result.setZero();

    for (int i = 0; i < result.rows(); ++i)
    {
        for (int j = 0; j < result.cols(); ++j)
        {
            for (int k = 0; k < _Rows1; ++k)
            {
                result(i, j) += m1(i, k) * m2(k, j);
            }
        }
    }
    return result;
}

template <typename _Scalar, int _Rows0, int _Cols0, int _Options0>
HD Array<_Scalar, _Rows0, _Cols0, _Options0> operator*(const Array<_Scalar, _Rows0, _Cols0, _Options0>& m1,
                                                       Array<_Scalar, _Rows0, _Cols0, _Options0> m2)
{
    Array<_Scalar, _Rows0, _Cols0, _Options0> result;
    for (int i = 0; i < m1.rows(); ++i)
    {
        for (int j = 0; j < m1.cols(); ++j)
        {
            result(i, j) = m1(i, j) * m2(i, j);
        }
    }
    return result;
}

template <typename _Scalar, int _Rows0, int _Cols0, int _Options0>
HD Array<_Scalar, _Rows0, _Cols0, _Options0> operator/(const Array<_Scalar, _Rows0, _Cols0, _Options0>& m1,
                                                       Array<_Scalar, _Rows0, _Cols0, _Options0> m2)
{
    Array<_Scalar, _Rows0, _Cols0, _Options0> result;
    for (int i = 0; i < m1.rows(); ++i)
    {
        for (int j = 0; j < m1.cols(); ++j)
        {
            result(i, j) = m1(i, j) / m2(i, j);
        }
    }
    return result;
}


template <typename Derived>
std::ostream& operator<<(std::ostream& strm, const MatrixBase<Derived>& m1)
{
    for (int i = 0; i < m1.rows(); ++i)
    {
        for (int j = 0; j < m1.cols(); ++j)
        {
            strm << m1(i, j) << " ";
        }
        if (i < m1.rows() - 1)
        {
            strm << "\n";
        }
    }
    return strm;
}


template <typename T, int rows>
using Vector = Matrix<T, rows, 1>;

}  // namespace Eigen