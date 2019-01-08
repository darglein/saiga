/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/assert.h"
#include "saiga/vision/MatrixScalar.h"

namespace Saiga
{
struct MatrixDimensions
{
    int rows;
    int cols;

    // element wise multiplication
    MatrixDimensions operator*(const MatrixDimensions& other) { return {rows * other.rows, cols * other.cols}; }
};


template <typename T>
struct EvaluateMatrixDimensions
{
};

// An actual scalar has dimensions [1,1]
template <>
struct EvaluateMatrixDimensions<double>
{
    static MatrixDimensions get() { return {1, 1}; }
};



template <int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct EvaluateMatrixDimensions<Eigen::Matrix<double, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>
{
    using ChildExpansion = EvaluateMatrixDimensions<double>;
    using MatrixType     = Eigen::Matrix<double, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;

    static MatrixDimensions get(const MatrixType& m)
    {
        return MatrixDimensions{(int)m.rows(), (int)m.cols()} * ChildExpansion::get();
    }
};


template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct EvaluateMatrixDimensions<Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>
{
    using ChildExpansion = EvaluateMatrixDimensions<_Scalar>;
    using MatrixType     = Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;

    static MatrixDimensions get(const MatrixType& m)
    {
        return MatrixDimensions{(int)m.rows(), (int)m.cols()} * ChildExpansion::get(m(0, 0));
    }
};

template <typename G>
struct EvaluateMatrixDimensions<MatrixScalar<G>>
{
    static MatrixDimensions get(const MatrixScalar<G>& m) { return EvaluateMatrixDimensions<G>::get(m.get()); }
};


// ====================================================================================================
template <typename T>
using ExpansionType = Eigen::Matrix<T, -1, -1>;



template <typename T>
struct ExpandImpl
{
    //    static_assert(false, "No viable spezialization found!");
};

template <int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct ExpandImpl<Eigen::Matrix<double, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>
{
    using Scalar     = double;
    using BaseScalar = double;
    using MatrixType = Eigen::Matrix<Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;
    static ExpansionType<Scalar> get(const MatrixType& m) { return ExpansionType<Scalar>(m); }
};

template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct ExpandImpl<Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>
{
    using Scalar = _Scalar;

    using ChildExpansion = ExpandImpl<_Scalar>;
    using BaseScalar     = typename ChildExpansion::BaseScalar;



    using MatrixType = Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;
    static ExpansionType<BaseScalar> get(const MatrixType& A)
    {
        MatrixDimensions dim = EvaluateMatrixDimensions<MatrixType>::get(A);
        auto n               = dim.rows / A.rows();
        auto m               = dim.cols / A.cols();

        ExpansionType<BaseScalar> result;
        result.resize(A.rows() * n, A.cols() * m);

        for (int i = 0; i < A.rows(); ++i)
        {
            for (int j = 0; j < A.cols(); ++j)
            {
                auto b = ChildExpansion::get(A(i, j));
                SAIGA_ASSERT(b.rows() == n && b.cols() == m);  // all blocks must have the same dimension
                result.block(i * n, j * m, n, m) = b;
            }
        }


        return result;
    }
};



template <typename G>
struct ExpandImpl<MatrixScalar<G>>
{
    using ChildExpansion = ExpandImpl<G>;
    using BaseScalar     = typename ChildExpansion::BaseScalar;

    static auto get(const MatrixScalar<G>& m) { return ExpandImpl<G>::get(m.get()); }
};


template <typename T>
auto expand(const T& v)
{
    return ExpandImpl<T>::get(v);
}

}  // namespace Saiga
