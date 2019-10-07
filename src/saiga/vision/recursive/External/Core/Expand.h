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

template <>
struct EvaluateMatrixDimensions<float>
{
    static MatrixDimensions get() { return {1, 1}; }
};

template <typename T>
struct EvaluateMatrixDimensions<MatrixBase<T>>
{
    using MatrixType     = MatrixBase<T>;
    using Scalar         = typename MatrixType::Scalar;
    using ChildExpansion = EvaluateMatrixDimensions<Scalar>;

    static MatrixDimensions get(const MatrixType& m)
    {
        return MatrixDimensions{(int)m.rows(), (int)m.cols()} * ChildExpansion::get(m(0, 0));
    }
};

template <int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct EvaluateMatrixDimensions<Matrix<double, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>
{
    using ChildExpansion = EvaluateMatrixDimensions<double>;
    using MatrixType     = Matrix<double, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;

    static MatrixDimensions get(const MatrixType& m)
    {
        return MatrixDimensions{(int)m.rows(), (int)m.cols()} * ChildExpansion::get();
    }
};

template <int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct EvaluateMatrixDimensions<Matrix<float, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>
{
    using ChildExpansion = EvaluateMatrixDimensions<float>;
    using MatrixType     = Matrix<float, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;

    static MatrixDimensions get(const MatrixType& m)
    {
        return MatrixDimensions{(int)m.rows(), (int)m.cols()} * ChildExpansion::get();
    }
};


template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct EvaluateMatrixDimensions<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>
{
    using ChildExpansion = EvaluateMatrixDimensions<_Scalar>;
    using MatrixType     = Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;

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
using ExpansionType = Matrix<T, -1, -1>;



template <typename T>
struct ExpandImpl
{
    //    static_assert(false, "No viable spezialization found!");
};

template <>
struct ExpandImpl<double>
{
    using Scalar = double;
    static ExpansionType<Scalar> get(const Scalar& m)
    {
        ExpansionType<Scalar> A(1, 1);
        A(0, 0) = m;
        return A;
    }
};

template <>
struct ExpandImpl<float>
{
    using Scalar = float;
    static ExpansionType<Scalar> get(const Scalar& m)
    {
        ExpansionType<Scalar> A(1, 1);
        A(0, 0) = m;
        return A;
    }
};

template <int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct ExpandImpl<Matrix<double, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>
{
    using Scalar     = double;
    using BaseScalar = double;
    using MatrixType = Matrix<Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;
    static ExpansionType<Scalar> get(const MatrixType& m) { return ExpansionType<Scalar>(m); }
};

template <int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct ExpandImpl<Matrix<float, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>
{
    using Scalar     = float;
    using BaseScalar = float;
    using MatrixType = Matrix<Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;
    static ExpansionType<Scalar> get(const MatrixType& m) { return ExpansionType<Scalar>(m); }
};

template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct ExpandImpl<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>
{
    using Scalar = _Scalar;

    using ChildExpansion = ExpandImpl<_Scalar>;
    using BaseScalar     = typename ChildExpansion::BaseScalar;



    using MatrixType = Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;
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
                eigen_assert(b.rows() == n && b.cols() == m);  // all blocks must have the same dimension
                result.block(i * n, j * m, n, m) = b;
            }
        }


        return result;
    }
};


template <typename T>
struct ExpandImpl<MatrixBase<T>>
{
    using MatrixType = MatrixBase<T>;

    using Scalar = typename MatrixType::Scalar;

    using ChildExpansion = ExpandImpl<Scalar>;
    using BaseScalar     = typename ChildExpansion::BaseScalar;



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
                eigen_assert(b.rows() == n && b.cols() == m);  // all blocks must have the same dimension
                result.block(i * n, j * m, n, m) = b;
            }
        }


        return result;
    }
};



template <typename _Scalar, int _Options>
struct ExpandImpl<SparseMatrix<_Scalar, _Options>>
{
    using ChildExpansion = ExpandImpl<ExpansionType<_Scalar>>;
    using BaseScalar     = typename ChildExpansion::BaseScalar;

    static auto get(const SparseMatrix<_Scalar, _Options>& m) { return ChildExpansion::get(m.toDense()); }
};


template <typename _Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
struct ExpandImpl<DiagonalMatrix<_Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
{
    using ChildExpansion = ExpandImpl<ExpansionType<_Scalar>>;
    using BaseScalar     = typename ChildExpansion::BaseScalar;

    static auto get(const DiagonalMatrix<_Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>& m)
    {
        return ChildExpansion::get(m.toDenseMatrix());
    }
};



template <typename G>
struct ExpandImpl<MatrixScalar<G>>
{
    using ChildExpansion = ExpandImpl<G>;
    using BaseScalar     = typename ChildExpansion::BaseScalar;

    static auto get(const MatrixScalar<G>& m) { return ChildExpansion::get(m.get()); }
};


template <typename T>
auto expand(const T& v)
{
    return ExpandImpl<T>::get(v);
}


// =======================================================================================================


template <typename T>
struct ScalarType
{
    using Type = typename ScalarType<typename T::Scalar>::Type;
};

template <>
struct ScalarType<double>
{
    using Type = double;
};

template <>
struct ScalarType<float>
{
    using Type = float;
};


template <typename G>
struct ScalarType<MatrixScalar<G>>
{
    using Type = typename ScalarType<G>::Type;
};


}  // namespace Eigen::Recursive


namespace Eigen
{
template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
std::ostream& operator<<(
    std::ostream& strm, const Matrix<Recursive::MatrixScalar<_Scalar>, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& rhs)
{
    strm << expand(rhs) << std::endl;
    return strm;
}
}  // namespace Eigen
