/**
 * This file is part of the Eigen Recursive Matrix Extension (ERME).
 *
 * Copyright (c) 2019 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "Eigen/Core"
#include "Eigen/Sparse"

namespace Eigen::Recursive
{
template <typename MatrixType>
struct MatrixScalar
{
    using M          = MatrixType;
    using Scalar     = typename MatrixType::Scalar;
    using ScalarType = MatrixScalar<MatrixType>;

    using TransposeBase = typename MatrixType::TransposeReturnType::NestedExpression;

    MatrixType data;


    EIGEN_ALWAYS_INLINE MatrixScalar() = default;

    // This constructor may seem a bit strange, but assigning anything different from a zero to a matrix is not well
    // defined. We also can't remove this constructor because some eigen functions use Scalar(0). Therefore we need
    // a constructor from a single (actual) scalar value.
    EIGEN_ALWAYS_INLINE MatrixScalar(Scalar v) { data.setZero(); }
    //    EIGEN_ALWAYS_INLINE MatrixScalar(int v) { data.setZero(); }

    EIGEN_ALWAYS_INLINE MatrixScalar(const MatrixType& v) : data(v) {}


    EIGEN_ALWAYS_INLINE MatrixScalar& operator=(const MatrixType& v)
    {
        data = v;
        return *this;
    }

    EIGEN_ALWAYS_INLINE Scalar& operator()(int i, int j) { return data(i, j); }
    EIGEN_ALWAYS_INLINE const Scalar& operator()(int i, int j) const { return data(i, j); }



    EIGEN_ALWAYS_INLINE MatrixScalar<TransposeBase> transpose() const { return {data.transpose()}; }


    EIGEN_ALWAYS_INLINE ScalarType operator-() const { return {-data}; }
    EIGEN_ALWAYS_INLINE ScalarType operator+(const ScalarType& other) const { return {data + other.data}; }
    EIGEN_ALWAYS_INLINE ScalarType operator-(const ScalarType& other) const { return {data - other.data}; }
    EIGEN_ALWAYS_INLINE ScalarType operator*(const ScalarType& other) const { return {data * other.data}; }
    //    ScalarType operator-(const ScalarType& other) const
    //    {
    //        cout << "-2" << endl;
    //        return {data - other.data};
    //    }

    template <typename G>
    EIGEN_ALWAYS_INLINE void operator+=(const MatrixScalar<G>& other)
    {
        data += other.data;
    }

    template <typename G>
    EIGEN_ALWAYS_INLINE void operator-=(const MatrixScalar<G>& other)
    {
        data -= other.data;
    }

    EIGEN_ALWAYS_INLINE void operator-=(const ScalarType& other) { data -= other.data; }

    // scalar product
    EIGEN_ALWAYS_INLINE ScalarType operator*(const Scalar& other) const { return {data * other}; }
    EIGEN_ALWAYS_INLINE void operator*=(const Scalar& other) { data *= other; }


    // general matrix product
    template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    EIGEN_ALWAYS_INLINE auto operator*(
        const MatrixScalar<Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>>& other) const
    {
        using ReturnType = Eigen::Matrix<_Scalar, MatrixType::RowsAtCompileTime, _Cols, _Options>;
        return MatrixScalar<ReturnType>(data * other.data);
    }


    //    template <typename T>
    //    auto operator*(const T& other) const
    //    {
    //        auto res = (data * other.data).evaluate();
    //        return MatrixScalar<decltype(res)>(res);
    //    }

    EIGEN_ALWAYS_INLINE MatrixType& get() { return data; }
    EIGEN_ALWAYS_INLINE const MatrixType& get() const { return data; }
};

template <typename T>
EIGEN_ALWAYS_INLINE auto makeMatrixScalar(const T& v)
{
    return MatrixScalar<T>(v);
}

template <typename T>
struct RemoveMatrixScalarImpl
{
    EIGEN_ALWAYS_INLINE static T& get(T& A) { return A; }
    EIGEN_ALWAYS_INLINE static const T& get(const T& A) { return A; }
};

template <typename G>
struct RemoveMatrixScalarImpl<MatrixScalar<G>>
{
    EIGEN_ALWAYS_INLINE static G& get(MatrixScalar<G>& A) { return A.get(); }
    EIGEN_ALWAYS_INLINE static const G& get(const MatrixScalar<G>& A) { return A.get(); }
};

template <typename T>
EIGEN_ALWAYS_INLINE auto& removeMatrixScalar(T& A)
{
    return RemoveMatrixScalarImpl<T>::get(A);
}

template <typename T>
EIGEN_ALWAYS_INLINE auto& removeMatrixScalar(const T& A)
{
    return RemoveMatrixScalarImpl<T>::get(A);
}

}  // namespace Eigen::Recursive


namespace Eigen
{
template <typename T, typename BinaryOp>
struct ScalarBinaryOpTraits<Recursive::MatrixScalar<T>, Recursive::MatrixScalar<T>, BinaryOp>
{
    typedef Recursive::MatrixScalar<T> ReturnType;
};


template <typename LHS, typename RHS, typename BinaryOp>
struct ScalarBinaryOpTraits<Recursive::MatrixScalar<LHS>, Recursive::MatrixScalar<RHS>, BinaryOp>
{
    enum
    {
        n = LHS::RowsAtCompileTime,
        m = RHS::ColsAtCompileTime,
        // Use Storage options of RHS type (might not be the best)
        options = RHS::Options
    };

    using ScalarType = typename ScalarBinaryOpTraits<typename LHS::Scalar, typename RHS::Scalar, BinaryOp>::ReturnType;
    using MatrixType = Matrix<ScalarType, n, m, options>;

    typedef Recursive::MatrixScalar<MatrixType> ReturnType;
};


}  // namespace Eigen
