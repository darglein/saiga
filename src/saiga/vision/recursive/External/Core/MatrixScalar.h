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
#define OLD_MS
template <typename T, bool = internal::is_arithmetic<T>::value>
struct BaseScalar
{
    using type = typename BaseScalar<typename T::Scalar>::type;
};
template <typename T>
struct BaseScalar<T, true>
{
    using type = T;
};


#ifndef OLD_MS


template <typename MatrixType>
struct MatrixScalar : public MatrixType
{
   public:
    using M           = MatrixType;
    using PlainObject = MatrixScalar<MatrixType>;

    MatrixScalar() = default;
    MatrixScalar(const MatrixType& a) : MatrixType(a) {}

    /**
     * This ctor is a bit questionable.
     *
     * It only exists because Eigen often uses the assignment
     * Scalar sum = Scalar(0);
     *
     * So the better solution would be to change this line into something more expessive for example:
     * Scalar sum = neutral_element_plus<Scalar>();
     *
     * But, since we don't want to change the complete source code we leave this here for now.
     */
    explicit MatrixScalar(int) { this->setZero(); }
    //    void operator=(int) { this->setZero(); }

    template <typename T, typename G>
    MatrixScalar(const Product<T, G>& a) : MatrixType(a)
    {
    }

    //    template <typename T>
    //    MatrixScalar(const Inverse<T>& a) : MatrixType(a)
    //    {
    //    }

    template <typename T, typename G, typename H>
    MatrixScalar(const CwiseBinaryOp<T, G, H>& a) : MatrixType(a)
    {
    }

    template <typename T, typename G>
    MatrixScalar(const CwiseUnaryOp<T, G>& a) : MatrixType(a)
    {
    }

    // This sadly doesn't work
    //    template <typename T>
    //    MatrixScalar(const MatrixBase<T>& a) : MatrixType(a)
    //    {
    //    }

    MatrixType& get() { return *this; }
    const MatrixType& get() const { return *this; }
};

#else
template <typename MatrixType>
struct MatrixScalar
{
    using M          = MatrixType;
    using Scalar     = typename MatrixType::Scalar;
    using ScalarType = MatrixScalar<MatrixType>;

    using BaseScalarType = typename BaseScalar<MatrixType>::type;
    using TransposeBase  = typename MatrixType::TransposeReturnType::NestedExpression;

    MatrixType data;


    EIGEN_ALWAYS_INLINE MatrixScalar() = default;

    // This constructor may seem a bit strange, but assigning anything different from a zero to a matrix is not well
    // defined. We also can't remove this constructor because some eigen functions use Scalar(0). Therefore we need
    // a constructor from a single (actual) scalar value.
    EIGEN_ALWAYS_INLINE MatrixScalar(BaseScalarType v)
    {
        data.setZero();
        //        data.setConstant(v);
    }
    //    EIGEN_ALWAYS_INLINE MatrixScalar(int v) { data.setZero(); }

    EIGEN_ALWAYS_INLINE MatrixScalar(const MatrixType& v) : data(v) {}

    //    template <typename T>
    //    EIGEN_ALWAYS_INLINE MatrixScalar(const MatrixBase<T>& v) : data(v)
    //    {
    //    }


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
    //        std::cout << "-2" << std::endl;
    //        return {data - other.data};
    //    }

    template <typename G>
    EIGEN_ALWAYS_INLINE void operator+=(const MatrixScalar<G>& other)
    {
        data += other.data;
    }

    template <typename G>
    EIGEN_ALWAYS_INLINE bool operator==(const MatrixScalar<G>& other)
    {
        return data == other.data;
    }

    template <typename G>
    EIGEN_ALWAYS_INLINE void operator-=(const MatrixScalar<G>& other)
    {
        data -= other.data;
    }

    EIGEN_ALWAYS_INLINE void operator-=(const ScalarType& other) { data -= other.data; }


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


// scalar product



#endif

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
// template <typename T, typename BinaryOp>
// struct ScalarBinaryOpTraits<double, Recursive::MatrixScalar<T>, BinaryOp>
//{
//    typedef Recursive::MatrixScalar<T> ReturnType;
//};

// template <typename T, typename BinaryOp>
// struct ScalarBinaryOpTraits<Recursive::MatrixScalar<T>, double, BinaryOp>
//{
//    typedef Recursive::MatrixScalar<T> ReturnType;
//};

#ifndef OLD_MS
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

template <typename LHS, typename T, typename BinaryOp>
struct ScalarBinaryOpTraits<Recursive::MatrixScalar<LHS>, MatrixBase<T>, BinaryOp>
{
    using ReturnType = int;
};


template <typename LHS, typename T, typename BinaryOp>
struct ScalarBinaryOpTraits<MatrixBase<T>, Recursive::MatrixScalar<LHS>, BinaryOp>
{
    using ReturnType = int;
};

#else
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


#endif
}  // namespace Eigen
namespace Eigen
{
template <typename _Scalar>
auto operator*(typename Recursive::BaseScalar<_Scalar>::type lhs, const Recursive::MatrixScalar<_Scalar>& rhs)
{
    using MatrixType  = Recursive::MatrixScalar<_Scalar>;
    MatrixType result = (lhs * rhs.get()).eval();
    return result;
}

template <typename _Scalar>
auto operator*(const Recursive::MatrixScalar<_Scalar>& lhs, typename Recursive::BaseScalar<_Scalar>::type rhs)
{
    using MatrixType  = Recursive::MatrixScalar<_Scalar>;
    MatrixType result = (lhs.get() * rhs).eval();
    return result;
}

/**
 * This the actual recursive scalar multiplication. B = A * x
 * with B and A being recursive types and x an actual scalar (double,float,...)
 *
 * Eigen's default scalar multiplication doens't work because Matrix::Scalar are still matrices
 * in recursive types.
 *
 * Unfortunately I didn't find a way to return a "scalar product" expression object, so this
 * might create an unnessecary temporary.
 */
template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
auto operator*(typename Recursive::BaseScalar<_Scalar>::type lhs,
               const Matrix<Recursive::MatrixScalar<_Scalar>, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& rhs)
{
    using MatrixType = Matrix<Recursive::MatrixScalar<_Scalar>, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;
    MatrixType result;
    result.resize(rhs.rows(), rhs.cols());
    for (int i = 0; i < rhs.rows(); ++i)
    {
        for (int j = 0; j < rhs.cols(); ++j)
        {
            result(i, j) = lhs * removeMatrixScalar(rhs(i, j));
        }
    }
    return result;
}
template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
auto operator*(const Eigen::Matrix<Recursive::MatrixScalar<_Scalar>, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& lhs,
               typename Recursive::BaseScalar<_Scalar>::type rhs)
{
    using MatrixType = Eigen::Matrix<Recursive::MatrixScalar<_Scalar>, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;
    MatrixType result;
    result.resize(lhs.rows(), lhs.cols());
    for (int i = 0; i < lhs.rows(); ++i)
    {
        for (int j = 0; j < lhs.cols(); ++j)
        {
            result(i, j) = lhs(i, j) * rhs;
        }
    }
    return result;
}


template <typename _Scalar, int _Options>
auto operator*(typename Recursive::BaseScalar<_Scalar>::type lhs,
               const SparseMatrix<Recursive::MatrixScalar<_Scalar>, _Options>& rhs)
{
    using MatrixType  = SparseMatrix<Recursive::MatrixScalar<_Scalar>, _Options>;
    MatrixType result = rhs;

    for (int i = 0; i < result.outerSize(); ++i)
    {
        for (typename MatrixType::InnerIterator it(result, i); it; ++it)
        {
            it.valueRef() = lhs * it.value();
        }
    }
    result.markAsRValue();
    return result;
}
template <typename _Scalar, int _Options>
auto operator*(const SparseMatrix<Recursive::MatrixScalar<_Scalar>, _Options>& lhs,
               typename Recursive::BaseScalar<_Scalar>::type rhs)
{
    using MatrixType  = SparseMatrix<Recursive::MatrixScalar<_Scalar>, _Options>;
    MatrixType result = lhs;

    for (int i = 0; i < result.outerSize(); ++i)
    {
        for (typename MatrixType::InnerIterator it(result, i); it; ++it)
        {
            it.valueRef() = it.value() * rhs;
        }
    }
    result.markAsRValue();
    return result;
}

}  // namespace Eigen
