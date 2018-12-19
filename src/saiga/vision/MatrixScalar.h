/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/assert.h"
#include "saiga/vision/VisionIncludes.h"
namespace Saiga
{
template <typename MatrixType>
struct MatrixScalar
{
    using Scalar = typename MatrixType::Scalar;
    MatrixType data;

    MatrixScalar() = default;
    MatrixScalar(Scalar v)
    {
        SAIGA_ASSERT(v == 0);
        data.setZero();
    }

    MatrixScalar(const MatrixType& v) : data(v) {}
    MatrixScalar& operator=(const MatrixType& v)
    {
        data = v;
        return *this;
    }

    explicit operator MatrixType() const { return data; }

    MatrixScalar operator+(const MatrixScalar& other) const { return {data + other.data}; }

    void operator+=(const MatrixScalar& other) { data += other.data; }

    template <typename T>
    void operator+=(const T& other)
    {
        data += other;
    }

    MatrixScalar operator*(const MatrixScalar& other) const { return {data * other.data}; }

    MatrixType& get() { return data; }
    const MatrixType& get() const { return data; }
};

/**
 * Convert a block vector (a vector of vectors) to a 1-dimensional vector.
 * The inner vector must be of constant size.
 */
template <typename MatrixType>
auto blockVectorToVector(const Eigen::Matrix<MatrixScalar<MatrixType>, -1, 1>& m)
{
    static_assert(MatrixType::RowsAtCompileTime > 0, "The inner size must be fixed.");
    Eigen::Matrix<typename MatrixType::Scalar, -1, 1> dense(m.rows() * MatrixType::RowsAtCompileTime);
    for (int i = 0; i < m.rows(); ++i)
    {
        dense.segment(i * MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime) = m(i).get();
    }
    return dense;
}


/**
 * Convert a block vector (a vector of vectors) to a 1-dimensional vector.
 * The inner vector must be of constant size.
 */
template <typename MatrixType>
auto blockDiagonalToMatrix(const Eigen::DiagonalMatrix<MatrixScalar<MatrixType>, -1>& m)
{
    static_assert(MatrixType::RowsAtCompileTime > 0, "The inner size must be fixed.");
    Eigen::Matrix<typename MatrixType::Scalar, -1, -1> dense(m.rows() * MatrixType::RowsAtCompileTime,
                                                             m.cols() * MatrixType::ColsAtCompileTime);
    dense.setZero();
    for (int i = 0; i < m.rows(); ++i)
    {
        dense.block(i * MatrixType::RowsAtCompileTime, i * MatrixType::ColsAtCompileTime, MatrixType::RowsAtCompileTime,
                    MatrixType::ColsAtCompileTime) = m.diagonal()(i).get();
    }
    return dense;
}


/**
 * Convert a block vector (a vector of vectors) to a 1-dimensional vector.
 * The inner vector must be of constant size.
 */
template <typename MatrixType>
auto blockMatrixToMatrix(const Eigen::Matrix<MatrixScalar<MatrixType>, -1, -1>& m)
{
    static_assert(MatrixType::RowsAtCompileTime > 0 && MatrixType::ColsAtCompileTime > 0,
                  "The inner size must be fixed.");

    Eigen::Matrix<typename MatrixType::Scalar, -1, -1> dense(m.rows() * MatrixType::RowsAtCompileTime,
                                                             m.cols() * MatrixType::ColsAtCompileTime);
    for (int i = 0; i < m.rows(); ++i)
    {
        for (int j = 0; j < m.cols(); ++j)
        {
            dense.block(i * MatrixType::RowsAtCompileTime, j * MatrixType::ColsAtCompileTime,
                        MatrixType::RowsAtCompileTime, MatrixType::ColsAtCompileTime) = m(i, j).get();
        }
    }
    return dense;
}

}  // namespace Saiga
