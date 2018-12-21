/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "Eigen/Sparse"
#include "saiga/util/assert.h"
#include "saiga/vision/VisionIncludes.h"
#include "MatrixScalar.h"
namespace Saiga
{

template <typename T, int options>
std::vector<Eigen::Triplet<T>> to_triplets(const Eigen::SparseMatrix<T, options>& M)
{
    std::vector<Eigen::Triplet<T>> v;
    for (int i = 0; i < M.outerSize(); i++)
        for (typename Eigen::SparseMatrix<T, options>::InnerIterator it(M, i); it; ++it)
            v.emplace_back(it.row(), it.col(), it.value());
    return v;
}

template <typename MatrixType, int options>
std::vector<Eigen::Triplet<typename MatrixType::Scalar>> sparseBlockToTriplets(
    const Eigen::SparseMatrix<MatrixScalar<MatrixType>, options>& M)
{
    std::vector<Eigen::Triplet<typename MatrixType::Scalar>> v;
    v.reserve(M.nonZeros() * MatrixType::ColsAtCompileTime * MatrixType::RowsAtCompileTime);

    for (int i = 0; i < M.outerSize(); i++)
    {
        for (typename Eigen::SparseMatrix<MatrixScalar<MatrixType>, options>::InnerIterator it(M, i); it; ++it)
        {
            int x = it.col() * MatrixType::ColsAtCompileTime;
            int y = it.row() * MatrixType::RowsAtCompileTime;

            for (int r = 0; r < MatrixType::RowsAtCompileTime; ++r)
            {
                for (int c = 0; c < MatrixType::ColsAtCompileTime; ++c)
                {
                    v.emplace_back(r + y, c + x, it.value().get()(r, c));
                }
            }
        }
    }
    return v;
}


}  // namespace Saiga
