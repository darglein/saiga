/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "Eigen/Sparse"
#include "MatrixScalar.h"

namespace Eigen::Recursive
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

template <typename T>
auto to_triplets(const T& M)
{
    std::vector<Eigen::Triplet<typename T::Scalar>> v;
    for (int i = 0; i < M.rows(); i++)
    {
        for (int j = 0; j < M.cols(); ++j)
        {
            v.emplace_back(i, j, M(i, j));
        }
    }
    return v;
}


template <typename T>
void addOffsetToTriplets(std::vector<Eigen::Triplet<T>>& v, int row, int col)
{
    for (auto& t : v)
    {
        t = Eigen::Triplet<T>(t.row() + row, t.col() + col, t.value());
    }
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


template <typename BlockType, typename T, int _options>
void sparseBlockToFlatMatrix(const Eigen::SparseMatrix<MatrixScalar<BlockType>, _options>& src,
                             Eigen::SparseMatrix<T, _options>& dst)
{
    using Lhs              = Eigen::SparseMatrix<MatrixScalar<BlockType>, _options>;
    using LhsInnerIterator = typename Lhs::InnerIterator;

    const int outerSizeBlock =
        (_options & Eigen::RowMajorBit) ? BlockType::RowsAtCompileTime : BlockType::ColsAtCompileTime;
    const int innerSizeBlock =
        (_options & Eigen::RowMajorBit) ? BlockType::ColsAtCompileTime : BlockType::RowsAtCompileTime;
    //    int outerSize = outerSizeBlock * src.outerSize();
    //    int innerSize = innerSizeBlock * src.innerSize();

    dst.resize(src.rows() * BlockType::RowsAtCompileTime, src.cols() * BlockType::ColsAtCompileTime);
    dst.reserve(src.nonZeros() * BlockType::RowsAtCompileTime * BlockType::ColsAtCompileTime);

    for (int i = 0; i < dst.outerSize() + 1; ++i)
    {
        dst.outerIndexPtr()[i] = 0;
    }

    for (int i = 0; i < src.outerSize(); ++i)
    {
        int numElementsBlock = src.outerIndexPtr()[i + 1] - src.outerIndexPtr()[i];
        int numElementsRow   = numElementsBlock * innerSizeBlock;

        // Set outer index pointers for dst
        for (int j = 0; j < outerSizeBlock; ++j)
        {
            dst.outerIndexPtr()[j + i * outerSizeBlock + 1] =
                dst.outerIndexPtr()[j + i * outerSizeBlock] + numElementsRow;
        }



        int innerValueOffset = 0;
        for (LhsInnerIterator it(src, i); it; ++it)
        {
            int innerIndexOffset = it.index() * innerSizeBlock;

            for (int j = 0; j < outerSizeBlock; ++j)
            {
                auto dstStart = dst.outerIndexPtr()[i * outerSizeBlock + j];
                for (int k = 0; k < innerSizeBlock; ++k)
                {
                    dst.innerIndexPtr()[dstStart + innerValueOffset + k] = innerIndexOffset + k;

                    if (_options & Eigen::RowMajorBit)
                    {
                        dst.valuePtr()[dstStart + innerValueOffset + k] = it.value().get()(j, k);
                    }
                    else
                    {
                        dst.valuePtr()[dstStart + innerValueOffset + k] = it.value().get()(k, j);
                    }
                }
            }
            //                res.coeffRef(it.index(), c) += (it.value() * rhs.coeff(j, c));
            innerValueOffset += innerSizeBlock;
        }


        if constexpr (_options & Eigen::RowMajorBit)
        {
        }
    }
}



}  // namespace Eigen::Recursive
