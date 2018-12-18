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
/**
 * A block diagonal matrix, where each block is a matrix with of type 'BlockType'.
 */

template <typename BlockType>
class BlockDiagonalMatrix
{
   public:
    using VectorType      = Eigen::Matrix<typename BlockType::Scalar, -1, 1>;
    using DenseMatrixType = Eigen::Matrix<typename BlockType::Scalar, -1, -1>;


    enum
    {
        _Rows = BlockType::RowsAtCompileTime
    };

    BlockDiagonalMatrix(size_t numBlocks) : blocks(numBlocks) {}
    int size() { return (int)blocks.size(); }
    int elements() { return size() * _Rows; }

    /**
     * Matrix-Vector multiplication.
     */
    VectorType operator*(const VectorType& v)
    {
        VectorType res(elements());
        for (auto i = 0; i < size(); ++i)
        {
            res.segment(i * _Rows, _Rows) = blocks[i] * res.segment(i * _Rows, _Rows);
        }
        return res;
    }

    /**
     * Solve Ax=b and returns x
     */
    VectorType solve(const VectorType& b)
    {
        VectorType x(elements());
        for (auto i = 0; i < size(); ++i)
        {
            x.segment(i * _Rows, _Rows) = blocks[i].ldlt().solve(b.segment(i * _Rows, _Rows));
        }
        return x;
    }

    DenseMatrixType inverse()
    {
        DenseMatrixType res(elements(), elements());
        res.setZero();
        for (auto i = 0; i < size(); ++i)
        {
            res.block(i * _Rows, i * _Rows, _Rows, _Rows) = blocks[i].inverse();
            //            res.block(i * _Rows, i * _Rows, _Rows, _Rows) = blocks[i].ldlt().solve(BlockType::Identity());
        }
        return res;
    }

    DenseMatrixType dense()
    {
        DenseMatrixType res(elements(), elements());
        res.setZero();
        for (auto i = 0; i < size(); ++i)
        {
            res.block(i * _Rows, i * _Rows, _Rows, _Rows) = blocks[i];
        }
        return res;
    }

    void addToDiagonal(const VectorType& b)
    {
        SAIGA_ASSERT(b.rows() == elements());
        for (auto i = 0; i < size(); ++i)
        {
            for (int r = 0; r < _Rows; ++r)
            {
                blocks[i](r, r) += b(i * _Rows + r);
            }
        }
    }



    void setZero()
    {
        for (auto& b : blocks)
        {
            b.setZero();
        }
    }

    BlockType& operator()(int i)
    {
        SAIGA_ASSERT(i >= 0 && i < blocks.size());
        return blocks[i];
    }
    const BlockType& operator()(int i) const
    {
        SAIGA_ASSERT(i >= 0 && i < blocks.size());
        return blocks[i];
    }

   private:
    AlignedVector<BlockType> blocks;
};

}  // namespace Saiga
