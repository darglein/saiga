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
class BlockVector
{
   public:
    using DenseVectorType = Eigen::Matrix<typename BlockType::Scalar, -1, 1>;

    enum
    {
        _Rows = BlockType::RowsAtCompileTime
    };

    BlockVector(int s) : blocks(s) {}

    void setZero()
    {
        for (auto& b : blocks)
        {
            b.setZero();
        }
    }


    DenseVectorType dense()
    {
        DenseVectorType res(blocks.size() * _Rows);
        res.setZero();
        for (auto i = 0; i < blocks.size(); ++i)
        {
            res.segment(i * _Rows, _Rows) = blocks[i];
        }
        return res;
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
