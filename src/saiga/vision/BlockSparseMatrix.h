/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionIncludes.h"
#include "saiga/util/assert.h"
#include <map>

namespace Saiga
{
/**
 * A block diagonal matrix, where each block is a matrix with of type 'BlockType'.
 */

template <typename BlockType>
class BlockSparseMatrix
{
   public:

    using SparseColumn = std::map<int, BlockType>;
using DenseMatrixType = Eigen::Matrix<typename BlockType::Scalar, -1, -1>;

    enum
    {
        _Rows = BlockType::RowsAtCompileTime,
        _Cols = BlockType::ColsAtCompileTime
    };

    BlockSparseMatrix(int n, int m) : n(n),m(m), columns(m) {}




    void setBlock(int i, int j, const BlockType& b){
        SAIGA_ASSERT(i>=0 && i <n && j >=0 && j < m);
        columns[j][i] = b;
    }

    DenseMatrixType dense()
    {
        DenseMatrixType res(n*_Rows,m*_Cols);
        res.setZero();
        for (auto j = 0; j < m; ++j)
        {
            for(auto& p : columns[j])
            {
                int i = p.first;
                res.block(i*_Rows,j*_Cols,_Rows,_Cols) = p.second;
            }
        }
        return res;
    }


    int n,m;
    std::vector<SparseColumn> columns;


};

}  // namespace Saiga
