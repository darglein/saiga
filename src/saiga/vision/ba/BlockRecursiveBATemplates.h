/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/vision/recursiveMatrices/All.h"

namespace Saiga
{
// ======================== Types ========================

#ifdef RECURSIVE_BA_VECTORIZE
const int blockSizeCamera = 8;
const int blockSizePoint  = 4;
#else
const int blockSizeCamera = 6;
const int blockSizePoint  = 3;
#endif

#ifdef RECURSIVE_BA_FLOAT
using BlockBAScalar = float;
#else
using BlockBAScalar       = double;
#endif

// block types
using ADiag  = Eigen::Matrix<BlockBAScalar, blockSizeCamera, blockSizeCamera, Eigen::RowMajor>;
using BDiag  = Eigen::Matrix<BlockBAScalar, blockSizePoint, blockSizePoint, Eigen::RowMajor>;
using WElem  = Eigen::Matrix<BlockBAScalar, blockSizeCamera, blockSizePoint, Eigen::RowMajor>;
using WTElem = Eigen::Matrix<BlockBAScalar, blockSizePoint, blockSizeCamera, Eigen::RowMajor>;
using ARes   = Eigen::Matrix<BlockBAScalar, blockSizeCamera, 1>;
using BRes   = Eigen::Matrix<BlockBAScalar, blockSizePoint, 1>;

// Block structured diagonal matrices
using UType = Eigen::DiagonalMatrix<Eigen::Recursive::MatrixScalar<ADiag>, -1>;
using VType = Eigen::DiagonalMatrix<Eigen::Recursive::MatrixScalar<BDiag>, -1>;

// Block structured vectors
using DAType = Eigen::Matrix<Eigen::Recursive::MatrixScalar<ARes>, -1, 1>;
using DBType = Eigen::Matrix<Eigen::Recursive::MatrixScalar<BRes>, -1, 1>;

// Block structured sparse matrix
using WType  = Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<WElem>, Eigen::RowMajor>;
using WTType = Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<WTElem>, Eigen::RowMajor>;
using SType  = Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<ADiag>, Eigen::RowMajor>;


using BAMatrix = Eigen::Recursive::SymmetricMixedMatrix2<UType, VType, WType>;
using BAVector = Eigen::Recursive::MixedVector2<DAType, DBType>;

using BASolver = Eigen::Recursive::MixedSymmetricRecursiveSolver<BAMatrix, BAVector>;
}  // namespace Saiga
