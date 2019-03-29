/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/vision/recursiveMatrices/All.h"


namespace Saiga
{
/**
 * Define all types, which are used for pose graph optimization.
 */

const int pgoBlockSizeCamera = 6;
#ifdef RECURSIVE_PGO_FLOAT
using BlockBAScalar = float;
#else
using BlockPGOScalar = double;
#endif

using PGOBlock   = Eigen::Matrix<BlockPGOScalar, pgoBlockSizeCamera, pgoBlockSizeCamera>;
using PGOVector  = Eigen::Matrix<BlockPGOScalar, pgoBlockSizeCamera, 1>;
using PSType     = Eigen::SparseMatrix<Eigen::Recursive::MatrixScalar<PGOBlock>, Eigen::RowMajor>;
using PSDiagType = Eigen::DiagonalMatrix<Eigen::Recursive::MatrixScalar<PGOBlock>, -1>;
using PBType     = Eigen::Matrix<Eigen::Recursive::MatrixScalar<PGOVector>, -1, 1>;

using PGOSolver = Eigen::Recursive::MixedSymmetricRecursiveSolver<PSType, PBType>;
}  // namespace Saiga
