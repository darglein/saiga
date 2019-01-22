/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/MatrixScalar.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/recursiveMatrices/Expand.h"
#include "saiga/vision/recursiveMatrices/Transpose.h"

#include "Eigen/Sparse"

namespace Saiga
{
// ======================== Types ========================


const int pgoBlockSizeCamera = 6;


#ifdef RECURSIVE_PGO_FLOAT
using BlockBAScalar = float;
#else
using BlockBAScalar = double;
#endif


using PGOBlock   = Eigen::Matrix<BlockBAScalar, pgoBlockSizeCamera, pgoBlockSizeCamera>;
using PGOVector  = Eigen::Matrix<BlockBAScalar, pgoBlockSizeCamera, 1>;
using PSType     = Eigen::SparseMatrix<MatrixScalar<PGOBlock>, Eigen::RowMajor>;
using PSDiagType = Eigen::DiagonalMatrix<MatrixScalar<PGOBlock>, -1>;
// using PSType = Eigen::Matrix<MatrixScalar<PGOBlock>, -1, -1, Eigen::RowMajor>;
using PBType = Eigen::Matrix<MatrixScalar<PGOVector>, -1, 1>;



}  // namespace Saiga
