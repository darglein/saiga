/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/vision/recursiveMatrices/RecursiveMatrices_sparse.h"


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
using PSType     = Eigen::SparseMatrix<MatrixScalar<PGOBlock>, Eigen::RowMajor>;
using PSDiagType = Eigen::DiagonalMatrix<MatrixScalar<PGOBlock>, -1>;
using PBType     = Eigen::Matrix<MatrixScalar<PGOVector>, -1, 1>;
}  // namespace Saiga

SAIGA_RM_CREATE_RETURN(Saiga::MatrixScalar<Saiga::PGOBlock>, Saiga::MatrixScalar<Saiga::PGOVector>,
                       Saiga::MatrixScalar<Saiga::PGOVector>);

SAIGA_RM_CREATE_SMV_ROW_MAJOR(Saiga::PBType);
