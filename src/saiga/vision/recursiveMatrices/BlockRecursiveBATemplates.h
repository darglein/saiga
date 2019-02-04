/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/vision/recursiveMatrices/RecursiveMatrices_sparse.h"

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
using ADiag  = Eigen::Matrix<BlockBAScalar, blockSizeCamera, blockSizeCamera>;
using BDiag  = Eigen::Matrix<BlockBAScalar, blockSizePoint, blockSizePoint>;
using WElem  = Eigen::Matrix<BlockBAScalar, blockSizeCamera, blockSizePoint>;
using WTElem = Eigen::Matrix<BlockBAScalar, blockSizePoint, blockSizeCamera>;
using ARes   = Eigen::Matrix<BlockBAScalar, blockSizeCamera, 1>;
using BRes   = Eigen::Matrix<BlockBAScalar, blockSizePoint, 1>;

// Block structured diagonal matrices
using UType = Eigen::DiagonalMatrix<MatrixScalar<ADiag>, -1>;
using VType = Eigen::DiagonalMatrix<MatrixScalar<BDiag>, -1>;

// Block structured vectors
using DAType = Eigen::Matrix<MatrixScalar<ARes>, -1, 1>;
using DBType = Eigen::Matrix<MatrixScalar<BRes>, -1, 1>;

// Block structured sparse matrix
using WType  = Eigen::SparseMatrix<MatrixScalar<WElem>, Eigen::RowMajor>;
using WTType = Eigen::SparseMatrix<MatrixScalar<WTElem>, Eigen::RowMajor>;
using SType  = Eigen::SparseMatrix<MatrixScalar<ADiag>, Eigen::RowMajor>;



}  // namespace Saiga



#if 0
// A * a = a
SAIGA_RM_CREATE_RETURN(Saiga::MatrixScalar<Saiga::ADiag>, Saiga::MatrixScalar<Saiga::ARes>,
                       Saiga::MatrixScalar<Saiga::ARes>);

// W * WT = A
SAIGA_RM_CREATE_RETURN(Saiga::MatrixScalar<Saiga::WElem>, Saiga::MatrixScalar<Saiga::WTElem>,
                       Saiga::MatrixScalar<Saiga::ADiag>);

// W * B = W
SAIGA_RM_CREATE_RETURN(Saiga::MatrixScalar<Saiga::WElem>, Saiga::MatrixScalar<Saiga::BDiag>,
                       Saiga::MatrixScalar<Saiga::WElem>);

// W * b = a
SAIGA_RM_CREATE_RETURN(Saiga::MatrixScalar<Saiga::WElem>, Saiga::MatrixScalar<Saiga::BRes>,
                       Saiga::MatrixScalar<Saiga::ARes>);

// WT * a = b
SAIGA_RM_CREATE_RETURN(Saiga::MatrixScalar<Saiga::WTElem>, Saiga::MatrixScalar<Saiga::ARes>,
                       Saiga::MatrixScalar<Saiga::BRes>);
#endif

// A * a
// SAIGA_RM_CREATE_SMV_ROW_MAJOR(Saiga::DAType);
// B * b
// SAIGA_RM_CREATE_SMV_ROW_MAJOR(Saiga::DBType);
// S = Y * WT
// SAIGA_RM_CREATE_SMM_YX_RRC(Saiga::WType);
// SAIGA_RM_CREATE_SMM_YX_RCC(Saiga::WType);
// SAIGA_RM_CREATE_SMM_YX_RCR(Saiga::WType);
