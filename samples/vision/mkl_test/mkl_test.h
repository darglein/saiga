/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/time/performanceMeasure.h"
#include "saiga/core/time/timer.h"
#include "saiga/core/util/random.h"
#include "saiga/vision/EigenRecursive/All.h"
#include "saiga/vision/mkl/mkl_cg.h"
#include "saiga/vision/mkl/mkl_helper.h"

#include "mkl.h"

#include <fstream>

#if !defined(SAIGA_USE_MKL) || !defined(SAIGA_USE_EIGEN)
#    error Saiga was compiled without the required libs for this example.
#endif



// ===================================================================================================
// Block Types
using namespace Saiga;
using namespace Eigen::Recursive;


namespace Saiga
{
template <typename T, int block_size, int factor>
class MKL_Test
{
    // Matrix dimension (in blocks)
    // divide by block size so the total number of nonzeros stays (roughly) the same by varying block_size
    const int n = 1024 * factor / block_size;
    const int m = 1024 * factor / block_size;

    // Non Zero Block per row
    const int nnzr = 32 * factor / block_size;



    using Block  = Eigen::Matrix<T, block_size, block_size, Eigen::RowMajor>;
    using Vector = Eigen::Matrix<T, block_size, 1>;

    using BlockVector = Eigen::Matrix<MatrixScalar<Vector>, -1, 1>;
    using BlockMatrix = Eigen::SparseMatrix<MatrixScalar<Block>, Eigen::RowMajor>;

   public:
    MKL_Test();
    ~MKL_Test()
    {
        mkl_sparse_destroy(mkl_A);
        mkl_sparse_destroy(mkl_B);
    }

    void sparseMatrixVector(int smv_its);
    void sparseMatrixMatrix(int smm_its);
    void sparseCG(int scg_its, int cg_inner_its);

   private:
    // Eigen data structures
    BlockVector x;
    BlockVector y;
    BlockMatrix A, B, C;

    // MKL data structures
    Eigen::Matrix<T, -1, 1> ex_x;
    Eigen::Matrix<T, -1, 1> ex_y;
    sparse_matrix_t mkl_A, mkl_B, mkl_C;
    matrix_descr mkl_A_desc, mkl_B_desc;
};
}  // namespace Saiga

#include "mkl_benchmark_cg.hpp"
#include "mkl_benchmark_mm.hpp"
#include "mkl_benchmark_mv.hpp"
#include "mkl_test.hpp"
