/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/core/time/performanceMeasure.h"
#include "saiga/core/time/timer.h"
#include "saiga/core/util/random.h"
#include "saiga/vision/recursiveMatrices/CG.h"
#include "saiga/vision/recursiveMatrices/RecursiveMatrices_sparse.h"

#include "mkl/mkl.h"

#include "mkl_cg.h"
#include "mkl_helper.h"

#if !defined(SAIGA_USE_MKL) || !defined(SAIGA_USE_EIGEN)
#    error Saiga was compiled without the required libs for this example.
#endif

// ===================================================================================================
// Performance test parameters for Block Sparse Matrix operations

// Type of benchmark

const int mat_mult = true;
const int vec_mult = false;
const int cg_mult  = false;


using T              = double;
const int block_size = 8;

// Matrix dimension (in blocks)
// divide by block size so the total number of nonzeros stays (roughly) the same by varying block_size
const int n = 1024 * 16 / block_size;
const int m = 1024 * 16 / block_size;

// Non Zero Block per row
const int nnzr = 512 / block_size;

const int smv_its = 100;
const int smm_its = 5;
const int scg_its = 100;

const int cg_inner_its = 5;

// ===================================================================================================
// Output for block sizes 6 and 8 on Xeon E5620 compiled with clang
/*
 * .
 * Block Size : 6x6
 * Matrix Size (in Blocks): 4000x4000
 * Matrix Size Total: 24000x24000
 * Non Zero blocks per row: 50
 * Non Zero BLocks: 200000
 * Non Zeros: 7200000
 * .
 *
 * Running Block Sparse Matrix-Vector Benchmark...
 * Number of Runs: 1000
 * Done.
 * Median Time Eigen : 0.0155178 -> 0.463985 GFlop/s
 * Median Time MKL   : 0.0268985 -> 0.267673 GFlop/s
 * MKL Speedup: -42.31%
 *
 * Running Block Sparse CG Benchmark...
 * Number of Runs: 500
 * Number of inner CG iterations: 5
 * Done.
 * Median Time Eigen : 0.0954034 -> 0.472436 GFlop/s
 * Median Time MKL   : 0.163822 -> 0.275129 GFlop/s
 * MKL Speedup: -41.7638%
 *
 *
 *
 *
 * Block Size : 8x8
 * Matrix Size (in Blocks): 4000x4000
 * Matrix Size Total: 32000x32000
 * Non Zero blocks per row: 50
 * Non Zero BLocks: 200000
 * Non Zeros: 12800000
 * .
 *
 * Running Block Sparse Matrix-Vector Benchmark...
 * Number of Runs: 1000
 * Done.
 * Median Time Eigen : 0.0370042 -> 0.345907 GFlop/s
 * Median Time MKL   : 0.037879 -> 0.337918 GFlop/s
 * MKL Speedup: -2.30946%
 *
 * Running Block Sparse CG Benchmark...
 * Number of Runs: 500
 * Number of inner CG iterations: 5
 * Done.
 * Median Time Eigen : 0.227134 -> 0.350807 GFlop/s
 * Median Time MKL   : 0.227447 -> 0.350323 GFlop/s
 * MKL Speedup: -0.13771%
 */


// ===================================================================================================
// Types and spezializations


using namespace Saiga;
using Block  = Eigen::Matrix<T, block_size, block_size, Eigen::RowMajor>;
using Vector = Eigen::Matrix<T, block_size, 1>;

using BlockVector = Eigen::Matrix<MatrixScalar<Vector>, -1, 1>;
using BlockMatrix = Eigen::SparseMatrix<MatrixScalar<Block>, Eigen::RowMajor>;

SAIGA_RM_CREATE_RETURN(MatrixScalar<Block>, MatrixScalar<Vector>, MatrixScalar<Vector>);
SAIGA_RM_CREATE_SMV_ROW_MAJOR(BlockVector);


using Block1       = Eigen::Matrix<T, 2, 4, Eigen::RowMajor>;
using Block2       = Eigen::Matrix<T, 4, 2, Eigen::RowMajor>;
using Block3       = Eigen::Matrix<T, 2, 2, Eigen::RowMajor>;
using BlockMatrix1 = Eigen::SparseMatrix<MatrixScalar<Block1>, Eigen::RowMajor>;
using BlockMatrix2 = Eigen::SparseMatrix<MatrixScalar<Block2>, Eigen::RowMajor>;
using BlockMatrix3 = Eigen::SparseMatrix<MatrixScalar<Block3>, Eigen::RowMajor>;
SAIGA_RM_CREATE_RETURN(MatrixScalar<Block1>, MatrixScalar<Block2>, MatrixScalar<Block3>);
// SAIGA_RM_CREATE_SMM_YX_RRR_NO_SORT(BlockMatrix1);

namespace Saiga
{
class MKL_Test
{
   public:
    MKL_Test();
    void sparseMatrixVector();
    void sparseMatrixMatrix();
    void sparseCG();

   private:
    // Eigen data structures
    BlockVector x;
    BlockVector y;
    BlockMatrix A, B, C;

    // MKL data structures
    std::vector<T> values;
    std::vector<MKL_INT> col_index;
    std::vector<MKL_INT> row_start;
    std::vector<MKL_INT> row_end;
    Eigen::Matrix<T, -1, 1> ex_x;
    Eigen::Matrix<T, -1, 1> ex_y;
    sparse_matrix_t mkl_A, mkl_B, mkl_C;
    matrix_descr mkl_A_desc, mkl_B_desc;
};
}  // namespace Saiga
