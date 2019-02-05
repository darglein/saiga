/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/time/performanceMeasure.h"
#include "saiga/core/time/timer.h"
#include "saiga/core/util/random.h"
#include "saiga/vision/recursiveMatrices/RecursiveMatrices.h"

#include "mkl/mkl.h"

#include <fstream>

#include "mkl_cg.h"
#include "mkl_helper.h"

#if !defined(SAIGA_USE_MKL) || !defined(SAIGA_USE_EIGEN)
#    error Saiga was compiled without the required libs for this example.
#endif

// ===================================================================================================
// Performance test parameters for Block Sparse Matrix operations



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
// Block Types
using namespace Saiga;



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
    void sparseMatrixVector(int smv_its);
    void sparseMatrixMatrix(int smm_its);
    void sparseCG(int scg_its, int cg_inner_its);

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

#include "mkl_benchmark_cg.hpp"
#include "mkl_benchmark_mm.hpp"
#include "mkl_benchmark_mv.hpp"
#include "mkl_test.hpp"
