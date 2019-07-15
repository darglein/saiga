/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/random.h"
#include "saiga/core/time/performanceMeasure.h"
#include "saiga/core/time/timer.h"

#include "EigenRecursive/All.h"

#if defined(SAIGA_USE_MKL)
#include "mkl.h"
#include "saiga/vision/mkl/mkl_helper.h"
#endif

#include <fstream>




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
        #if defined(SAIGA_USE_MKL)
        mkl_sparse_destroy(mkl_A);
        mkl_sparse_destroy(mkl_B);
#endif
    }

    void sparseMatrixVector(int smv_its);
    void sparseMatrixMatrix(int smm_its);
    void testResultMatrixMatrix();
void testResultMatrixVector();
   private:
    // Recursive data structures
    BlockVector x;
    BlockVector y;
    BlockMatrix A, B, C;

    #if defined(SAIGA_USE_MKL)
    // MKL data structures
    Eigen::Matrix<T, -1, 1> ex_x;
    Eigen::Matrix<T, -1, 1> ex_y;
    sparse_matrix_t mkl_A, mkl_B, mkl_C;
    matrix_descr mkl_A_desc, mkl_B_desc;
#endif
};


template <typename T, int block_size, int factor>
inline MKL_Test<T, block_size, factor>::MKL_Test()
{
    A.resize(n, m);
    B.resize(n, m);
    C.resize(n, m);
    x.resize(n);
    y.resize(n);

    // ============= Create the Eigen Recursive Data structures =============
    Saiga::Random::setSeed(357609435);

    // fast creation for non symmetric matrix
    A.reserve(n * nnzr);
    for (int i = 0; i < n; ++i)
    {
        auto indices = Random::uniqueIndices(nnzr, m);
        std::sort(indices.begin(), indices.end());

        A.startVec(i);
        for (auto j : indices)
        {
            Block b;
            setRandom(b);
            A.insertBackByOuterInner(i, j) = b;
        }
        x(i) = Vector::Random();
        y(i) = Vector::Random();
    }
    A.finalize();

    B = A;


    #if defined(SAIGA_USE_MKL)
        // mkl matrix
        createBlockMKLFromEigen(A, &mkl_A, &mkl_A_desc, block_size);
        createBlockMKLFromEigen(B, &mkl_B, &mkl_B_desc, block_size);
        ex_x = expand(x);
        ex_y = expand(y);
        mkl_set_num_threads_local(1);
        mkl_set_num_threads(1);
    #endif

    std::cout << "Init done." << std::endl;

    // Print some stats
    std::cout << "." << std::endl;
    std::cout << "Block Size : " << block_size << "x" << block_size << std::endl;
    std::cout << "Matrix Size (in Blocks): " << n << "x" << m << std::endl;
    std::cout << "Matrix Size Total: " << n * block_size << "x" << m * block_size << std::endl;
    std::cout << "Non Zero blocks per row: " << nnzr << std::endl;
    std::cout << "Non Zero BLocks: " << nnzr * n << std::endl;
    std::cout << "Non Zeros: " << nnzr * n * block_size * block_size << std::endl;
    std::cout << "." << std::endl;
    std::cout << std::endl;
}

}  // namespace Saiga


#include "mkl_benchmark_mm.hpp"
#include "mkl_benchmark_mv.hpp"

