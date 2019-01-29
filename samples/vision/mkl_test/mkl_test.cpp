/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "mkl/mkl.h"

#include "saiga/time/performanceMeasure.h"
#include "saiga/time/timer.h"
#include "saiga/util/random.h"
#include "saiga/vision/recursiveMatrices/RecursiveMatrices_sparse.h"


#if !defined(SAIGA_USE_MKL) || !defined(SAIGA_USE_EIGEN)
#    error Saiga was compiled without the required libs for this example.
#endif


// ===================================================================================================
// Performance test parameters for Block Sparse Matrix operations

// Type of test
// If true:  Matrix-Matrix Product  C = A * B
// If false: Matrix-Vector Product  y = A * x
const int mat_mult = true;


using T              = double;
const int block_size = 9;

// Matrix dimension (in blocks)
const int n = 200;
const int m = 200;

// Non Zero Block per row
const int nnzr = 30;



// ===================================================================================================
// Types and spezializations

using namespace Saiga;

using Block  = Eigen::Matrix<T, block_size, block_size, Eigen::RowMajor>;
using Vector = Eigen::Matrix<T, block_size, 1>;

using BlockVector = Eigen::Matrix<MatrixScalar<Vector>, -1, 1>;
using BlockMatrix = Eigen::SparseMatrix<MatrixScalar<Block>, Eigen::RowMajor>;

SAIGA_RM_CREATE_RETURN(MatrixScalar<Block>, MatrixScalar<Vector>, MatrixScalar<Vector>);
SAIGA_RM_CREATE_SMV_ROW_MAJOR(BlockVector);

inline void multMKL(const sparse_matrix_t A, struct matrix_descr descr, const double* x, double* y)
{
    auto ret = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, A, descr, x, 0, y);
    SAIGA_ASSERT(ret == SPARSE_STATUS_SUCCESS);
}

inline void multMKL(const sparse_matrix_t A, struct matrix_descr descr, const float* x, float* y)
{
    auto ret = mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, A, descr, x, 0, y);
    SAIGA_ASSERT(ret == SPARSE_STATUS_SUCCESS);
}

inline void multMKLMM(const sparse_matrix_t A, const sparse_matrix_t B, sparse_matrix_t* C)
{
    auto ret = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, A, B, C);
    SAIGA_ASSERT(ret == SPARSE_STATUS_SUCCESS);
}

inline auto createMKL(sparse_matrix_t* A, MKL_INT* rows_start, MKL_INT* rows_end, MKL_INT* col_indx, double* values)
{
    return mkl_sparse_d_create_bsr(A, SPARSE_INDEX_BASE_ZERO, SPARSE_LAYOUT_ROW_MAJOR, n, m, block_size, rows_start,
                                   rows_end, col_indx, values);
}

inline auto createMKL(sparse_matrix_t* A, MKL_INT* rows_start, MKL_INT* rows_end, MKL_INT* col_indx, float* values)
{
    return mkl_sparse_s_create_bsr(A, SPARSE_INDEX_BASE_ZERO, SPARSE_LAYOUT_ROW_MAJOR, n, m, block_size, rows_start,
                                   rows_end, col_indx, values);
}


void sparseMatrixVector()
{
    // ============= Create the Eigen Recursive Data structures =============
    BlockVector x(m);
    BlockVector y(n);
    BlockMatrix A(n, m), B, C;

    A.reserve(n * nnzr);
    for (int i = 0; i < n; ++i)
    {
        auto indices = Random::uniqueIndices(nnzr, m);
        std::sort(indices.begin(), indices.end());

        A.startVec(i);
        for (auto j : indices)
        {
            A.insertBackByOuterInner(i, j) = Block::Random();
        }
        x(i) = Vector::Random();
    }
    A.finalize();
    B = A;
    //    A.makeCompressed();



    // ============= Create the MKL Data structures =============
    std::vector<T> values;
    std::vector<MKL_INT> col_index;
    std::vector<MKL_INT> row_start;
    std::vector<MKL_INT> row_end;

    for (int k = 0; k < A.outerSize(); ++k)
    {
        row_start.push_back(A.outerIndexPtr()[k]);
        row_end.push_back(A.outerIndexPtr()[k + 1]);
        for (BlockMatrix::InnerIterator it(A, k); it; ++it)
        {
            col_index.push_back(it.index());

            for (auto i = 0; i < block_size; ++i)
            {
                for (auto j = 0; j < block_size; ++j)
                {
                    auto block = it.valueRef();
                    values.push_back(block.get()(i, j));
                }
            }
        }
    }
    auto ex_x = expand(x);
    auto ex_y = expand(y);
    ex_y.setZero();

    sparse_matrix_t mkl_A, mkl_B, mkl_C;
    auto ret = createMKL(&mkl_A, row_start.data(), row_end.data(), col_index.data(), values.data());
    ret      = createMKL(&mkl_B, row_start.data(), row_end.data(), col_index.data(), values.data());
    matrix_descr mkl_A_desc, mkl_B_desc, mkl_C_desc;
    mkl_A_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_B_desc      = mkl_A_desc;
    SAIGA_ASSERT(ret == SPARSE_STATUS_SUCCESS);
    mkl_set_num_threads_local(1);
    mkl_set_num_threads(1);
    cout << "Init done." << endl;



    // ============= Benchmark =============

    int its = 50;

    // Print some stats
    cout << "." << endl;
    cout << "Block Size : " << block_size << "x" << block_size << endl;
    cout << "Matrix Size (in Blocks): " << n << "x" << m << endl;
    cout << "Matrix Size Total: " << n * block_size << "x" << m * block_size << endl;
    cout << "Non Zero blocks per row: " << nnzr << endl;
    cout << "Non Zero BLocks: " << nnzr * n << endl;
    cout << "Non Zeros: " << nnzr * n * block_size * block_size << endl;
    cout << "Test Iterations: " << its << endl;
    cout << "." << endl;


    double flop;

    Statistics<float> stat_eigen, stat_mkl;

    //    stat_eigen = measureObject(its, [&]() { y = A * x; });

    if (mat_mult)
    {
        // Note: No idea how many flops there are (it depends on the random initialization of the matrix)
        flop       = double(nnzr) * n * n * block_size * block_size;
        stat_eigen = measureObject(its, [&]() { C = A * B; });
        stat_mkl   = measureObject(its, [&]() { multMKLMM(mkl_A, mkl_B, &mkl_C); });


#if 0
        MKL_INT rows, cols, block_size;
        MKL_INT *rows_start, *rows_end, *col_indx;
        float* values;
        sparse_index_base_t indexing;
        sparse_layout_t block_layout;
        mkl_sparse_s_export_bsr(mkl_C, &indexing, &block_layout, &rows, &cols, &block_size, &rows_start, &rows_end,
                                &col_indx, &values);
        cout << indexing << " " << block_layout << " " << rows << " " << cols << " " << block_size << endl;
#endif
    }
    else
    {
        // Note: Matrix Vector Mult is exactly #nonzeros FMA instructions
        flop       = double(nnzr) * n * block_size * block_size;
        stat_eigen = measureObject(its, [&]() { y = A * x; });
        stat_mkl   = measureObject(its, [&]() { multMKL(mkl_A, mkl_A_desc, ex_x.data(), ex_y.data()); });
    }

#if 0
    // More precise timing stats
    cout << stat_eigen << endl;
    cout << stat_mkl << endl;
    cout << endl;
#endif
    // time in seconds
    double ts_eigen = stat_eigen.median / 1000.0;
    double ts_mkl   = stat_mkl.median / 1000.0;

    cout << "Done." << endl;

    cout << "Time Eigen : " << ts_eigen << " " << flop / (ts_eigen * 1000 * 1000 * 1000) << " GFlop/s" << endl;
    cout << "Time MKL   : " << ts_mkl << " " << flop / (ts_mkl * 1000 * 1000 * 1000) << " GFlop/s" << endl;
    cout << "MKL Speedup: " << (ts_eigen / ts_mkl - 1) * 100 << "%" << endl;
}

int main(int argc, char* argv[])
{
    Saiga::EigenHelper::checkEigenCompabitilty<2765>();

    sparseMatrixVector();
    return 0;
}
