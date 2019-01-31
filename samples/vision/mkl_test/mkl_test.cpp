/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/time/performanceMeasure.h"
#include "saiga/time/timer.h"
#include "saiga/util/random.h"
#include "saiga/vision/recursiveMatrices/CG.h"
#include "saiga/vision/recursiveMatrices/RecursiveMatrices_sparse.h"

#include "mkl_cg.h"
#include "mkl_helper.h"


#if !defined(SAIGA_USE_MKL) || !defined(SAIGA_USE_EIGEN)
#    error Saiga was compiled without the required libs for this example.
#endif

// ===================================================================================================
// Performance test parameters for Block Sparse Matrix operations

// Type of benchmark

// Matrix-Matrix mult
const int mat_mult = false;
// Matrix-Vecor mult
const int vec_mult = true;


using T              = double;
const int block_size = 2;

// Matrix dimension (in blocks)
const int n = 4;
const int m = 4;

// Non Zero Block per row
const int nnzr = 2;

const int its = 50;


// ===================================================================================================
// Types and spezializations


using namespace Saiga;
using Block  = Eigen::Matrix<T, block_size, block_size, Eigen::RowMajor>;
using Vector = Eigen::Matrix<T, block_size, 1>;

using BlockVector = Eigen::Matrix<MatrixScalar<Vector>, -1, 1>;
using BlockMatrix = Eigen::SparseMatrix<MatrixScalar<Block>, Eigen::RowMajor>;

SAIGA_RM_CREATE_RETURN(MatrixScalar<Block>, MatrixScalar<Vector>, MatrixScalar<Vector>);
SAIGA_RM_CREATE_SMV_ROW_MAJOR(BlockVector);


namespace Saiga
{
class MKL_Test
{
   public:
    MKL_Test()
    {
        A.resize(n, m);
        B.resize(n, m);
        C.resize(n, m);
        x.resize(n);
        y.resize(n);

        // ============= Create the Eigen Recursive Data structures =============
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
            y(i) = Vector::Random();
        }
        A.finalize();
        B = A;


        // ============= Create the MKL Data structures =============
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
        ex_x = expand(x);
        ex_y = expand(y);

        auto ret =
            createMKL(&mkl_A, row_start.data(), row_end.data(), col_index.data(), values.data(), n, m, block_size);
        ret = createMKL(&mkl_B, row_start.data(), row_end.data(), col_index.data(), values.data(), n, m, block_size);

        mkl_A_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
        mkl_B_desc      = mkl_A_desc;
        SAIGA_ASSERT(ret == SPARSE_STATUS_SUCCESS);
        mkl_set_num_threads_local(1);
        mkl_set_num_threads(1);
        cout << "Init done." << endl;

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
    }
    void sparseMatrixVector()
    {
        // ============= Benchmark =============

        cout << "Running Sparse Matrix-Vector Benchmark..." << endl;

        double flop;

        Statistics<float> stat_eigen, stat_mkl;

        //    stat_eigen = measureObject(its, [&]() { y = A * x; });


        // Note: Matrix Vector Mult is exactly #nonzeros FMA instructions
        flop       = double(nnzr) * n * block_size * block_size;
        stat_eigen = measureObject(its, [&]() { y = A * x; });
        stat_mkl   = measureObject(its, [&]() { multMKL(mkl_A, mkl_A_desc, ex_x.data(), ex_y.data()); });


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

    void sparseMatrixMatrix()
    {
        cout << "Running Sparse Matrix-Matrix Benchmark..." << endl;
        // ============= Benchmark =============
        double flop;
        Statistics<float> stat_eigen, stat_mkl;


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

    void sparseCG()
    {
        // ============= Benchmark =============

        cout << "Running Sparse CG Benchmark..." << endl;

        double flop;

        Statistics<float> stat_eigen, stat_mkl;

        //    stat_eigen = measureObject(its, [&]() { y = A * x; });


        // Note: Matrix Vector Mult is exactly #nonzeros FMA instructions
        flop = double(nnzr) * n * block_size * block_size;
        //        stat_eigen = measureObject(its, [&]() { y = A * x; });
        //        stat_mkl   = measureObject(its, [&]() { multMKL(mkl_A, mkl_A_desc, ex_x.data(), ex_y.data()); });

        Eigen::Index iters = 1;
        T tol              = 1e-50;

        int its  = 1;
        stat_mkl = measureObject(its, [&]() {
            ex_x.setZero();
            //            x.setZero();
            //            Eigen::IdentityPreconditioner P;

            mklcg(mkl_A, mkl_A_desc, ex_x.data(), ex_y.data(), tol, iters, n, block_size);

            //            recursive_conjugate_gradient([&](const BlockVector& v) { return A * v; }, y, x, P, iters,
            //            tol);
        });


        stat_eigen = measureObject(its, [&]() {
            x.setZero();
            Eigen::IdentityPreconditioner P;


            recursive_conjugate_gradient([&](const BlockVector& v) { return A * v; }, y, x, P, iters, tol);
        });
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


int main(int argc, char* argv[])
{
    Saiga::EigenHelper::checkEigenCompabitilty<2765>();

    Saiga::MKL_Test t;
    t.sparseCG();
    //    if (vec_mult) t.sparseMatrixVector();
    //    if (mat_mult) t.sparseMatrixMatrix();
    return 0;
}
