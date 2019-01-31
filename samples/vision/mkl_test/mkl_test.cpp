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

#include "mkl_cg.h"
#include "mkl_helper.h"


#if !defined(SAIGA_USE_MKL) || !defined(SAIGA_USE_EIGEN)
#    error Saiga was compiled without the required libs for this example.
#endif

// ===================================================================================================
// Performance test parameters for Block Sparse Matrix operations

// Type of benchmark

const int mat_mult = false;
const int vec_mult = true;


using T              = double;
const int block_size = 6;

// Matrix dimension (in blocks)
const int n = 4000;
const int m = 4000;

// Non Zero Block per row
const int nnzr = 50;

const int smv_its = 1000;
const int smm_its = 5;
const int scg_its = 500;

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


        Saiga::Random::setSeed(357609435);
#if 0
        // fast creation for non symmetric matrix
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
#else
        // create a symmetric matrix
        std::vector<Eigen::Triplet<Block>> trips;
        trips.reserve(nnzr * n * 2);

        for (int i = 0; i < n; ++i)
        {
            auto indices = Random::uniqueIndices(nnzr, m);
            std::sort(indices.begin(), indices.end());

            for (auto j : indices)
            {
                if (i != j)
                {
                    Block b = Block::Random();
                    trips.emplace_back(i, j, b);
                    trips.emplace_back(j, i, b.transpose());
                }
            }

            // Make sure we have a symmetric diagonal block
            Vector dv = Vector::Random();
            Block D   = dv * dv.transpose();

            // Make sure the matrix is positiv
            D.diagonal() += Vector::Ones() * 5;
            trips.emplace_back(i, i, D);

            x(i) = Vector::Random();
            y(i) = Vector::Random();
        }
        A.setFromTriplets(trips.begin(), trips.end());
        A.makeCompressed();

//        cout << expand(A) << endl;
#endif
        B = A;

        createBlockMKLFromEigen(A, &mkl_A, row_start, row_end, col_index, values, n, m, block_size);


        ex_x = expand(x);
        ex_y = expand(y);

        auto ret =
            createMKL(&mkl_B, row_start.data(), row_end.data(), col_index.data(), values.data(), n, m, block_size);

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
        cout << "." << endl;
        cout << endl;
    }
    void sparseMatrixVector()
    {
        // ============= Benchmark =============

        cout << "Running Block Sparse Matrix-Vector Benchmark..." << endl;
        cout << "Number of Runs: " << smv_its << endl;

        double flop;

        Statistics<float> stat_eigen, stat_mkl;

        //    stat_eigen = measureObject(its, [&]() { y = A * x; });


        // Note: Matrix Vector Mult is exactly #nonzeros FMA instructions
        flop       = double(nnzr) * n * block_size * block_size;
        stat_eigen = measureObject(smv_its, [&]() { y = A * x; });
        stat_mkl   = measureObject(smv_its, [&]() { multMKL(mkl_A, mkl_A_desc, ex_x.data(), ex_y.data()); });


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
        cout << "Median Time Eigen : " << ts_eigen << " -> " << flop / (ts_eigen * 1000 * 1000 * 1000) << " GFlop/s"
             << endl;
        cout << "Median Time MKL   : " << ts_mkl << " -> " << flop / (ts_mkl * 1000 * 1000 * 1000) << " GFlop/s"
             << endl;
        cout << "MKL Speedup: " << (ts_eigen / ts_mkl - 1) * 100 << "%" << endl;
        cout << endl;
    }

    void sparseMatrixMatrix()
    {
        cout << "Running Block Sparse Matrix-Matrix Benchmark..." << endl;
        cout << "Number of Runs: " << smm_its << endl;
        // ============= Benchmark =============
        double flop;
        Statistics<float> stat_eigen, stat_mkl;


        // Note: No idea how many flops there are (it depends on the random initialization of the matrix)
        flop       = double(nnzr) * n * n * block_size * block_size;
        stat_eigen = measureObject(smm_its, [&]() { C = A * B; });
        stat_mkl   = measureObject(smm_its, [&]() { multMKLMM(mkl_A, mkl_B, &mkl_C); });


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
        cout << "Median Time Eigen : " << ts_eigen << " -> " << flop / (ts_eigen * 1000 * 1000 * 1000) << " GFlop/s"
             << endl;
        cout << "Median Time MKL   : " << ts_mkl << " -> " << flop / (ts_mkl * 1000 * 1000 * 1000) << " GFlop/s"
             << endl;
        cout << "MKL Speedup: " << (ts_eigen / ts_mkl - 1) * 100 << "%" << endl;
        cout << endl;
    }

    void sparseCG()
    {
        // ============= Benchmark =============

        T tol = 1e-50;

        cout << "Running Block Sparse CG Benchmark..." << endl;
        cout << "Number of Runs: " << scg_its << endl;
        cout << "Number of inner CG iterations: " << cg_inner_its << endl;


        Statistics<float> stat_eigen, stat_mkl;


        // Main Matrix-Vector > 95%
        double flop_MV = double(nnzr) * n * block_size * block_size * (cg_inner_its + 1);
        // Precond-Vector ~2%
        double flop_PV = double(n) * block_size * block_size * (cg_inner_its + 1);
        // 7 Vector opterations ~2% (vector additions / dotproducts / scalar-vector product)
        double flop_V = double(n) * block_size * (cg_inner_its + 1) * 7;


        double flop = flop_MV + flop_PV + flop_V;

        //         Eigen::IdentityPreconditioner P;
        RecursiveDiagonalPreconditioner<MatrixScalar<Block>> P;
        P.compute(A);

        BlockMatrix Pm(n, m);
        for (int i = 0; i < n; ++i)
        {
            Pm.insert(i, i) = P.getDiagElement(i);
        }
        Pm.makeCompressed();


        // create the mkl preconditioner
        std::vector<T> pvalues;
        std::vector<MKL_INT> pcol_index;
        std::vector<MKL_INT> prow_start;
        std::vector<MKL_INT> prow_end;
        sparse_matrix_t mkl_P;
        createBlockMKLFromEigen(Pm, &mkl_P, prow_start, prow_end, pcol_index, pvalues, n, m, block_size);
        matrix_descr mkl_P_desc;
        mkl_P_desc.type = SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL;
        mkl_P_desc.diag = SPARSE_DIAG_NON_UNIT;

        stat_eigen = measureObject(scg_its, [&]() {
            Eigen::Index iters = cg_inner_its;
            tol                = 1e-50;
            x.setZero();
            recursive_conjugate_gradient([&](const BlockVector& v) { return A * v; }, y, x, P, iters, tol);
        });

        stat_mkl = measureObject(scg_its, [&]() {
            auto iters = cg_inner_its;
            tol        = 1e-50;
            ex_x.setZero();
            mklcg(mkl_A, mkl_A_desc, mkl_P, mkl_P_desc, ex_x.data(), ex_y.data(), tol, iters, n, block_size);
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
        cout << "Median Time Eigen : " << ts_eigen << " -> " << flop / (ts_eigen * 1000 * 1000 * 1000) << " GFlop/s"
             << endl;
        cout << "Median Time MKL   : " << ts_mkl << " -> " << flop / (ts_mkl * 1000 * 1000 * 1000) << " GFlop/s"
             << endl;
        cout << "MKL Speedup: " << (ts_eigen / ts_mkl - 1) * 100 << "%" << endl;
        cout << endl;
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
    if (vec_mult) t.sparseMatrixVector();
    if (mat_mult) t.sparseMatrixMatrix();
    t.sparseCG();
    return 0;
}
