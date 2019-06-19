/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "mkl_test.h"

namespace Saiga
{
template <typename T, int block_size, int factor>
inline void MKL_Test<T, block_size, factor>::sparseMatrixMatrix(int smm_its)
{
    std::ofstream strm("eigen_mkl_mm.csv", std::ostream::app);

    std::cout << "Running Block Sparse Matrix-Matrix Benchmark..." << std::endl;
    std::cout << "Number of Runs: " << smm_its << std::endl;
    // ============= Benchmark =============
    double flop;
    Statistics<float> stat_eigen, stat_mkl;

    // Compute stats

    long inner_mm = 0;

    // The inner block product is most likely evaluated with a simple n^3 algorithm
    // -> n^3 multiply-add instructions
    long inner_flops = block_size * block_size * block_size;

    {
        // compute the number of inner block-block multiplications

        using namespace Eigen;



        auto lhs = A;
        auto rhs = B;

        using Rhs = BlockMatrix;
        using Lhs = BlockMatrix;

        Index cols = rhs.outerSize();

        for (Index j = 0; j < cols; ++j)
        {
            for (typename Rhs::InnerIterator rhsIt(rhs, j); rhsIt; ++rhsIt)
            {
                Index k = rhsIt.index();
                for (typename Lhs::InnerIterator lhsIt(lhs, k); lhsIt; ++lhsIt)
                {
                    inner_mm++;
                }
            }
        }
    }

    // This should be a lower bound approximation of the required flops
    // There might be one additional block-add per inner_mm, but that
    // could be optimized away by the compiler, so we omit it here.
    flop = inner_mm * inner_flops;
#if 1
    //        C = A * B;

    //        C.resize(n, m);
    //        C.setZero();
    //        Eigen::internal::conservative_sparse_sparse_product_impl_yx(A, B, C);

    //        Eigen::SparseMatrix<MatrixScalar<Block>> asdf = C;
    //        C                                             = asdf;


//        double error = (expand(C) - expand(A) * expand(B)).norm();
//        std::cout << "error " << error << std::endl;

//        std::cout << expand(C) << std::endl << std::endl;

//        std::cout << expand(A) * expand(B) << std::endl << std::endl;

//        exit(0);

//        C.makeCompressed();
//        int c_nnz = C.nonZeros();
//        std::cout << C.nonZeros() << " = " << A.nonZeros() << " * " << B.nonZeros() << std::endl;
#endif



    C.resize(n, m);
    stat_eigen = measureObject(smm_its, [&]() {  // We call the spmm kernel directly, because the multiplication
                                                 // with operator* also sorts the output columns ascending.
        // The mkl spmm doesn't sort so a comparison would have been unfair.
        C = BlockMatrix(n, m);
        Eigen::internal::conservative_sparse_sparse_product_impl_yx(A, B, C);
    });
    stat_mkl   = measureObject(smm_its, [&]() {
        mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, mkl_A, mkl_B, &mkl_C);
        mkl_sparse_destroy(mkl_C);
    });



#if 0
        MKL_INT rows, cols, block_size;
        MKL_INT *rows_start, *rows_end, *col_indx;
        double* values;
        sparse_index_base_t indexing;
        sparse_layout_t block_layout;
        mkl_sparse_d_export_bsr(mkl_C, &indexing, &block_layout, &rows, &cols, &block_size, &rows_start, &rows_end,
                                &col_indx, &values);
        std::cout << indexing << " " << block_layout << " " << rows << " " << cols << " " << block_size << " "
             << rows_end[rows - 1] << std::endl
             << std::endl;
        ;


        Eigen::Matrix<MatrixScalar<Block>, -1, -1> test_C(n, m);
        test_C.setZero();

#    if 0
        for (int i = 0; i < rows; ++i)
        {
            std::cout << rows_start[i] << " ";
        }
        std::cout << std::endl;
        for (int i = 0; i < rows; ++i)
        {
            std::cout << rows_end[i] << " ";
        }
        std::cout << std::endl;
        for (int i = 0; i < rows_end[rows - 1]; ++i)
        {
            std::cout << col_indx[i] << " ";
        }
        std::cout << std::endl;
        for (int i = 0; i < rows_end[rows - 1]; ++i)
        {
            std::cout << C.innerIndexPtr()[i] << " ";
        }
        std::cout << std::endl;
#    endif

        for (int r = 0; r < rows; ++r)
        {
            for (int s = rows_start[r]; s < rows_end[r]; ++s)
            {
                int c = col_indx[s];
                Eigen::Map<Block> bm(values + s * block_size * block_size);
                test_C(r, c).get() += bm;
            }
        }


        double diff = (expand(C) - expand(test_C)).norm();
        std::cout << "diff " << diff << std::endl;
        //        std::cout << expand(test_C) << std::endl << std::endl;
        //        for (int i = 0; i < block_size * block_size; ++i)
        //        {
        //            std::cout << values[i] << " ";
        //        }
        //        std::cout << std::endl;

        //        std::cout << expand(C) << std::endl << std::endl;

        //        std::cout << expand(A) * expand(B) << std::endl << std::endl;
        exit(0);
//        for (int i = 0; i < block_size * block_size; ++i)
//        {
//            std::cout << C.valuePtr()[0].get().data()[i] << " ";
//        }
//        std::cout << std::endl;
#endif



#if 0
        // More precise timing stats
        std::cout << stat_eigen << std::endl;
        std::cout << stat_mkl << std::endl;
        std::cout << std::endl;
#endif
    // time in seconds
    double ts_eigen = stat_eigen.median / 1000.0;
    double ts_mkl   = stat_mkl.median / 1000.0;

    double gflop_eigen = flop / (ts_eigen * 1000 * 1000 * 1000);
    double gflop_mkl   = flop / (ts_mkl * 1000 * 1000 * 1000);

    std::cout << "Done." << std::endl;
    std::cout << "Median Time Eigen : " << ts_eigen << " -> " << gflop_eigen << " GFlop/s" << std::endl;
    std::cout << "Median Time MKL   : " << ts_mkl << " -> " << gflop_mkl << " GFlop/s" << std::endl;
    std::cout << "Eigen Speedup: " << (ts_mkl / ts_eigen) << std::endl;
    std::cout << std::endl;

    strm << block_size << "," << n << "," << nnzr << "," << typeid(T).name() << "," << ts_eigen << "," << gflop_eigen
         << "," << ts_mkl << "," << gflop_mkl << "," << (ts_mkl / ts_eigen) << ",1" << std::endl;
}

}  // namespace Saiga
