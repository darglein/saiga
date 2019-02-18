/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 16

#include "mkl_test.h"

namespace Saiga
{
void MKL_Test::sparseMatrixMatrix()
{
    cout << "Running Block Sparse Matrix-Matrix Benchmark..." << endl;
    cout << "Number of Runs: " << smm_its << endl;
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
//        cout << "error " << error << endl;

//        cout << expand(C) << endl << endl;

//        cout << expand(A) * expand(B) << endl << endl;

//        exit(0);

//        C.makeCompressed();
//        int c_nnz = C.nonZeros();
//        cout << C.nonZeros() << " = " << A.nonZeros() << " * " << B.nonZeros() << endl;
#endif



    C.resize(n, m);
    stat_eigen = measureObject(smm_its, [&]() {  // We call the spmm kernel directly, because the multiplication
                                                 // with operator* also sorts the output columns ascending.
        // The mkl spmm doesn't sort so a comparison would have been unfair.
        Eigen::internal::conservative_sparse_sparse_product_impl_yx(A, B, C);
    });
    stat_mkl = measureObject(smm_its, [&]() { mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, mkl_A, mkl_B, &mkl_C); });


#if 0
        MKL_INT rows, cols, block_size;
        MKL_INT *rows_start, *rows_end, *col_indx;
        double* values;
        sparse_index_base_t indexing;
        sparse_layout_t block_layout;
        mkl_sparse_d_export_bsr(mkl_C, &indexing, &block_layout, &rows, &cols, &block_size, &rows_start, &rows_end,
                                &col_indx, &values);
        cout << indexing << " " << block_layout << " " << rows << " " << cols << " " << block_size << " "
             << rows_end[rows - 1] << endl
             << endl;
        ;


        Eigen::Matrix<MatrixScalar<Block>, -1, -1> test_C(n, m);
        test_C.setZero();

#    if 0
        for (int i = 0; i < rows; ++i)
        {
            cout << rows_start[i] << " ";
        }
        cout << endl;
        for (int i = 0; i < rows; ++i)
        {
            cout << rows_end[i] << " ";
        }
        cout << endl;
        for (int i = 0; i < rows_end[rows - 1]; ++i)
        {
            cout << col_indx[i] << " ";
        }
        cout << endl;
        for (int i = 0; i < rows_end[rows - 1]; ++i)
        {
            cout << C.innerIndexPtr()[i] << " ";
        }
        cout << endl;
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
        cout << "diff " << diff << endl;
        //        cout << expand(test_C) << endl << endl;
        //        for (int i = 0; i < block_size * block_size; ++i)
        //        {
        //            cout << values[i] << " ";
        //        }
        //        cout << endl;

        //        cout << expand(C) << endl << endl;

        //        cout << expand(A) * expand(B) << endl << endl;
        exit(0);
//        for (int i = 0; i < block_size * block_size; ++i)
//        {
//            cout << C.valuePtr()[0].get().data()[i] << " ";
//        }
//        cout << endl;
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
    cout << "Median Time MKL   : " << ts_mkl << " -> " << flop / (ts_mkl * 1000 * 1000 * 1000) << " GFlop/s" << endl;
    cout << "MKL Speedup: " << (ts_eigen / ts_mkl - 1) * 100 << "%" << endl;
    cout << endl;
}

}  // namespace Saiga