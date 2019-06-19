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
inline void MKL_Test<T, block_size, factor>::sparseMatrixVector(int smv_its)
{
    std::ofstream strm("eigen_mkl_mv.csv", std::ostream::app);
    // ============= Benchmark =============

    std::cout << "Running Block Sparse Matrix-Vector Benchmark..." << std::endl;
    std::cout << "Number of Runs: " << smv_its << std::endl;

    double flop;

    Statistics<float> stat_eigen, stat_mkl;

    //    stat_eigen = measureObject(its, [&]() { y = A * x; });

#if 0
    // Test if the result is correct
    y.setZero();
    y = A * x;
    std::cout << expand(x).norm() << " " << expand(y).norm() << std::endl;

    ex_y.setZero();
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, mkl_A, mkl_A_desc, ex_x.data(), 0, ex_y.data());
    std::cout << expand(ex_x).norm() << " " << expand(ex_y).norm() << std::endl;

    exit(0);
    y.resize(x.rows());

    y = (A * x).eval();
#endif


    // Note: Matrix Vector Mult is exactly #nonzeros FMA instructions
    flop       = double(nnzr) * n * block_size * block_size;
    stat_eigen = measureObject(smv_its, [&]() { y = A * x; });
    //        stat_mkl   = measureObject(smv_its, [&]() { multMKL(mkl_A, mkl_A_desc, ex_x.data(), ex_y.data()); });
    stat_mkl = measureObject(smv_its, [&]() {
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, mkl_A, mkl_A_desc, ex_x.data(), 1, ex_y.data());
    });


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
