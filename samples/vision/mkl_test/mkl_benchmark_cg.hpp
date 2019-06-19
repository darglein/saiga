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
inline void MKL_Test<T, block_size, factor>::sparseCG(int scg_its, int cg_inner_its)
{
    std::ofstream strm("eigen_mkl_cg.csv", std::ostream::app);
    // ============= Benchmark =============

    T tol = 1e-50;

    std::cout << "Running Block Sparse CG Benchmark..." << std::endl;
    std::cout << "Number of Runs: " << scg_its << std::endl;
    std::cout << "Number of inner CG iterations: " << cg_inner_its << std::endl;


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
    sparse_matrix_t mkl_P;
    //    createBlockMKLFromEigen2(Pm, &mkl_P, prow_start, prow_end, pcol_index, pvalues, n, m, block_size);

    matrix_descr mkl_P_desc;
    createBlockMKLFromEigen(Pm, &mkl_P, &mkl_P_desc, block_size);
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

    mkl_sparse_destroy(mkl_P);
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
