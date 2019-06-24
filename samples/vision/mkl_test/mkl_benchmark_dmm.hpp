/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once

#include "mkl_test.h"

namespace Saiga
{
template <typename T, int level1, int level2, int level3, int factor>
class DenseMKLTest
{
    const int n = 1024;



    // most inner block
    using L1Block         = Eigen::Matrix<T, level1, level1, Eigen::ColMajor>;
    using L2Block         = Eigen::Matrix<MatrixScalar<L1Block>, level2, level2, Eigen::ColMajor>;
    using L3Block         = Eigen::Matrix<MatrixScalar<L2Block>, level3, level3, Eigen::ColMajor>;
    using RecursiveMatrix = Eigen::Matrix<MatrixScalar<L1Block>, -1, -1, Eigen::ColMajor>;


   public:
    DenseMKLTest()
    {
        std::cout << "DenseMKLTest" << std::endl;
        SAIGA_ASSERT(n % (level1 * level2 * level3) == 0);

        int m = n;
        int k = n;
        //        eA    = Eigen::Matrix<T, -1, -1>(m, n);
        //        eB    = Eigen::Matrix<T, -1, -1>(m, n);
        //        eC    = Eigen::Matrix<T, -1, -1>(m, n);
        eA.resize(m, n);
        eB.resize(m, n);
        eC.resize(m, n);
        eC2 = eC;



        //        eA.setRandom
        eA.setRandom();
        eB.setRandom();
        eC.setRandom();

        eC = eA * eB;
        C  = A * B;

        //        dgemm()
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, eA.data(), k, eB.data(), m, 0, eC2.data(),
                    n);

        std::cout << eC << std::endl << std::endl;
        std::cout << eC2 << std::endl;
    }
    ~DenseMKLTest()
    {
        //        mkl_sparse_destroy(mkl_A);
        //        mkl_sparse_destroy(mkl_B);
    }

    void denseMatrixMatrix(int smm_its) {}

   private:
    // Eigen data structures
    Eigen::Matrix<T, -1, -1> eA, eB, eC, eC2;
    RecursiveMatrix A, B, C;
    //    BlockVector x;
    //    BlockVector y;
    //    BlockMatrix A, B, C;

    // MKL data structures
    //    Eigen::Matrix<T, -1, 1> ex_x;
    //    Eigen::Matrix<T, -1, 1> ex_y;
    //    sparse_matrix_t mkl_A, mkl_B, mkl_C;
    //    matrix_descr mkl_A_desc, mkl_B_desc;
    //    matrix_descr
};


}  // namespace Saiga
