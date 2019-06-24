/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


// for matrix-matrix
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 1


#include "mkl_benchmark_dmm.hpp"


void bench_dmm()
{
    Saiga::DenseMKLTest<double, 2, 2, 2, 1> t;
    t.denseMatrixMatrix(100);
}
