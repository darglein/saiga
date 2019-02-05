/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// for matrix-vector
//#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 128

// for matrix-matrix
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 16

#include "mkl_test.h"

void bench_mm();
void bench_mv();
void bench_cg();

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;
    Saiga::EigenHelper::checkEigenCompabitilty<2765>();

    // Type of benchmark

    const int mat_mult = true;
    const int vec_mult = true;
    const int cg_mult  = true;

    if (vec_mult) bench_mv();
    if (mat_mult) bench_mm();
    if (cg_mult) bench_cg();
    return 0;
}
