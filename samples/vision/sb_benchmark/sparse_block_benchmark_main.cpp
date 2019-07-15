/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "sparse_block_benchmark.h"

#include "saiga/vision/Eigen_Compile_Checker.h"


void bench_mm();
void bench_mv();


int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;
    Saiga::EigenHelper::checkEigenCompabitilty<2765>();

    // Type of benchmark

    const int vec_mult = false;
    const int mat_mult = true;



    if (vec_mult) bench_mv();
    if (mat_mult) bench_mm();

    return 0;
}
