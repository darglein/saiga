/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

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

    const int vec_mult = true;
    const int mat_mult = false;
    const int cg_mult  = false;

    if (vec_mult) bench_mv();
    if (mat_mult) bench_mm();
    if (cg_mult) bench_cg();
    return 0;
}
