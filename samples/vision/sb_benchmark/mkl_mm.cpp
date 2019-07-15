/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


// for matrix-matrix
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 1


#include "sparse_block_benchmark.h"


template <int START, int END, typename T, int factor>
struct LauncherLoop
{
    void operator()()
    {
        {
            Saiga::MKL_Test<T, START, factor> t;
            t.sparseMatrixMatrix(10);
        }
        LauncherLoop<START + 1, END, T, factor> l;
        l();
    }
};

template <int END, typename T, int factor>
struct LauncherLoop<END, END, T, factor>
{
    void operator()() {}
};

#ifdef EIGEN_GPU_COMPILE_PHASE
#    error something went wrong
#endif

void bench_mm()
{
    LauncherLoop<2, 2 + 1, double, 2> l;
    l();
}
