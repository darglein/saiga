/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


// for matrix-matrix
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 16


#include "mkl_test.h"


template <int START, int END, typename T, int factor>
struct LauncherLoop
{
    void operator()()
    {
        {
            Saiga::MKL_Test<T, START, factor> t;
            t.sparseMatrixMatrix(100);
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


void bench_mm()
{
    LauncherLoop<2, 64 + 1, double, 8> l;
    l();
}
