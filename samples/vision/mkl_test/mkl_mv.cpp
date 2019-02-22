/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// for matrix-vector
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 128
//#define SAIGA_GENEREAL_MV

#include "mkl_test.h"



template <int START, int END, typename T, int factor>
struct LauncherLoop
{
    void operator()()
    {
        {
            Saiga::MKL_Test<T, START, factor> t;
            t.sparseMatrixVector(50);
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


void bench_mv()
{
    {
        LauncherLoop<32, 32 + 1, double, 32> l;
        l();
    }
    {
        LauncherLoop<43, 43 + 1, double, 64> l;
        l();
    }
}
