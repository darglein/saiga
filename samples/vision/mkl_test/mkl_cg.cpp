/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


// for matrix-vector
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 1



#include "mkl_test.h"


template <int START, int END, typename T, int factor>
struct LauncherLoop
{
    void operator()()
    {
        {
            Saiga::MKL_Test<T, START, factor> t;
            t.sparseCG(200, 5);
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


void bench_cg()
{
    LauncherLoop<2, 2 + 1, double, 32> l;
    l();
}
