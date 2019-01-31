/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/util/assert.h"

#include "mkl/mkl.h"

#include <math.h>
#include <vector>

/**
 * Only a few simple wrappers for mkl calls
 */
inline void mklcg(const sparse_matrix_t A, struct matrix_descr descr, const double* x, double* rhs, double tol_error,
                  int iters, int n2, int block_size)
{
    int n = n2 * block_size;

    using T = double;
    std::vector<T> z(n);
    std::vector<T> tmp(n);
    std::vector<T> residual(n);

    double tol = tol_error;

    // residual = rhs - A * x;
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1, A, descr, x, 0, residual.data());
    vdAdd(n, rhs, residual.data(), residual.data());


    double rhsNorm2  = cblas_dnrm2(n, rhs, 1);
    double threshold = tol * tol * rhsNorm2;

    double residualNorm2 = cblas_dnrm2(n, residual.data(), 1);
    if (residualNorm2 < threshold)
    {
        iters     = 0;
        tol_error = sqrt(residualNorm2 / rhsNorm2);
        return;
    }

    std::vector<T> p(n);
    p = residual;

    double absNew = cblas_ddot(n, residual.data(), 1, p.data(), 1);
    int i         = 0;

    cout << "absnew " << absNew << endl;
    //    for (int i = 0; i < n; ++i) cout << x[i] << " ";
    //    cout << endl;

    //    for (int i = 0; i < n; ++i) cout << rhs[i] << " ";
    //    cout << endl;

    //    for (auto s : residual) cout << s << " ";
    //    cout << endl;
}
