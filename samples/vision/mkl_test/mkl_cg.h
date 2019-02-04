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
inline void mklcg(const sparse_matrix_t A, struct matrix_descr descr, const sparse_matrix_t P,
                  struct matrix_descr pdescr, double* x, double* rhs, double tol_error, int iters, int n2,
                  int block_size)
{
    using T = double;

    int n = n2 * block_size;

#if 0
    // Create them locally
    std::vector<T> v_z(n);
    std::vector<T> v_residual(n);
    std::vector<T> v_p(n);
#else
    // Use static variables so a repeated call with the same size doesn't allocate memory
    static thread_local std::vector<T> v_z;
    static thread_local std::vector<T> v_residual;
    static thread_local std::vector<T> v_p;
    v_z.resize(n);
    v_residual.resize(n);
    v_p.resize(n);
#endif

    // Extract pointers because all blas function uses them
    auto z        = v_z.data();
    auto residual = v_residual.data();
    auto p        = v_p.data();

    double tol   = tol_error;
    int maxIters = iters;

    // residual = rhs - A * x;
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1, A, descr, x, 0, residual);
    vdAdd(n, rhs, residual, residual);


    //    double rhsNorm2 = cblas_dnrm2(n, rhs, 1);
    //    rhsNorm2        = rhsNorm2 * rhsNorm2;
    double rhsNorm2 = cblas_ddot(n, rhs, 1, rhs, 1);
    if (rhsNorm2 == 0)
    {
        iters     = 0;
        tol_error = 0;
        return;
    }

    double threshold = tol * tol * rhsNorm2;

    //    double residualNorm2 = cblas_dnrm2(n, residual, 1);
    //    residualNorm2        = residualNorm2 * residualNorm2;
    double residualNorm2 = cblas_ddot(n, residual, 1, residual, 1);

    if (residualNorm2 < threshold)
    {
        iters     = 0;
        tol_error = sqrt(residualNorm2 / rhsNorm2);
        return;
    }



    // p = Precond * residual
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, P, pdescr, residual, 0, p);
    //    cblas_dcopy(n, residual, 1, p, 1);


    double absNew = cblas_ddot(n, residual, 1, p, 1);
    int i         = 0;
    while (i < maxIters)
    {
        // tmp = A * p (main bottleneck of CG)
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1, A, descr, p, 0, z);

        double dptmp = cblas_ddot(n, p, 1, z, 1);
        double alpha = absNew / dptmp;

        // x += alpha * p
        cblas_daxpby(n, alpha, p, 1, 1, x, 1);

        // residual -= alpha * tmp
        cblas_daxpby(n, -alpha, z, 1, 1, residual, 1);

        //        residualNorm2 = cblas_dnrm2(n, residual, 1);
        //        residualNorm2 = residualNorm2 * residualNorm2;
        residualNorm2 = cblas_ddot(n, residual, 1, residual, 1);
        //        cout << i << " " << residualNorm2 << endl;
        if (residualNorm2 < threshold) break;

        // z = Precond * residual
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, P, pdescr, residual, 0, z);

        double absOld = absNew;
        absNew        = cblas_ddot(n, residual, 1, z, 1);
        double beta   = absNew / absOld;

        // p = z + beta * p
        cblas_daxpby(n, 1, z, 1, beta, p, 1);

        i++;
    }
    tol_error = sqrt(residualNorm2 / rhsNorm2);
    iters     = i;
}
