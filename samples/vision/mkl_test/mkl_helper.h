/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/util/assert.h"

#include "mkl/mkl.h"

/**
 * Only a few simple wrappers for mkl calls
 */
inline void multMKL(const sparse_matrix_t A, struct matrix_descr descr, const double* x, double* y)
{
    auto ret = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, A, descr, x, 0, y);
    SAIGA_ASSERT(ret == SPARSE_STATUS_SUCCESS);
}

inline void multMKL(const sparse_matrix_t A, struct matrix_descr descr, const float* x, float* y)
{
    auto ret = mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, A, descr, x, 0, y);
    SAIGA_ASSERT(ret == SPARSE_STATUS_SUCCESS);
}

inline void multMKLMM(const sparse_matrix_t A, const sparse_matrix_t B, sparse_matrix_t* C)
{
    auto ret = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, A, B, C);
    SAIGA_ASSERT(ret == SPARSE_STATUS_SUCCESS);
}

inline auto createMKL(sparse_matrix_t* A, MKL_INT* rows_start, MKL_INT* rows_end, MKL_INT* col_indx, double* values,
                      int n, int m, int block_size)
{
    return mkl_sparse_d_create_bsr(A, SPARSE_INDEX_BASE_ZERO, SPARSE_LAYOUT_ROW_MAJOR, n, m, block_size, rows_start,
                                   rows_end, col_indx, values);
}

inline auto createMKL(sparse_matrix_t* A, MKL_INT* rows_start, MKL_INT* rows_end, MKL_INT* col_indx, float* values,
                      int n, int m, int block_size)
{
    return mkl_sparse_s_create_bsr(A, SPARSE_INDEX_BASE_ZERO, SPARSE_LAYOUT_ROW_MAJOR, n, m, block_size, rows_start,
                                   rows_end, col_indx, values);
}
