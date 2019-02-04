/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/util/assert.h"
#include "saiga/vision/recursiveMatrices/RecursiveMatrices.h"

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


template <typename MatrixType, typename T>
inline void createBlockMKLFromEigen(const MatrixType& A, sparse_matrix_t* mklA, std::vector<MKL_INT>& rows_start,
                                    std::vector<MKL_INT>& rows_end, std::vector<MKL_INT>& col_indx,
                                    std::vector<T>& values, int n, int m, int block_size)
{
    rows_start.clear();
    rows_end.clear();
    col_indx.clear();
    values.clear();

    for (int k = 0; k < A.outerSize(); ++k)
    {
        rows_start.push_back(A.outerIndexPtr()[k]);
        rows_end.push_back(A.outerIndexPtr()[k + 1]);
        for (typename MatrixType::InnerIterator it(A, k); it; ++it)
        {
            col_indx.push_back(it.index());

            for (auto i = 0; i < block_size; ++i)
            {
                for (auto j = 0; j < block_size; ++j)
                {
                    auto block = it.valueRef();
                    values.push_back(block.get()(i, j));
                }
            }
        }
    }

    mkl_sparse_d_create_bsr(mklA, SPARSE_INDEX_BASE_ZERO, SPARSE_LAYOUT_ROW_MAJOR, n, m, block_size, rows_start.data(),
                            rows_end.data(), col_indx.data(), values.data());
}
