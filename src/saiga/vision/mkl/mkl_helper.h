/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/assert.h"

#include "EigenRecursive/All.h"


#if !defined(SAIGA_USE_MKL)
#    error Saiga was compiled without MKL!
#endif

#include "mkl.h"


template <typename T>
struct MKLBlockMatrix
{
    int n, m;
    int blockSize;

    std::vector<T> values;
    std::vector<MKL_INT> col_index;
    std::vector<MKL_INT> row_start;
    std::vector<MKL_INT> row_end;

    sparse_matrix_t A;
    matrix_descr desc;
    bool allocated = false;

    ~MKLBlockMatrix()
    {
        if (allocated)
        {
            mkl_sparse_destroy(A);
        }
    }

    void setBlock(int offset, const T* data)
    {
        T* dst = &values[offset * blockSize * blockSize];

        for (int i = 0; i < blockSize * blockSize; ++i)
        {
            dst[i] = data[i];
        }
    }

    void resize(int _n, int _m, int nnz, int _blockSize)
    {
        n         = _n;
        m         = _m;
        blockSize = _blockSize;

        values.resize(nnz * blockSize * blockSize);
        col_index.resize(nnz);

        row_start.resize(n);
        row_end.resize(n);
    }

    void allocate()
    {
        mkl_sparse_d_create_bsr(&A, SPARSE_INDEX_BASE_ZERO, SPARSE_LAYOUT_ROW_MAJOR, n, m, blockSize, row_start.data(),
                                row_end.data(), col_index.data(), values.data());
        desc.type = SPARSE_MATRIX_TYPE_GENERAL;
        allocated = true;
    }
    template <typename BlockType, int options>
    void create(const Eigen::DiagonalMatrix<BlockType, options>& D, int block_size)
    {
        int n = D.size();
        resize(n, n, n, block_size);

        for (int i = 0; i < n; ++i)
        {
            const double* ptr = D.diagonal()(i).get().data();
            setBlock(i, ptr);
            col_index[i] = i;
            row_start[i] = i;
            row_end[i]   = i + 1;
        }


        allocate();

        desc.type = SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL;
        desc.diag = SPARSE_DIAG_NON_UNIT;
    }

    operator sparse_matrix_t() { return A; }
    operator matrix_descr() { return desc; }
};


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
    auto ret = mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, A, descr, x, 1, y);
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


template <typename BlockType, int options>
inline void createBlockMKLFromEigen(Eigen::SparseMatrix<BlockType, options>& A, sparse_matrix_t* mklA,
                                    matrix_descr* desc, int block_size)
{
    static_assert(options == Eigen::RowMajor, "matrix must be row major");
    using T = typename Eigen::Recursive::ScalarType<BlockType>::Type;
    int n   = A.rows();
    int m   = A.cols();
    mkl_sparse_d_create_bsr(mklA, SPARSE_INDEX_BASE_ZERO, SPARSE_LAYOUT_ROW_MAJOR, n, m, block_size, A.outerIndexPtr(),
                            A.outerIndexPtr() + 1, A.innerIndexPtr(), (T*)A.valuePtr());
    desc->type = SPARSE_MATRIX_TYPE_GENERAL;
}

template <typename BlockType, int options>
inline void createEigenFromBlockMKL(Eigen::SparseMatrix<BlockType, options>& A, sparse_matrix_t* mklA, int block_size)
{
    static_assert(options == Eigen::RowMajor, "matrix must be row major");
    using Block = typename BlockType::M;

    MKL_INT rows, cols, block_size2;
    MKL_INT *rows_start, *rows_end, *col_indx;
    double* values;
    sparse_index_base_t indexing;
    sparse_layout_t block_layout;
    mkl_sparse_d_export_bsr(*mklA, &indexing, &block_layout, &rows, &cols, &block_size2, &rows_start, &rows_end,
                            &col_indx, &values);
    SAIGA_ASSERT(block_layout == SPARSE_LAYOUT_ROW_MAJOR);
    SAIGA_ASSERT(block_size2 == block_size);
    //    std::cout << "mlk matrix data: " << indexing << " " << block_layout << " " << rows << " " << cols << " "
    //              << block_size << " " << rows_end[rows - 1] << std::endl
    //              << std::endl;

    A.setZero();
    A.resize(rows, cols);
    A.reserve(rows_end[rows - 1]);

    for (int r = 0; r < rows; ++r)
    {
        *(A.outerIndexPtr() + r) = rows_start[r];
        for (int s = rows_start[r]; s < rows_end[r]; ++s)
        {
            int c = col_indx[s];
            Eigen::Map<Block> bm(values + s * block_size * block_size);
            *(A.innerIndexPtr() + s) = c;
            *(A.valuePtr() + s)      = bm;
        }
    }
    *(A.outerIndexPtr() + rows) = rows_end[rows - 1];
}

template <typename MatrixType, typename T>
inline void createBlockMKLFromEigen2(const MatrixType& A, sparse_matrix_t* mklA, std::vector<MKL_INT>& rows_start,
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
