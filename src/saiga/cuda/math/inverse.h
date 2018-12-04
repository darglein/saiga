#pragma once

#include "saiga/config.h"

namespace Saiga
{
namespace CUDA
{
enum MatrixIndexOp
{
    Half,
    Full
};


template <typename T, int N, MatrixIndexOp op>
struct MatrixIndex
{
};

/**
 * Full Matrix:
 *
 * | a b c |
 * | d e f |
 * | g h i |
 *
 * Stored as: [a,b,c,d,e,f,g,h,i]
 */
template <typename T, int N>
struct MatrixIndex<T, N, Full>
{
    T* A;

    HD inline MatrixIndex(T* A) : A(A) {}

    HD inline T& operator()(int row, int col) { return A[row * N + col]; }
};

/**
 * Half (lower) Matrix:
 *
 * | a * * |
 * | b c * |
 * | d e f |
 *
 * Stored as: [a,b,c,d,e,f]
 */
template <typename T, int N>
struct MatrixIndex<T, N, Half>
{
    T* A;

    HD inline MatrixIndex(T* A) : A(A) {}

    HD inline T& operator()(int row, int col) { return A[(row * (row + 1) / 2) + col]; }
};

template <typename T, int N, MatrixIndexOp op = Full>
HD inline void choleskyKernel(const T* A, T* C)
{
    MatrixIndex<const T, N, op> M(A);
    MatrixIndex<T, N, op> R(C);

    // for all rows
#pragma unroll
    for (int i = 0; i < N; i++)
    {
        // for all cols until diagonal
#pragma unroll
        for (int j = 0; j < N; j++)
            if (j <= i)
            {
                double s = 0;
                // dot product of row i with row j,
                // but only until col j
                // this requires the top left block to be computed
#pragma unroll
                for (int k = 0; k < N; k++)
                    if (k < j)
                    {
                        s += R(i, k) * R(j, k);
                    }
                s       = M(i, j) - s;
                R(i, j) = (i == j) ? sqrt(s) : (1.0 / R(j, j) * (s));
            }
    }
}


template <typename T, int N>
HD inline void choleskyKernel2(T* C)
{
#pragma unroll
    for (int i = 0; i < N; i++)
    {
#pragma unroll
        for (int j = 0; j < N; j++)
            if (j <= i)
            {
                double s = 0;
#pragma unroll
                for (int k = 0; k < N; k++)
                    if (k < j)
                    {
                        s += C[i * N + k] * C[j * N + k];
                    }
                s            = C[i * N + j] - s;
                C[i * N + j] = (i == j) ? sqrt(s) : (1.0 / C[j * N + j] * (s));
            }
    }
}

template <typename T, int N, MatrixIndexOp op = Full>
HD inline void inverseTriangularKernel(T* A)
{
    MatrixIndex<T, N, op> M(A);

    for (int i = 0; i < N; ++i)
    {
        // invert the diagonal element
        M(i, i) = 1.0 / M(i, i);
    }

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
            if (j < i)
            {
                T sum = 0;
                // dot product of row i with col j
                for (int k = 0; k < N; ++k)
                    if (k >= j && k < i)
                    {
                        sum += M(i, k) * M(k, j);
                    }
                // divide by diagonal element of this row
                M(i, j) = -M(i, i) * sum;
            }
    }
}



template <typename T, int N, MatrixIndexOp op = Full>
HD inline void multLTL(T* A)
{
    MatrixIndex<T, N, op> M(A);

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j <= N; ++j)
            if (j <= i)
            {
                // dot product of col i with col j
                double sum = 0;
                // Note: we can start at i here because all values above the diagonal are 0 and i >= j
                for (int k = 0; k < N; ++k)
                    if (k >= i)
                    {
                        sum += M(k, i) * M(k, j);
                    }
                M(i, j) = sum;
                if (op == Full) M(j, i) = sum;
            }
    }
}


template <typename T, int N, MatrixIndexOp op = Full>
HD inline void invertSymmetric(T* A)
{
    choleskyKernel<T, N, op>(A, A);
    inverseTriangularKernel<T, N, op>(A);
    multLTL<T, N, op>(A);
}

template <typename T, int N, MatrixIndexOp op = Full>
HD inline void invertSymmetric(const T* source, T* res)
{
    choleskyKernel<T, N, op>(source, res);
    inverseTriangularKernel<T, N, op>(res);
    multLTL<T, N, op>(res);
}



template <typename T>
inline HD void inverse3x3(const T* A, T* C)
{
    const int N = 3;
    MatrixIndex<const T, N, MatrixIndexOp::Full> m(A);
    MatrixIndex<T, N, MatrixIndexOp::Full> minv(C);

    // https://stackoverflow.com/questions/983999/simple-3x3-matrix-inverse-code-c

    // computes the inverse of a matrix m
    T det = m(0, 0) * (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2)) - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
            m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));

    T invdet = T(1) / det;

    minv(0, 0) = (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2)) * invdet;
    minv(0, 1) = (m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) * invdet;
    minv(0, 2) = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) * invdet;
    minv(1, 0) = (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) * invdet;
    minv(1, 1) = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) * invdet;
    minv(1, 2) = (m(1, 0) * m(0, 2) - m(0, 0) * m(1, 2)) * invdet;
    minv(2, 0) = (m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1)) * invdet;
    minv(2, 1) = (m(2, 0) * m(0, 1) - m(0, 0) * m(2, 1)) * invdet;
    minv(2, 2) = (m(0, 0) * m(1, 1) - m(1, 0) * m(0, 1)) * invdet;
}


template <typename T>
inline HD void inverse3x3Symmetric(const T* A, T* C)
{
    const int N = 3;
    MatrixIndex<const T, N, MatrixIndexOp::Half> m(A);
    MatrixIndex<T, N, MatrixIndexOp::Half> minv(C);

    // https://stackoverflow.com/questions/983999/simple-3x3-matrix-inverse-code-c

    // computes the inverse of a matrix m
    T det = m(0, 0) * (m(1, 1) * m(2, 2) - m(2, 1) * m(2, 1)) - m(1, 0) * (m(1, 0) * m(2, 2) - m(2, 1) * m(2, 0)) +
            m(2, 0) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));

    T invdet = T(1) / det;

    minv(0, 0) = (m(1, 1) * m(2, 2) - m(2, 1) * m(2, 1)) * invdet;
    minv(1, 0) = (m(2, 1) * m(2, 0) - m(1, 0) * m(2, 2)) * invdet;
    minv(1, 1) = (m(0, 0) * m(2, 2) - m(2, 0) * m(2, 0)) * invdet;
    minv(2, 0) = (m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1)) * invdet;
    minv(2, 1) = (m(2, 0) * m(1, 0) - m(0, 0) * m(2, 1)) * invdet;
    minv(2, 2) = (m(0, 0) * m(1, 1) - m(1, 0) * m(1, 0)) * invdet;
}

}  // namespace CUDA
}  // namespace Saiga
