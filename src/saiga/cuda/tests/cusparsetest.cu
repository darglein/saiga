/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/cusparseHelper.h"
#include "saiga/cuda/tests/test.h"
#include "saiga/cuda/tests/test_helper.h"
#include "saiga/cuda/thread_info.h"
#include "saiga/core/time/timer.h"

namespace Saiga
{
namespace CUDA
{
#ifdef SAIGA_USE_CUSPARSE

void testCuSparse()
{
    //    0  3 0  0 0
    //    22 0 0  0 17
    //    7  5 0  1 0
    //    0  0 0  0 0
    //    0  0 14 0 8

    // in column major
    std::vector<double> denseMatrix = {0, 22, 7, 0, 0, 3, 0, 5, 0, 0, 0, 0, 0, 0, 14, 0, 0, 1, 0, 0, 0, 17, 0, 0, 8};

    std::vector<double> denseVector = {1, 2, 3, 4, 5};

    // result of the matrix vector product
    std::vector<double> ytarget = {6, 107, 21, 0, 82};


    std::vector<double> values = {22, 7, 3, 5, 14, 1, 17, 8};

    std::vector<int> rowIndx = {1, 2, 0, 2, 4, 2, 1, 4};

    std::vector<int> colPtr = {0, 2, 4, 5, 6, 8};

    thrust::device_vector<double> d_values = values;
    thrust::device_vector<int> d_rowIndx   = rowIndx;
    thrust::device_vector<int> d_colPtr    = colPtr;
    thrust::device_vector<double> d_x      = denseVector;
    thrust::device_vector<double> d_y(denseVector.size(), 0);

    cusparseMatDescr_t mat;
    cusparseCreateMatDescr(&mat);

    double alpha      = 1;
    const double beta = 2;

    cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, 5, 5, values.size(), &alpha, mat,
                   thrust::raw_pointer_cast(d_values.data()), thrust::raw_pointer_cast(d_colPtr.data()),
                   thrust::raw_pointer_cast(d_rowIndx.data()), thrust::raw_pointer_cast(d_x.data()), &beta,
                   thrust::raw_pointer_cast(d_y.data()));

    thrust::host_vector<double> y = d_y;
    //    for(double d : y){
    //        std::cout << d << " ";
    //    }
    //    std::cout << std::endl;


    SAIGA_ASSERT(y == ytarget);



    std::cout << "cuSPARSE test: SUCCESS!" << std::endl;
}


/* Matrix size */
#    define N (275)

/* Host implementation of a simple version of sgemm */
static void simple_sgemm(int n, float alpha, const float* A, const float* B, float beta, float* C)
{
    int i;
    int j;
    int k;

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            float prod = 0;

            for (k = 0; k < n; ++k)
            {
                prod += A[k * n + i] * B[j * n + k];
            }

            C[j * n + i] = alpha * prod + beta * C[j * n + i];
        }
    }
}


void testCuBLAS()
{
    cublasStatus_t status;
    float* h_A;
    float* h_B;
    float* h_C;
    float* h_C_ref;
    float* d_A  = 0;
    float* d_B  = 0;
    float* d_C  = 0;
    float alpha = 1.0f;
    float beta  = 0.0f;
    int n2      = N * N;
    int i;
    float error_norm;
    float ref_norm;
    float diff;



    /* Allocate host memory for the matrices */
    h_A = (float*)malloc(n2 * sizeof(h_A[0]));

    if (h_A == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (A)\n");
        SAIGA_ASSERT(0);
    }

    h_B = (float*)malloc(n2 * sizeof(h_B[0]));

    if (h_B == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (B)\n");
        SAIGA_ASSERT(0);
    }

    h_C = (float*)malloc(n2 * sizeof(h_C[0]));

    if (h_C == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (C)\n");
        SAIGA_ASSERT(0);
    }

    /* Fill the matrices with test data */
    for (i = 0; i < n2; i++)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = rand() / (float)RAND_MAX;
    }

    /* Allocate device memory for the matrices */
    if (cudaMalloc((void**)&d_A, n2 * sizeof(d_A[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
        SAIGA_ASSERT(0);
    }

    if (cudaMalloc((void**)&d_B, n2 * sizeof(d_B[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
        SAIGA_ASSERT(0);
    }

    if (cudaMalloc((void**)&d_C, n2 * sizeof(d_C[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
        SAIGA_ASSERT(0);
    }

    /* Initialize the device matrices with the host matrices */
    status = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write A)\n");
        SAIGA_ASSERT(0);
    }

    status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write B)\n");
        SAIGA_ASSERT(0);
    }

    status = cublasSetVector(n2, sizeof(h_C[0]), h_C, 1, d_C, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write C)\n");
        SAIGA_ASSERT(0);
    }

    /* Performs operation using plain C code */
    simple_sgemm(N, alpha, h_A, h_B, beta, h_C);
    h_C_ref = h_C;

    /* Performs operation using cublas */
    status = cublasSgemm(cublashandle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! kernel execution error.\n");
        SAIGA_ASSERT(0);
    }

    /* Allocate host memory for reading back the result from device memory */
    h_C = (float*)malloc(n2 * sizeof(h_C[0]));

    if (h_C == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (C)\n");
        SAIGA_ASSERT(0);
    }

    /* Read the result back */
    status = cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (read C)\n");
        SAIGA_ASSERT(0);
    }

    /* Check result against reference */
    error_norm = 0;
    ref_norm   = 0;

    for (i = 0; i < n2; ++i)
    {
        diff = h_C_ref[i] - h_C[i];
        error_norm += diff * diff;
        ref_norm += h_C_ref[i] * h_C_ref[i];
    }

    error_norm = (float)sqrt((double)error_norm);
    ref_norm   = (float)sqrt((double)ref_norm);

    if (fabs(ref_norm) < 1e-7)
    {
        fprintf(stderr, "!!!! reference norm is 0\n");
        SAIGA_ASSERT(0);
    }

    /* Memory clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    if (cudaFree(d_A) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (A)\n");
        SAIGA_ASSERT(0);
    }

    if (cudaFree(d_B) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (B)\n");
        SAIGA_ASSERT(0);
    }

    if (cudaFree(d_C) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (C)\n");
        SAIGA_ASSERT(0);
    }


    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        SAIGA_ASSERT(0);
    }

    if (error_norm / ref_norm < 1e-6f)
    {
        std::cout << "cuBLAS test: SUCCESS!" << std::endl;
    }
    else
    {
        printf("simpleCUBLAS test failed.\n");
        SAIGA_ASSERT(0);
    }
}

#endif

}  // namespace CUDA
}  // namespace Saiga
