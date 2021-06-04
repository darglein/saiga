/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/tests/test.h"
#include "saiga/cuda/tests/test_helper.h"
#include "saiga/cuda/thread_info.h"
#include "saiga/core/time/timer.h"

#include <thrust/extrema.h>
#include <thrust/sort.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace Saiga
{
namespace CUDA
{
__global__ static void addFive(float* g_idata, float* g_odata)
{
    g_odata[threadIdx.x] = g_idata[threadIdx.x] + 5;
}

void testCuda()
{
    CUDA_SYNC_CHECK_ERROR();
    unsigned int num_threads = 32;
    unsigned int mem_size    = sizeof(float) * num_threads;

    // allocate host memory
    float* h_idata = (float*)malloc(mem_size);

    // initalize the memory
    for (unsigned int i = 0; i < num_threads; ++i)
    {
        h_idata[i] = (float)i;
    }

    // allocate device memory
    float* d_idata;
    cudaMalloc((void**)&d_idata, mem_size);
    CUDA_SYNC_CHECK_ERROR();
    // copy host memory to device
    cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice);

    CUDA_SYNC_CHECK_ERROR();
    // allocate device memory for result
    float* d_odata;
    cudaMalloc((void**)&d_odata, mem_size);
    CUDA_SYNC_CHECK_ERROR();


    // execute the kernel
    addFive<<<1, num_threads>>>(d_idata, d_odata);
    CUDA_SYNC_CHECK_ERROR();

    // allocate mem for the result on host side
    float* h_odata = (float*)malloc(mem_size);
    // copy result from device to host
    cudaMemcpy(h_odata, d_odata, sizeof(float) * num_threads, cudaMemcpyDeviceToHost);
    CUDA_SYNC_CHECK_ERROR();
    bool result = true;
    for (unsigned int i = 0; i < num_threads; ++i)
    {
        if (h_odata[i] != i + 5) result = false;
    }

    if (result)
    {
        std::cout << "CUDA test: SUCCESS!" << std::endl;
    }
    else
    {
        std::cout << "CUDA test: FAILED!" << std::endl;
        SAIGA_ASSERT(0);
    }

    // cleanup memory
    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);

    CUDA_SYNC_CHECK_ERROR();
}

struct MySortStruct
{
    int value;
    float key;

    __host__ __device__ MySortStruct() {}
    __host__ __device__ MySortStruct(int v, float k) : value(v), key(k) {}
};

__host__ __device__ bool operator<(const MySortStruct& a, const MySortStruct& b)
{
    return a.key < b.key;
}

__host__ __device__ bool operator==(const MySortStruct& a, const MySortStruct& b)
{
    return a.key == b.key && a.value == b.value;
}

struct ReduceMySortStructOp
{
    __host__ __device__ MySortStruct operator()(const MySortStruct& a, const MySortStruct& b)
    {
        MySortStruct res;
        res.key   = a.key + b.key;
        res.value = a.value + b.value;
        return res;
    }
};

void testThrust()
{
    CUDA_SYNC_CHECK_ERROR();
    {
        // simple sort test
        thrust::host_vector<int> H(4);
        H[0] = 38;
        H[1] = 20;
        H[2] = 42;
        H[3] = 5;

        thrust::device_vector<int> D = H;


        thrust::sort(H.begin(), H.end());
        thrust::sort(D.begin(), D.end());
        CUDA_SYNC_CHECK_ERROR();
        SAIGA_ASSERT(H == D);
    }

    {
        // sort of custom struct test
        thrust::host_vector<MySortStruct> H(4);
        H[0] = {1, 2.0f};
        H[1] = {2, 1.0f};
        H[2] = {3, 573.0f};
        H[3] = {4, -934.0f};

        thrust::device_vector<MySortStruct> D = H;


        thrust::sort(H.begin(), H.end());
        thrust::sort(D.begin(), D.end());
        CUDA_SYNC_CHECK_ERROR();
        SAIGA_ASSERT(H == D);
    }

    {
        // find maximum test
        thrust::host_vector<MySortStruct> H(4);
        H[0] = {1, 2.0f};
        H[1] = {2, 1.0f};
        H[2] = {3, 573.0f};
        H[3] = {4, -934.0f};

        thrust::device_vector<MySortStruct> D = H;

        auto max           = thrust::max_element(D.begin(), D.end());
        MySortStruct maxel = *max;
        CUDA_SYNC_CHECK_ERROR();
        SAIGA_ASSERT(maxel.key == 573.0f);
    }


    {
        // reduce test
        thrust::host_vector<MySortStruct> H(4);
        H[0] = {1, 2.0f};
        H[1] = {2, 1.0f};
        H[2] = {3, 573.0f};
        H[3] = {4, -934.0f};

        thrust::device_vector<MySortStruct> D = H;

        auto sum = thrust::reduce(D.begin(), D.end(), MySortStruct(0, 0), ReduceMySortStructOp());

        CUDA_SYNC_CHECK_ERROR();
        SAIGA_ASSERT(sum.value == 10 && sum.key == -358.0f);
    }

    std::cout << "Thrust test: SUCCESS!" << std::endl;
    CUDA_SYNC_CHECK_ERROR();
}

}  // namespace CUDA
}  // namespace Saiga
