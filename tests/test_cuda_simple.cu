/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/cuda/imageProcessing/NppiHelper.h"
//
#include "saiga/core/framework/framework.h"
#include "saiga/core/image/all.h"
#include "saiga/cuda/CudaInfo.h"
#include "saiga/cuda/imageProcessing/image.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"

namespace Saiga
{
TEST(CudaSimple, Memcpy)
{
    int N = 10000;

    std::vector<int> h_data(N);

    for (auto& i : h_data)
    {
        i = Random::uniformInt(0, 100000);
    }

    size_t size = sizeof(int) * N;

    int* d_data;
    cudaMalloc((void**)&d_data, size);
    int* d_data2;
    cudaMalloc((void**)&d_data2, size);

    cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data2, d_data, size, cudaMemcpyDeviceToDevice);

    std::vector<int> h_data2(N);
    cudaMemcpy(h_data2.data(), d_data2, size, cudaMemcpyDeviceToHost);

    EXPECT_EQ(h_data, h_data2);
}

__global__ static void addFive(int* data, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N) return;
    data[tid] = data[tid] + 5;
}


TEST(CudaSimple, AddFive)
{
    int N = 10000;
    std::vector<int> h_data(N);
    for (auto& i : h_data)
    {
        i = Random::uniformInt(0, 100000);
    }
    size_t size = sizeof(int) * N;
    int* d_data;
    cudaMalloc((void**)&d_data, size);
    cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice);


    addFive<<<iDivUp(N, 128), 128>>>(d_data, N);

    std::vector<int> h_data2(N);
    cudaMemcpy(h_data2.data(), d_data, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
    {
        EXPECT_EQ(h_data2[i], h_data[i] + 5);
    }
}


}  // namespace Saiga

int main()
{
    Saiga::CUDA::initCUDA();
    Saiga::CUDA::printCUDAInfo();

    Saiga::initSaigaSampleNoWindow();
    testing::InitGoogleTest();

    return RUN_ALL_TESTS();
}
