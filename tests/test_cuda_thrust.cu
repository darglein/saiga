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
#include "saiga/cuda/thrust_helper.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"

namespace Saiga
{
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

TEST(CudaThrust, UploadDownload)
{
    thrust::host_vector<int> H(4);
    H[0] = 38;
    H[1] = 20;
    H[2] = 42;
    H[3] = 5;

    thrust::device_vector<int> D = H;


    thrust::host_vector<int> H2 = D;

    EXPECT_EQ(H, H2);

    H[2] = 1924;
    D[2] = 1924;
    H2   = D;
    EXPECT_EQ(H, H2);
}

TEST(CudaThrust, Sort)
{
    thrust::host_vector<int> H(4);
    H[0] = 38;
    H[1] = 20;
    H[2] = 42;
    H[3] = 5;

    thrust::device_vector<int> D = H;

    thrust::sort(H.begin(), H.end());
    thrust::sort(D.begin(), D.end());


    thrust::host_vector<int> H2 = D;
    EXPECT_EQ(H, H2);
}


TEST(CudaThrust, CustomSort)
{
    thrust::host_vector<MySortStruct> H(4);
    H[0] = {1, 2.0f};
    H[1] = {2, 1.0f};
    H[2] = {3, 573.0f};
    H[3] = {4, -934.0f};

    thrust::device_vector<MySortStruct> D = H;


    thrust::sort(H.begin(), H.end());
    thrust::sort(D.begin(), D.end());

    thrust::host_vector<MySortStruct> H2 = D;
    EXPECT_EQ(H, H2);
}

TEST(CudaThrust, Maximum)
{
    thrust::host_vector<MySortStruct> H(4);
    H[0] = {1, 2.0f};
    H[1] = {2, 1.0f};
    H[2] = {3, 573.0f};
    H[3] = {4, -934.0f};

    thrust::device_vector<MySortStruct> D = H;

    auto max           = thrust::max_element(D.begin(), D.end());
    MySortStruct maxel = *max;

    EXPECT_EQ(maxel.key, 573.0f);
}

TEST(CudaThrust, Reduce)
{
    thrust::host_vector<MySortStruct> H(4);
    H[0] = {1, 2.0f};
    H[1] = {2, 1.0f};
    H[2] = {3, 573.0f};
    H[3] = {4, -934.0f};

    thrust::device_vector<MySortStruct> D = H;

    auto sum = thrust::reduce(D.begin(), D.end(), MySortStruct(0, 0), ReduceMySortStructOp());

    CUDA_SYNC_CHECK_ERROR();
    EXPECT_EQ(sum.value, 10);
    EXPECT_EQ(sum.key, -358.0f);
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
