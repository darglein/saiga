/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/imageProcessing/imageProcessing.h"

namespace Saiga
{
namespace CUDA
{
template <typename T, int BLOCK_W, int BLOCK_H, int ROWS_PER_THREAD = 1>
__global__ static void d_scaleDown2EveryOther(ImageView<T> src, ImageView<T> dst, int h)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * BLOCK_W + tx;
    int y = blockIdx.y * BLOCK_H + ty;


    if (x >= dst.width) return;

#pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; ++i, y += h)
    {
        if (y < dst.height)
        {
            dst(y, x) = src(y * 2, x * 2);
        }
    }
}

template <typename T>
void scaleDown2EveryOther(ImageView<T> src, ImageView<T> dst)
{
    SAIGA_ASSERT(src.width / 2 == dst.width && src.height / 2 == dst.height);

    const int ROWS_PER_THREAD = 2;
    const int BLOCK_W         = 128;
    const int BLOCK_H         = 1;

    int w = dst.width;
    int h = iDivUp(dst.height, ROWS_PER_THREAD);
    dim3 blocks(iDivUp(w, BLOCK_W), iDivUp(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);
    d_scaleDown2EveryOther<T, BLOCK_W, BLOCK_H, ROWS_PER_THREAD><<<blocks, threads>>>(src, dst, h);
}

// create code for common types
template SAIGA_CUDA_API void scaleDown2EveryOther<float>(ImageView<float> src, ImageView<float> dst);
template SAIGA_CUDA_API void scaleDown2EveryOther<uchar3>(ImageView<uchar3> src, ImageView<uchar3> dst);
template SAIGA_CUDA_API void scaleDown2EveryOther<uchar4>(ImageView<uchar4> src, ImageView<uchar4> dst);

}  // namespace CUDA
}  // namespace Saiga
