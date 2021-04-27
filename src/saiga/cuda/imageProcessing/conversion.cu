/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/imageProcessing/imageProcessing.h"
#include "saiga/core/math/math.h"


namespace Saiga
{
namespace CUDA
{
template <int BLOCK_W, int BLOCK_H, int ROWS_PER_THREAD = 1>
__global__ static void d_convertRGBtoRGBA(ImageView<uchar3> src, ImageView<uchar4> dst, int h, unsigned char alpha)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    int x        = blockIdx.x * BLOCK_W + tx;
    int y        = blockIdx.y * BLOCK_H + ty;
    if (x >= dst.width) return;
#pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; ++i, y += h)
    {
        if (y < dst.height)
        {
            uchar3 v3 = src(y, x);
            uchar4 v4;
            v4.x      = v3.x;
            v4.y      = v3.y;
            v4.z      = v3.z;
            v4.w      = alpha;
            dst(y, x) = v4;
        }
    }
}


void convertRGBtoRGBA(ImageView<uchar3> src, ImageView<uchar4> dst, unsigned char alpha)
{
    SAIGA_ASSERT(src.width == dst.width && src.height == dst.height);
    const int ROWS_PER_THREAD = 4;
    const int BLOCK_W         = 128;
    const int BLOCK_H         = 1;
    int w                     = dst.width;
    int h                     = iDivUp(dst.height, ROWS_PER_THREAD);
    dim3 blocks(iDivUp(w, BLOCK_W), iDivUp(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);
    d_convertRGBtoRGBA<BLOCK_W, BLOCK_H, ROWS_PER_THREAD><<<blocks, threads>>>(src, dst, h, alpha);
}


template <int BLOCK_W, int BLOCK_H, int ROWS_PER_THREAD = 1>
__global__ static void d_convertRGBAtoRGB(ImageView<uchar4> src, ImageView<uchar3> dst, int h)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    int x        = blockIdx.x * BLOCK_W + tx;
    int y        = blockIdx.y * BLOCK_H + ty;
    if (x >= dst.width) return;
#pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; ++i, y += h)
    {
        if (y < dst.height)
        {
            uchar4 v4 = src(y, x);
            uchar3 v3;
            v3.x      = v4.x;
            v3.y      = v4.y;
            v3.z      = v4.z;
            dst(y, x) = v3;
        }
    }
}


void convertRGBAtoRGB(ImageView<uchar4> src, ImageView<uchar3> dst)
{
    SAIGA_ASSERT(src.width == dst.width && src.height == dst.height);
    const int ROWS_PER_THREAD = 4;
    const int BLOCK_W         = 128;
    const int BLOCK_H         = 1;
    int w                     = dst.width;
    int h                     = iDivUp(dst.height, ROWS_PER_THREAD);
    dim3 blocks(iDivUp(w, BLOCK_W), iDivUp(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);
    d_convertRGBAtoRGB<BLOCK_W, BLOCK_H, ROWS_PER_THREAD><<<blocks, threads>>>(src, dst, h);
}



template <int BLOCK_W, int BLOCK_H, int ROWS_PER_THREAD = 1>
__global__ static void d_convertRGBAtoGrayscale(ImageView<uchar4> src, ImageView<float> dst, int h)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    int x = blockIdx.x * BLOCK_W + tx;
    int y = blockIdx.y * BLOCK_H + ty;

    if (x >= dst.width) return;

    //    const vec3 conv(0.2126f / 255.0f,0.7152f / 255.0f,0.0722f / 255.0f);
    const vec3 conv(0.2126f, 0.7152f, 0.0722f);

#pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; ++i, y += h)
    {
        if (y < dst.height)
        {
            uchar4 u  = src(y, x);
            vec3 uv   = vec3(u.x, u.y, u.z);
            float v   = dot(uv, conv);
            dst(y, x) = v;
        }
    }
}

void convertRGBAtoGrayscale(ImageView<uchar4> src, ImageView<float> dst)
{
    SAIGA_ASSERT(src.width == dst.width && src.height == dst.height);
    const int ROWS_PER_THREAD = 4;
    const int BLOCK_W         = 128;
    const int BLOCK_H         = 1;
    int w                     = dst.width;
    int h                     = iDivUp(dst.height, ROWS_PER_THREAD);
    dim3 blocks(iDivUp(w, BLOCK_W), iDivUp(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);
    d_convertRGBAtoGrayscale<BLOCK_W, BLOCK_H, ROWS_PER_THREAD><<<blocks, threads>>>(src, dst, h);
}


template <int BLOCK_W, int BLOCK_H, int ROWS_PER_THREAD = 1>
__global__ static void d_convertBGRtoRGBA(ImageView<uchar3> src, ImageView<uchar4> dst, int h, unsigned char alpha)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    int x        = blockIdx.x * BLOCK_W + tx;
    int y        = blockIdx.y * BLOCK_H + ty;
    if (x >= dst.width) return;
#pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; ++i, y += h)
    {
        if (y < dst.height)
        {
            uchar3 v3 = src(y, x);
            uchar4 v4;
            v4.x      = v3.z;
            v4.y      = v3.y;
            v4.z      = v3.x;
            v4.w      = alpha;
            dst(y, x) = v4;
        }
    }
}


void convertBGRtoRGBA(ImageView<uchar3> src, ImageView<uchar4> dst, unsigned char alpha)
{
    SAIGA_ASSERT(src.width == dst.width && src.height == dst.height);
    const int ROWS_PER_THREAD = 4;
    const int BLOCK_W         = 128;
    const int BLOCK_H         = 1;
    int w                     = dst.width;
    int h                     = iDivUp(dst.height, ROWS_PER_THREAD);
    dim3 blocks(iDivUp(w, BLOCK_W), iDivUp(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);
    d_convertBGRtoRGBA<BLOCK_W, BLOCK_H, ROWS_PER_THREAD><<<blocks, threads>>>(src, dst, h, alpha);
}


template <int BLOCK_W, int BLOCK_H, int ROWS_PER_THREAD = 1>
__global__ static void d_convertRGBAtoBGR(ImageView<uchar4> src, ImageView<uchar3> dst, int h)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    int x        = blockIdx.x * BLOCK_W + tx;
    int y        = blockIdx.y * BLOCK_H + ty;
    if (x >= dst.width) return;
#pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; ++i, y += h)
    {
        if (y < dst.height)
        {
            uchar4 v4 = src(y, x);
            uchar3 v3;
            v3.x      = v4.z;
            v3.y      = v4.y;
            v3.z      = v4.x;
            dst(y, x) = v3;
        }
    }
}


void convertRGBAtoBGR(ImageView<uchar4> src, ImageView<uchar3> dst)
{
    SAIGA_ASSERT(src.width == dst.width && src.height == dst.height);
    const int ROWS_PER_THREAD = 4;
    const int BLOCK_W         = 128;
    const int BLOCK_H         = 1;
    int w                     = dst.width;
    int h                     = iDivUp(dst.height, ROWS_PER_THREAD);
    dim3 blocks(iDivUp(w, BLOCK_W), iDivUp(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);
    d_convertRGBAtoBGR<BLOCK_W, BLOCK_H, ROWS_PER_THREAD><<<blocks, threads>>>(src, dst, h);
}

}  // namespace CUDA
}  // namespace Saiga
