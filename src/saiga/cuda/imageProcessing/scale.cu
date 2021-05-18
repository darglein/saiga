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
template <int BLOCK_W, int BLOCK_H, int ROWS_PER_THREAD = 1>
__global__ static void d_fill(ImageView<float> img, int h, float value)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    int x = blockIdx.x * BLOCK_W + tx;
    int y = blockIdx.y * BLOCK_H + ty;

    if (x >= img.width) return;

    // process a fixed number of elements per thread to maximise instruction level parallelism
    for (int i = 0; i < ROWS_PER_THREAD; ++i, y += h)
    {
        if (y < img.height) img(y, x) = value;
    }
}

void fill(ImageView<float> img, float value)
{
    const int ROWS_PER_THREAD = 4;
    const int BLOCK_W         = 128;
    const int BLOCK_H         = 1;
    int w                     = img.width;
    int h                     = iDivUp(img.height, ROWS_PER_THREAD);
    dim3 blocks(iDivUp(w, BLOCK_W), iDivUp(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);
    d_fill<BLOCK_W, BLOCK_H, ROWS_PER_THREAD><<<blocks, threads>>>(img, h, value);
}



template <int BLOCK_W, int BLOCK_H, int ROWS_PER_THREAD = 1>
__global__ static void d_add(ImageView<float> img, int h, float value)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    int x = blockIdx.x * BLOCK_W + tx;
    int y = blockIdx.y * BLOCK_H + ty;

    if (x >= img.width) return;

    // process a fixed number of elements per thread to maximise instruction level parallelism
    for (int i = 0; i < ROWS_PER_THREAD; ++i, y += h)
    {
        if (y < img.height) img(y, x) += value;
    }
}

void add(ImageView<float> img, float value)
{
    const int ROWS_PER_THREAD = 4;
    const int BLOCK_W         = 128;
    const int BLOCK_H         = 1;
    int w                     = img.width;
    int h                     = iDivUp(img.height, ROWS_PER_THREAD);
    dim3 blocks(iDivUp(w, BLOCK_W), iDivUp(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);
    d_add<BLOCK_W, BLOCK_H, ROWS_PER_THREAD><<<blocks, threads>>>(img, h, value);
}



template <int BLOCK_W, int BLOCK_H, int ROWS_PER_THREAD = 1>
__global__ static void d_mult(ImageView<float> img, int h, float value)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    int x = blockIdx.x * BLOCK_W + tx;
    int y = blockIdx.y * BLOCK_H + ty;

    if (x >= img.width) return;

    // process a fixed number of elements per thread to maximise instruction level parallelism
    for (int i = 0; i < ROWS_PER_THREAD; ++i, y += h)
    {
        if (y < img.height) img(y, x) *= value;
    }
}

void mult(ImageView<float> img, float value)
{
    const int ROWS_PER_THREAD = 4;
    const int BLOCK_W         = 128;
    const int BLOCK_H         = 1;
    int w                     = img.width;
    int h                     = iDivUp(img.height, ROWS_PER_THREAD);
    dim3 blocks(iDivUp(w, BLOCK_W), iDivUp(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);
    d_mult<BLOCK_W, BLOCK_H, ROWS_PER_THREAD><<<blocks, threads>>>(img, h, value);
}


template <int BLOCK_W, int BLOCK_H, int ROWS_PER_THREAD = 1>
__global__ static void d_abs(ImageView<float> img, int h)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    int x = blockIdx.x * BLOCK_W + tx;
    int y = blockIdx.y * BLOCK_H + ty;

    if (x >= img.width) return;

    // process a fixed number of elements per thread to maximise instruction level parallelism
    for (int i = 0; i < ROWS_PER_THREAD; ++i, y += h)
    {
        if (y < img.height) img(y, x) = fabsf(img(y, x));
    }
}

void abs(ImageView<float> img)
{
    const int ROWS_PER_THREAD = 4;
    const int BLOCK_W         = 128;
    const int BLOCK_H         = 1;
    int w                     = img.width;
    int h                     = iDivUp(img.height, ROWS_PER_THREAD);
    dim3 blocks(iDivUp(w, BLOCK_W), iDivUp(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);
    d_abs<BLOCK_W, BLOCK_H, ROWS_PER_THREAD><<<blocks, threads>>>(img, h);
}



#define USE_HARDWARE_INTER

#ifdef USE_HARDWARE_INTER
static texture<float, cudaTextureType2D, cudaReadModeElementType> floatTex;
#endif

template <int BLOCK_W, int BLOCK_H, int ROWS_PER_THREAD = 1>
__global__ static void d_scaleUp2Linear(ImageView<float> src, ImageView<float> dst, int h, double scale_x,
                                        double scale_y)
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
#ifdef USE_HARDWARE_INTER
            // use hardware bil. interpolation
            float xf  = (float(x) + 0.5f) * scale_x;
            float yf  = (float(y) + 0.5f) * scale_y;
            dst(y, x) = tex2D(floatTex, xf, yf);
#else
            // software bil. interpolation
            float xf  = (float(x) + 0.5f) * scale_x - 0.5f;
            float yf  = (float(y) + 0.5f) * scale_y - 0.5f;
            dst(y, x) = src.inter(yf, xf);
#endif
        }
    }
}


void scaleUp2Linear(ImageView<float> src, ImageView<float> dst)
{
    SAIGA_ASSERT(src.width * 2 == dst.width && src.height * 2 == dst.height);

#ifdef USE_HARDWARE_INTER
    textureReference& floatTexRef = floatTex;
    cudaChannelFormatDesc desc    = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    size_t offset;
    SAIGA_ASSERT(src.pitchBytes % 256 == 0);
    CHECK_CUDA_ERROR(cudaBindTexture2D(&offset, &floatTexRef, src.data, &desc, src.width, src.height, src.pitchBytes));
    SAIGA_ASSERT(offset == 0);
    floatTexRef.addressMode[0] = cudaAddressModeClamp;
    floatTexRef.addressMode[1] = cudaAddressModeClamp;
    floatTexRef.filterMode     = cudaFilterModeLinear;
    floatTexRef.normalized     = false;
#endif



    double inv_scale_x = (double)dst.width / src.width;
    double inv_scale_y = (double)dst.height / src.height;
    double scale_x = 1. / inv_scale_x, scale_y = 1. / inv_scale_y;


    const int ROWS_PER_THREAD = 4;
    const int BLOCK_W         = 128;
    const int BLOCK_H         = 1;
    int w                     = dst.width;
    int h                     = iDivUp(dst.height, ROWS_PER_THREAD);
    dim3 blocks(iDivUp(w, BLOCK_W), iDivUp(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);
    d_scaleUp2Linear<BLOCK_W, BLOCK_H, ROWS_PER_THREAD><<<blocks, threads>>>(src, dst, h, scale_x, scale_y);
}

}  // namespace CUDA
}  // namespace Saiga
