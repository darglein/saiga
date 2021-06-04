/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Note: the intial source code ist taken from opencv.
// Opencv license
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/imageProcessing/imageProcessing.h"

namespace Saiga
{
namespace CUDA
{
//todo maybe change
static __constant__ float d_Kernel[SAIGA_MAX_KERNEL_SIZE];


template <int KSIZE>
__global__ void linearColumnFilter(ImageView<float> src, ImageView<float> dst, const int anchor)
{
    const int BLOCK_DIM_X     = 16;
    const int BLOCK_DIM_Y     = 16;
    const int PATCH_PER_BLOCK = 4;
    const int HALO_SIZE       = KSIZE <= 16 ? 1 : 2;


    using sum_t = float;
    using T     = float;

    __shared__ sum_t smem[(PATCH_PER_BLOCK + 2 * HALO_SIZE) * BLOCK_DIM_Y][BLOCK_DIM_X];

    const int x = blockIdx.x * BLOCK_DIM_X + threadIdx.x;

    if (x >= src.cols) return;

    const T* src_col = src.rowPtr(0) + x;

    const int yStart = blockIdx.y * (BLOCK_DIM_Y * PATCH_PER_BLOCK) + threadIdx.y;

    if (blockIdx.y > 0)
    {
        // Upper halo
#pragma unroll
        for (int j = 0; j < HALO_SIZE; ++j)
            smem[threadIdx.y + j * BLOCK_DIM_Y][threadIdx.x] = src(yStart - (HALO_SIZE - j) * BLOCK_DIM_Y, x);
    }
    else
    {
        // Upper halo
#pragma unroll
        for (int j = 0; j < HALO_SIZE; ++j)
            //            smem[threadIdx.y + j * BLOCK_DIM_Y][threadIdx.x] = brd.at_low(yStart - (HALO_SIZE - j) *
            //            BLOCK_DIM_Y, src_col, src.pitchBytes);
            smem[threadIdx.y + j * BLOCK_DIM_Y][threadIdx.x] = src(max(yStart - (HALO_SIZE - j) * BLOCK_DIM_Y, 0), x);
    }

    if (blockIdx.y + 2 < gridDim.y)
    {
        // Main data
#pragma unroll
        for (int j = 0; j < PATCH_PER_BLOCK; ++j)
            smem[threadIdx.y + HALO_SIZE * BLOCK_DIM_Y + j * BLOCK_DIM_Y][threadIdx.x] =
                src(yStart + j * BLOCK_DIM_Y, x);

            // Lower halo
#pragma unroll
        for (int j = 0; j < HALO_SIZE; ++j)
            smem[threadIdx.y + (PATCH_PER_BLOCK + HALO_SIZE) * BLOCK_DIM_Y + j * BLOCK_DIM_Y][threadIdx.x] =
                src(yStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_Y, x);
    }
    else
    {
        // Main data
#pragma unroll
        for (int j = 0; j < PATCH_PER_BLOCK; ++j)
            //            smem[threadIdx.y + HALO_SIZE * BLOCK_DIM_Y + j * BLOCK_DIM_Y][threadIdx.x] =
            //            brd.at_high(yStart + j * BLOCK_DIM_Y, src_col, src.pitchBytes);
            smem[threadIdx.y + HALO_SIZE * BLOCK_DIM_Y + j * BLOCK_DIM_Y][threadIdx.x] =
                src(min(yStart + j * BLOCK_DIM_Y, src.height - 1), x);

            // Lower halo
#pragma unroll
        for (int j = 0; j < HALO_SIZE; ++j)
            //            smem[threadIdx.y + (PATCH_PER_BLOCK + HALO_SIZE) * BLOCK_DIM_Y + j * BLOCK_DIM_Y][threadIdx.x]
            //            = brd.at_high(yStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_Y, src_col, src.pitchBytes);
            smem[threadIdx.y + (PATCH_PER_BLOCK + HALO_SIZE) * BLOCK_DIM_Y + j * BLOCK_DIM_Y][threadIdx.x] =
                src(min(yStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_Y, src.height - 1), x);
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < PATCH_PER_BLOCK; ++j)
    {
        const int y = yStart + j * BLOCK_DIM_Y;

        if (y < src.rows)
        {
            sum_t sum = 0;

#pragma unroll
            for (int k = 0; k < KSIZE; ++k)
                sum = sum + smem[threadIdx.y + HALO_SIZE * BLOCK_DIM_Y + j * BLOCK_DIM_Y - anchor + k][threadIdx.x] *
                                d_Kernel[k];

            dst(y, x) = sum;
        }
    }
}

template <typename T, int RADIUS>
static void convolveCol(ImageView<float> src, ImageView<float> dst)
{
    int BLOCK_DIM_X     = 16;
    int BLOCK_DIM_Y     = 16;
    int PATCH_PER_BLOCK = 4;

    const dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    const dim3 grid(iDivUp(src.cols, BLOCK_DIM_X), iDivUp(src.rows, BLOCK_DIM_Y * PATCH_PER_BLOCK));

    const int ksize = RADIUS * 2 + 1;
    int anchor      = ksize >> 1;
    linearColumnFilter<ksize><<<grid, block>>>(src, dst, anchor);
}

void convolveCol(ImageView<float> src, ImageView<float> dst, Saiga::ArrayView<float> kernel, int radius)
{
    SAIGA_ASSERT(kernel.size() > 0 && kernel.size() <= SAIGA_MAX_KERNEL_SIZE);
    CHECK_CUDA_ERROR(
        cudaMemcpyToSymbol(d_Kernel, kernel.data(), kernel.size() * sizeof(float), 0, cudaMemcpyDeviceToDevice));

    switch (radius)
    {
        case 1:
            convolveCol<float, 1>(src, dst);
            break;
        case 2:
            convolveCol<float, 2>(src, dst);
            break;
        case 3:
            convolveCol<float, 3>(src, dst);
            break;
        case 4:
            convolveCol<float, 4>(src, dst);
            break;
        case 5:
            convolveCol<float, 5>(src, dst);
            break;
        case 6:
            convolveCol<float, 6>(src, dst);
            break;
        case 7:
            convolveCol<float, 7>(src, dst);
            break;
        case 8:
            convolveCol<float, 8>(src, dst);
            break;
        case 9:
            convolveCol<float, 9>(src, dst);
            break;
        case 10:
            convolveCol<float, 10>(src, dst);
            break;
        case 11:
            convolveCol<float, 11>(src, dst);
            break;
        case 12:
            convolveCol<float, 12>(src, dst);
            break;
        case 13:
            convolveCol<float, 13>(src, dst);
            break;
        case 14:
            convolveCol<float, 14>(src, dst);
            break;
        case 15:
            convolveCol<float, 15>(src, dst);
            break;
        case 16:
            convolveCol<float, 16>(src, dst);
            break;
        case 17:
            convolveCol<float, 17>(src, dst);
            break;
        case 18:
            convolveCol<float, 18>(src, dst);
            break;
        case 19:
            convolveCol<float, 19>(src, dst);
            break;
        case 20:
            convolveCol<float, 20>(src, dst);
            break;
        case 21:
            convolveCol<float, 21>(src, dst);
            break;
        case 22:
            convolveCol<float, 22>(src, dst);
            break;
        case 23:
            convolveCol<float, 23>(src, dst);
            break;
        case 24:
            convolveCol<float, 24>(src, dst);
            break;
        default:
            SAIGA_ASSERT(0);
    }
}

}  // namespace CUDA
}  // namespace Saiga
