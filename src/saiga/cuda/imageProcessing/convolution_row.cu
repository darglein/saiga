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
__global__ void linearRowFilter(ImageView<float> src, ImageView<float> dst, const int anchor)
{
    const int BLOCK_DIM_X     = 32;
    const int BLOCK_DIM_Y     = 8;
    const int PATCH_PER_BLOCK = 4;
    const int HALO_SIZE       = 1;


    using sum_t = float;

    __shared__ sum_t smem[BLOCK_DIM_Y][(PATCH_PER_BLOCK + 2 * HALO_SIZE) * BLOCK_DIM_X];

    const int y = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;

    if (y >= src.height) return;



    const float* src_row = src.rowPtr(y);

    const int xStart = blockIdx.x * (PATCH_PER_BLOCK * BLOCK_DIM_X) + threadIdx.x;

    if (blockIdx.x > 0)
    {
        // Load left halo
#pragma unroll
        for (int j = 0; j < HALO_SIZE; ++j)
            smem[threadIdx.y][threadIdx.x + j * BLOCK_DIM_X] = src_row[xStart - (HALO_SIZE - j) * BLOCK_DIM_X];
    }
    else
    {
        // Load left halo
#pragma unroll
        for (int j = 0; j < HALO_SIZE; ++j)
            //            smem[threadIdx.y][threadIdx.x + j * BLOCK_DIM_X] = brd.at_low(xStart - (HALO_SIZE - j) *
            //            BLOCK_DIM_X, src_row);
            smem[threadIdx.y][threadIdx.x + j * BLOCK_DIM_X] = src_row[max(xStart - (HALO_SIZE - j) * BLOCK_DIM_X, 0)];
    }

    if (blockIdx.x + 2 < gridDim.x)
    {
        // Load main data
#pragma unroll
        for (int j = 0; j < PATCH_PER_BLOCK; ++j)
            smem[threadIdx.y][threadIdx.x + HALO_SIZE * BLOCK_DIM_X + j * BLOCK_DIM_X] =
                src_row[xStart + j * BLOCK_DIM_X];

            // Load right halo
#pragma unroll
        for (int j = 0; j < HALO_SIZE; ++j)
            smem[threadIdx.y][threadIdx.x + (PATCH_PER_BLOCK + HALO_SIZE) * BLOCK_DIM_X + j * BLOCK_DIM_X] =
                src_row[xStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_X];
    }
    else
    {
        // Load main data
#pragma unroll
        for (int j = 0; j < PATCH_PER_BLOCK; ++j)
            //            smem[threadIdx.y][threadIdx.x + HALO_SIZE * BLOCK_DIM_X + j * BLOCK_DIM_X] =
            //            brd.at_high(xStart + j * BLOCK_DIM_X, src_row);
            smem[threadIdx.y][threadIdx.x + HALO_SIZE * BLOCK_DIM_X + j * BLOCK_DIM_X] =
                src_row[min(xStart + j * BLOCK_DIM_X, src.width - 1)];

            // Load right halo
#pragma unroll
        for (int j = 0; j < HALO_SIZE; ++j)
            //            smem[threadIdx.y][threadIdx.x + (PATCH_PER_BLOCK + HALO_SIZE) * BLOCK_DIM_X + j * BLOCK_DIM_X]
            //            = brd.at_high(xStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_X, src_row);
            smem[threadIdx.y][threadIdx.x + (PATCH_PER_BLOCK + HALO_SIZE) * BLOCK_DIM_X + j * BLOCK_DIM_X] =
                src_row[min(xStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_X, src.width - 1)];
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < PATCH_PER_BLOCK; ++j)
    {
        const int x = xStart + j * BLOCK_DIM_X;

        if (x < src.width)
        {
            sum_t sum = 0;

#pragma unroll
            for (int k = 0; k < KSIZE; ++k)
                sum = sum + smem[threadIdx.y][threadIdx.x + HALO_SIZE * BLOCK_DIM_X + j * BLOCK_DIM_X - anchor + k] *
                                d_Kernel[k];

            dst(y, x) = sum;
        }
    }
}

template <typename T, int RADIUS>
static void convolveRow(ImageView<float> src, ImageView<float> dst)
{
    const int BLOCK_W         = 32;
    const int BLOCK_H         = 8;
    const int PATCH_PER_BLOCK = 4;

    const dim3 block(BLOCK_W, BLOCK_H);
    const dim3 grid(iDivUp(src.width, BLOCK_W * PATCH_PER_BLOCK), iDivUp(src.height, BLOCK_H));

    const int ksize = RADIUS * 2 + 1;
    int anchor      = ksize >> 1;
    linearRowFilter<ksize><<<grid, block>>>(src, dst, anchor);
}

void convolveRow(ImageView<float> src, ImageView<float> dst, Saiga::ArrayView<float> kernel, int radius)
{
    SAIGA_ASSERT(kernel.size() > 0 && kernel.size() <= SAIGA_MAX_KERNEL_SIZE);
    CHECK_CUDA_ERROR(
        cudaMemcpyToSymbol(d_Kernel, kernel.data(), kernel.size() * sizeof(float), 0, cudaMemcpyDeviceToDevice));

    switch (radius)
    {
        case 1:
            convolveRow<float, 1>(src, dst);
            break;
        case 2:
            convolveRow<float, 2>(src, dst);
            break;
        case 3:
            convolveRow<float, 3>(src, dst);
            break;
        case 4:
            convolveRow<float, 4>(src, dst);
            break;
        case 5:
            convolveRow<float, 5>(src, dst);
            break;
        case 6:
            convolveRow<float, 6>(src, dst);
            break;
        case 7:
            convolveRow<float, 7>(src, dst);
            break;
        case 8:
            convolveRow<float, 8>(src, dst);
            break;
        case 9:
            convolveRow<float, 9>(src, dst);
            break;
        case 10:
            convolveRow<float, 10>(src, dst);
            break;
        case 11:
            convolveRow<float, 11>(src, dst);
            break;
        case 12:
            convolveRow<float, 12>(src, dst);
            break;
        case 13:
            convolveRow<float, 13>(src, dst);
            break;
        case 14:
            convolveRow<float, 14>(src, dst);
            break;
        case 15:
            convolveRow<float, 15>(src, dst);
            break;
        case 16:
            convolveRow<float, 16>(src, dst);
            break;
        case 17:
            convolveRow<float, 17>(src, dst);
            break;
        case 18:
            convolveRow<float, 18>(src, dst);
            break;
        case 19:
            convolveRow<float, 19>(src, dst);
            break;
        case 20:
            convolveRow<float, 20>(src, dst);
            break;
        case 21:
            convolveRow<float, 21>(src, dst);
            break;
        case 22:
            convolveRow<float, 22>(src, dst);
            break;
        case 23:
            convolveRow<float, 23>(src, dst);
            break;
        case 24:
            convolveRow<float, 24>(src, dst);
            break;
        default:
            SAIGA_ASSERT(0);
    }
}

}  // namespace CUDA
}  // namespace Saiga
