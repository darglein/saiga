/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/imageProcessing/convolution.h"
#include "saiga/cuda/tests/test_helper.h"
#include "saiga/cuda/tests/test.h"
#include "saiga/cuda/thread_info.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/time/timer.h"

using std::cout;
using std::endl;

namespace Saiga {
namespace CUDA {


__constant__ float d_Kernel[MAX_RADIUS*2+1];


template<typename T, int RADIUS, int BLOCK_W, int BLOCK_H>
__global__
void singlePassConvolve2(ImageView<T> src, ImageView<T> dst)
{
    //for radius = 4: elements = (32+8) * (16+8) = 960 = 3840
    __shared__ volatile T buffer[BLOCK_H + 2*RADIUS][BLOCK_W + 2*RADIUS];
    //for radius = 4: elements = (32+8) * (16) = 640 = 2560
    __shared__ volatile T buffer2[BLOCK_H][BLOCK_W + 2*RADIUS];
    //total s mem per block = 6400
    //with 512 threads per block smem per sm: 25600 -> 100% occ


    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int t = tx + ty * BLOCK_W;
    const int xp = blockIdx.x*BLOCK_W + tx;
    const int yp = blockIdx.y*BLOCK_H + ty;


    int blockStartX = blockIdx.x*BLOCK_W - RADIUS;
    int blockStartY = blockIdx.y*BLOCK_H - RADIUS;

    const int blockSizeX = BLOCK_W + 2*RADIUS;
    const int blockSizeY = BLOCK_H + 2*RADIUS;

    //fill buffer
    for(int i = t; i < blockSizeX * blockSizeY; i += (BLOCK_W*BLOCK_H)){
        int x = i % blockSizeX;
        int y = i / blockSizeX;
        int gx = x + blockStartX;
        int gy = y + blockStartY;
        src.clampToEdge(gx,gy);
        buffer[y][x] = src(gx,gy);
    }

    __syncthreads();


    T *kernel = d_Kernel;

    for(int i = t; i < blockSizeX * BLOCK_H; i += (BLOCK_W*BLOCK_H)){
        int x = i % blockSizeX;
        int y = i / blockSizeX;
        T sum = 0;
#pragma unroll
        for (int j=-RADIUS;j<=RADIUS;j++){
            int kernelIndex = j + RADIUS;
            sum += buffer[y + RADIUS + j][x] * kernel[kernelIndex];
        }
        buffer2[y][x] = sum;
    }

    __syncthreads();

    T sum = 0;

#pragma unroll
    for (int j=-RADIUS;j<=RADIUS;j++){
        int kernelIndex = j + RADIUS;
        sum += buffer2[ty][tx + RADIUS + j] * kernel[kernelIndex];
    }

    if(dst.inImage(xp,yp))
        dst(xp,yp) = sum;
}


template<typename T, int RADIUS>
inline
void convolve(ImageView<T> src, ImageView<T> dst){
    int w = src.width;
    int h = src.height;

    const int BLOCK_W = 32;
    const int BLOCK_H = 16;

    dim3 blocks(Saiga::CUDA::getBlockCount(w, BLOCK_W), Saiga::CUDA::getBlockCount(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);

    singlePassConvolve2<T,RADIUS,BLOCK_W,BLOCK_H> <<<blocks, threads>>>(src,dst);
}

void convolve(ImageView<float> src, ImageView<float> dst, int radius){
    switch (radius){
    case 1: CUDA::convolve<float,1>(src,dst); break;
    case 2: CUDA::convolve<float,2>(src,dst); break;
    case 3: CUDA::convolve<float,3>(src,dst); break;
    case 4: CUDA::convolve<float,4>(src,dst); break;
    case 5: CUDA::convolve<float,5>(src,dst); break;
    case 6: CUDA::convolve<float,6>(src,dst); break;
    case 7: CUDA::convolve<float,7>(src,dst); break;
    case 8: CUDA::convolve<float,8>(src,dst); break;
    case 9: CUDA::convolve<float,9>(src,dst); break;
    case 10: CUDA::convolve<float,10>(src,dst); break;
    }

}


void copyConvolutionKernel(Saiga::array_view<float> kernel){
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_Kernel, kernel.data(), kernel.size()*sizeof(float)));
}


}
}
