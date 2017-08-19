#pragma once

#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/imageProcessing/imageView.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>


namespace Saiga {
namespace CUDA {


#define MAX_RADIUS 10

__constant__ static float d_Kernel[MAX_RADIUS*2+1];


template<typename T, int RADIUS, int BLOCK_W, int BLOCK_H>
__global__
void singlePassConvolve2(ImageView<T> src, ImageView<T> dst)
{
    //for radius = 4: elements = (32+8) * (16+8) = 960 = 3840
    __shared__ T buffer[BLOCK_H + 2*RADIUS][BLOCK_W + 2*RADIUS];
    //for radius = 4: elements = (32+8) * (16) = 640 = 2560
    __shared__ T buffer2[BLOCK_H][BLOCK_W + 2*RADIUS];
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


inline
void copyConvolutionKernel(Saiga::array_view<float> kernel){
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_Kernel, kernel.data(), kernel.size()*sizeof(float)));
}

template<typename T, int RADIUS>
inline
void gaussianBlur(ImageView<T> src, ImageView<T> dst, float sigma){
    const int ELEMENTS = RADIUS * 2 + 1;
    float kernel[ELEMENTS];
    float kernelSum = 0.0f;
    float ivar2 = 1.0f/(2.0f*sigma*sigma);
    for (int j=-RADIUS;j<=RADIUS;j++) {
        kernel[j+RADIUS] = (float)expf(-(double)j*j*ivar2);
        kernelSum += kernel[j+RADIUS];
    }
    for (int j=-RADIUS;j<=RADIUS;j++)
        kernel[j+RADIUS] /= kernelSum;

    CUDA::copyConvolutionKernel(Saiga::array_view<float>(kernel,RADIUS*2+1));
    CUDA::convolve<float,RADIUS>(src,dst);
}

}
}
