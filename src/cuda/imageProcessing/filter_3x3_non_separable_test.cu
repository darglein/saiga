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

__constant__ float d_Kernel[3][3];


template<unsigned int TILE_W, unsigned int TILE_H>
__global__ static
void d_convolve3x3(ImageView<float> src, ImageView<float> dst
                   )
{
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int t = ty * TILE_W + tx;

    const unsigned int x_tile = blockIdx.x * TILE_W;
    const unsigned int y_tile = blockIdx.y * TILE_H;

    const unsigned int x = x_tile + tx;
    const unsigned int y = y_tile + ty;

    float sum = 0;
    for(int dy = -1; dy <= 1; ++dy){
        for(int dx = -1; dx <= 1; ++dx){
            int gx = x + dx;
            int gy = y + dy;
            src.clampToEdge(gx,gy);
            sum += src(gx,gy);
        }
    }
    dst(x,y) = sum;
}


template<unsigned int TILE_W, unsigned int TILE_H>
__global__ static
void d_convolve3x3Shared(ImageView<float> src, ImageView<float> dst)
{
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int t = ty * TILE_W + tx;

    const unsigned int x_tile = blockIdx.x * TILE_W;
    const unsigned int y_tile = blockIdx.y * TILE_H;
    const unsigned int blockStartX = x_tile - 1;
    const unsigned int blockStartY = y_tile - 1;


    const unsigned int TILE_SIZE =  TILE_H * TILE_W;
    const unsigned int TILE_SIZE_WITH_BORDER = (TILE_H+2) * (TILE_W+2);
    __shared__ float sbuffer[TILE_H + 2][TILE_W + 2];


    for(int i = t; i < TILE_SIZE_WITH_BORDER; i += TILE_SIZE){
        int x = i % (TILE_W+2);
        int y = i / (TILE_W+2);
        int gx = x + blockStartX;
        int gy = y + blockStartY;
        src.clampToEdge(gx,gy);
        sbuffer[y][x] = src(gx,gy);
    }

    __syncthreads();


    float sum = 0;
#if 1
    for(int dy = -1; dy <= 1; ++dy){
        for(int dx = -1; dx <= 1; ++dx){
            int x = tx + 1 + dx;
            int y = ty + 1 + dy;
            sum += sbuffer[y][x];
        }
    }
#endif

    const unsigned int x = x_tile + tx;
    const unsigned int y = y_tile + ty;
    dst(x,y) = sum;
}



template<unsigned int TILE_W, unsigned int TILE_H>
__global__ static
void d_convolve3x3Shared2(ImageView<float> src, ImageView<float> dst)
{
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int t = ty * TILE_W + tx;

    const unsigned int x_tile = blockIdx.x * TILE_W;
    const unsigned int y_tile = blockIdx.y * TILE_H;
    const unsigned int blockStartX = x_tile - 1;
    const unsigned int blockStartY = y_tile - 1;


    const unsigned int TILE_SIZE =  TILE_H * TILE_W;
    const unsigned int TILE_SIZE_WITH_BORDER = (TILE_H+2) * (TILE_W+2);
    __shared__ float sbuffer[TILE_H + 2][TILE_W + 2];


    //copy main data


    __syncthreads();


    float sum = 0;
#if 1
    for(int dy = -1; dy <= 1; ++dy){
        for(int dx = -1; dx <= 1; ++dx){
            int x = tx + 1 + dx;
            int y = ty + 1 + dy;
            sum += sbuffer[y][x];
        }
    }
#endif

    const unsigned int x = x_tile + tx;
    const unsigned int y = y_tile + ty;
    dst(x,y) = sum;
}


template<unsigned int TILE_W, unsigned int TILE_H>
__global__ static
void d_copySharedSync(ImageView<float> src, ImageView<float> dst)
{
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int t = ty * TILE_W + tx;
    const unsigned int x_tile = blockIdx.x * TILE_W;
    const unsigned int y_tile = blockIdx.y * TILE_H;
    const unsigned int x = x_tile + tx;
    const unsigned int y = y_tile + ty;

    __shared__ float sbuffer[TILE_H][TILE_W];

    sbuffer[ty][tx]  = src(x,y);
    __syncthreads();
    dst(x,y) = sbuffer[ty][tx];
}

void convolutionTest3x3(){
    CUDA_SYNC_CHECK_ERROR();

    int h = 2048;
    int w = h * 2;

    size_t N = w * h;
    size_t readWrites = N * 2 * sizeof(float);

    Saiga::CUDA::PerformanceTestHelper pth("filter 3x3 separable", readWrites);

    thrust::device_vector<float> src(N,0.1);
    thrust::device_vector<float> dest(N,0.1);
    thrust::device_vector<float> tmp(N,0.1);

    thrust::host_vector<float> h_src = src;
    thrust::host_vector<float> h_dest = dest;
    thrust::host_vector<float> h_tmp = dest;
    thrust::host_vector<float> h_ref = dest;

    ImageView<float> imgSrc(w,h,thrust::raw_pointer_cast(src.data()));
    ImageView<float> imgDst(w,h,thrust::raw_pointer_cast(dest.data()));
    ImageView<float> imgTmp(w,h,thrust::raw_pointer_cast(tmp.data()));


    ImageView<float> h_imgSrc(w,h,thrust::raw_pointer_cast(h_src.data()));
    ImageView<float> h_imgDst(w,h,thrust::raw_pointer_cast(h_dest.data()));
    ImageView<float> h_imgTmp(w,h,thrust::raw_pointer_cast(h_tmp.data()));

    thrust::host_vector<float> h_kernel(9,1);

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_Kernel, h_kernel.data(), h_kernel.size()*sizeof(float),0,cudaMemcpyHostToDevice));

    {
        for(int y = 0; y < h; ++y){
            for(int x = 0; x < w; ++x){
                h_imgSrc(x,y) = (rand()%3) - 1;
            }
        }
        src = h_src;
    }

    {
        for(int y = 0; y < h; ++y){
            for(int x = 0; x < w; ++x){
                float sum = 0;
                for(int dy = -1; dy <= 1; ++dy){
                    for(int dx = -1; dx <= 1; ++dx){
                        int gx = x + dx;
                        int gy = y + dy;
                        h_imgSrc.clampToEdge(gx,gy);
                        sum += h_imgSrc(gx,gy);
                    }
                }
                h_imgDst(x,y) = sum;
            }
        }
        h_ref = h_dest;
    }

    {
        float time;
        {
            Saiga::CUDA::CudaScopedTimer t(time);
            const int TILE_W = 128;
            const int TILE_H = 1;
            dim3 blocks(
                        Saiga::iDivUp(w, TILE_W),
                        Saiga::iDivUp(h, TILE_H),
                        1
                        );
            dim3 threads(TILE_W,TILE_H);
            d_convolve3x3<TILE_W,TILE_H><<<blocks,threads>>>(imgSrc,imgDst);
        }
        pth.addMeassurement("d_convolve3x3", time);
        h_dest = dest;
        SAIGA_ASSERT(h_dest == h_ref);
    }

    {
        float time;
        {
            Saiga::CUDA::CudaScopedTimer t(time);
            const int TILE_W = 32;
            const int TILE_H = 16;
            dim3 blocks(
                        Saiga::iDivUp(w, TILE_W),
                        Saiga::iDivUp(h, TILE_H),
                        1
                        );
            dim3 threads(TILE_W,TILE_H);
            d_convolve3x3Shared<TILE_W,TILE_H><<<blocks,threads>>>(imgSrc,imgDst);
        }
        pth.addMeassurement("d_convolve3x3Shared", time);
        h_dest = dest;
        SAIGA_ASSERT(h_dest == h_ref);
    }
    {
        float time;
        {
            Saiga::CUDA::CudaScopedTimer t(time);
            const int TILE_W = 32;
            const int TILE_H = 16;
            dim3 blocks(
                        Saiga::iDivUp(w, TILE_W),
                        Saiga::iDivUp(h, TILE_H),
                        1
                        );
            dim3 threads(TILE_W,TILE_H);
            d_convolve3x3Shared2<TILE_W,TILE_H><<<blocks,threads>>>(imgSrc,imgDst);
        }
        pth.addMeassurement("d_convolve3x3Shared2", time);
        h_dest = dest;
//        SAIGA_ASSERT(h_dest == h_ref);
    }
    {
        float time;
        {
            Saiga::CUDA::CudaScopedTimer t(time);
            const int TILE_W = 32;
            const int TILE_H = 16;
            dim3 blocks(
                        Saiga::iDivUp(w, TILE_W),
                        Saiga::iDivUp(h, TILE_H),
                        1
                        );
            dim3 threads(TILE_W,TILE_H);
            d_copySharedSync<TILE_W,TILE_H><<<blocks,threads>>>(imgSrc,imgDst);
        }
        pth.addMeassurement("d_copySharedSync", time);
        h_dest = dest;
//        SAIGA_ASSERT(h_dest == h_ref);
    }

    {
        float time;
        {
            Saiga::CUDA::CudaScopedTimer t(time);
            cudaMemcpy(thrust::raw_pointer_cast(dest.data()),thrust::raw_pointer_cast(src.data()),N * sizeof(int),cudaMemcpyDeviceToDevice);

        }
        pth.addMeassurement("cudaMemcpy", time);
    }
    CUDA_SYNC_CHECK_ERROR();

}

}
}
