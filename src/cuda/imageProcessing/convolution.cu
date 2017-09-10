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
__global__ static
void singlePassConvolve(ImageView<T> src, ImageView<T> dst)
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

    singlePassConvolve<T,RADIUS,BLOCK_W,BLOCK_H> <<<blocks, threads>>>(src,dst);
}


template<typename T, int RADIUS, unsigned int BLOCK_W, unsigned int BLOCK_H, unsigned int Y_ELEMENTS>
__global__ static
void singlePassConvolve2(ImageView<T> src, ImageView<T> dst)
{
    const unsigned BLOCK_H2 = BLOCK_H * Y_ELEMENTS;

    //for radius = 4: elements = (32+8) * (16+8) = 960 = 3840
    __shared__ T buffer[BLOCK_H2 + 2*RADIUS][BLOCK_W + 2*RADIUS];
    //for radius = 4: elements = (32+8) * (16) = 640 = 2560
    __shared__ T buffer2[BLOCK_H2][BLOCK_W + 2*RADIUS];
    //total s mem per block = 6400
    //with 512 threads per block smem per sm: 25600 -> 100% occ


    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int t = tx + ty * BLOCK_W;
    int xp = blockIdx.x*BLOCK_W + tx;
    int yp = blockIdx.y*BLOCK_H2 + ty;


    int blockStartX = blockIdx.x*BLOCK_W - RADIUS;
    int blockStartY = blockIdx.y*BLOCK_H2 - RADIUS;

    const int blockSizeX = BLOCK_W + 2*RADIUS;
    const int blockSizeY = BLOCK_H2 + 2*RADIUS;

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

    for(int i = t; i < blockSizeX * BLOCK_H2; i += (BLOCK_W*BLOCK_H)){
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

    for(int i =0; i < Y_ELEMENTS; ++i){
        T sum = 0;
#pragma unroll
        for (int j=-RADIUS;j<=RADIUS;j++){
            int kernelIndex = j + RADIUS;
            sum += buffer2[ty][tx + RADIUS + j] * kernel[kernelIndex];
        }

        if(dst.inImage(xp,yp))
            dst(xp,yp) = sum;
        yp += BLOCK_H;
        ty += BLOCK_H;
    }
}

template<typename T, int RADIUS>
inline
void convolve2(ImageView<T> src, ImageView<T> dst){
    int w = src.width;
    int h = src.height;

    const int BLOCK_W = 32;
    const int BLOCK_H = 16;
    const int Y_ELEMENTS = 2;
    dim3 blocks(
                Saiga::iDivUp(w, BLOCK_W ),
                Saiga::iDivUp(h, BLOCK_H * Y_ELEMENTS),
                1
                );

    //    dim3 blocks(Saiga::CUDA::getBlockCount(w, BLOCK_W), Saiga::CUDA::getBlockCount(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);

    singlePassConvolve2<T,RADIUS,BLOCK_W,BLOCK_H,Y_ELEMENTS> <<<blocks, threads>>>(src,dst);
}


template<typename T, int RADIUS, unsigned int BLOCK_W, unsigned int BLOCK_H, unsigned int Y_ELEMENTS>
__global__ static
void singlePassConvolve3(ImageView<T> src, ImageView<T> dst)
{
    const unsigned int BLOCK_H2 = BLOCK_H * Y_ELEMENTS;
    const unsigned int WARPS_PER_BLOCK = BLOCK_W * BLOCK_H / 32; //16
    static_assert(WARPS_PER_BLOCK == 16, "warps per block wrong");



    //for radius = 4: elements = (32+8) * (16+8) = 960 = 3840
    __shared__ T buffer[BLOCK_H2 + 2*RADIUS][BLOCK_W + 2*RADIUS];
    //for radius = 4: elements = (32+8) * (16) = 640 = 2560
    __shared__ T buffer2[BLOCK_H2][BLOCK_W + 2*RADIUS];
    //total s mem per block = 6400
    //with 512 threads per block smem per sm: 25600 -> 100% occ


    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int t = tx + ty * BLOCK_W;
    const unsigned int warp_lane = t / 32;
    const unsigned int lane_id = t & 31;

    int xp = blockIdx.x*BLOCK_W + tx;
    int yp = blockIdx.y*BLOCK_H2 + ty;
    int x = xp;
    int y = yp;

    const unsigned int x_tile = blockIdx.x * BLOCK_W;
    const unsigned int y_tile = blockIdx.y * BLOCK_H2;

    int blockStartX = blockIdx.x*BLOCK_W - RADIUS;
    int blockStartY = blockIdx.y*BLOCK_H2 - RADIUS;

    const int blockSizeX = BLOCK_W + 2*RADIUS;
    const int blockSizeY = BLOCK_H2 + 2*RADIUS;

    //copy main data
    for(int i = 0; i < Y_ELEMENTS; ++i)
    {
        buffer[ty + i * BLOCK_H + RADIUS][tx + RADIUS]  = src.clampedRead(x,y + i * BLOCK_H);
    }

    //top and bottom halo
    if(warp_lane < 4)
    {
        const unsigned int num_warps = 4;
        for(int i = warp_lane; i < RADIUS; i+=num_warps)
        {
            buffer[i][lane_id + RADIUS]  =
                    src.clampedRead(x_tile + lane_id,blockStartY + i);

            buffer[BLOCK_H2 + RADIUS + i][lane_id + RADIUS]  =
                    src.clampedRead(x_tile + lane_id,blockStartY + BLOCK_H2 + RADIUS  + i);
        }
    }

    const unsigned int side_halo_rows_per_warp = 32 / RADIUS;

    int local_warp_id = lane_id / RADIUS;
    int local_lane_id = lane_id % RADIUS;

    //left halo
    if(warp_lane >= 4 && warp_lane < 10)
    {
        const unsigned int num_warps = 6;
        int wid = warp_lane - 4;
        int rows = BLOCK_H2 + 2 * RADIUS;

        for(int i = wid * side_halo_rows_per_warp + local_warp_id;i < rows; i += num_warps*side_halo_rows_per_warp)
        {
            if(local_warp_id < side_halo_rows_per_warp)
            {
                buffer[i][local_lane_id]  =
                        src.clampedRead(blockStartX + local_lane_id,blockStartY + i);
            }
        }
    }

    //right halo
    if(warp_lane >= 10 && warp_lane < 16)
    {
        const unsigned int num_warps = 6;
        int wid = warp_lane - 10;
        int rows = BLOCK_H2 + 2 * RADIUS;

        for(int i = wid * side_halo_rows_per_warp + local_warp_id;i < rows; i += num_warps*side_halo_rows_per_warp)
        {
            if(local_warp_id < side_halo_rows_per_warp)
            {
                buffer[i][local_lane_id + RADIUS + BLOCK_W]  =
                        src.clampedRead(blockStartX + local_lane_id + RADIUS + BLOCK_W,blockStartY + i);
            }
        }
    }

    __syncthreads();


    T *kernel = d_Kernel;

    for(int i = t; i < blockSizeX * BLOCK_H2; i += (BLOCK_W*BLOCK_H)){
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

    for(int i =0; i < Y_ELEMENTS; ++i){
        T sum = 0;
#pragma unroll
        for (int j=-RADIUS;j<=RADIUS;j++){
            int kernelIndex = j + RADIUS;
            sum += buffer2[ty][tx + RADIUS + j] * kernel[kernelIndex];
        }

        if(dst.inImage(xp,yp))
            dst(xp,yp) = sum;
        yp += BLOCK_H;
        ty += BLOCK_H;
    }
}

template<typename T, int RADIUS>
inline
void convolve3(ImageView<T> src, ImageView<T> dst){
    int w = src.width;
    int h = src.height;

    const int BLOCK_W = 32;
    const int BLOCK_H = 16;
    const int Y_ELEMENTS = 2;
    dim3 blocks(
                Saiga::iDivUp(w, BLOCK_W ),
                Saiga::iDivUp(h, BLOCK_H * Y_ELEMENTS),
                1
                );

    //    dim3 blocks(Saiga::CUDA::getBlockCount(w, BLOCK_W), Saiga::CUDA::getBlockCount(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);

    singlePassConvolve3<T,RADIUS,BLOCK_W,BLOCK_H,Y_ELEMENTS> <<<blocks, threads>>>(src,dst);
}


template<typename T, int RADIUS, unsigned int BLOCK_W, unsigned int BLOCK_H, unsigned int Y_ELEMENTS>
__global__ static
void singlePassConvolve4(ImageView<T> src, ImageView<T> dst)
{
    const unsigned int TILE_H = BLOCK_H;
    const unsigned int TILE_W = BLOCK_W;

    const unsigned int TILE_H2 = TILE_H * Y_ELEMENTS;
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    int t = tx + ty * BLOCK_W;

    int x_tile = blockIdx.x * (TILE_W - 2 * RADIUS) - RADIUS;
    int y_tile = blockIdx.y * (TILE_H2 - 2 * RADIUS) - RADIUS;

    int x = x_tile + tx;
    int y = y_tile + ty;


    __shared__ T buffer[TILE_H2][TILE_W];
    __shared__ T buffer2[TILE_H2 - RADIUS * 2][TILE_W];



    //copy main data
    for(int i = 0; i < Y_ELEMENTS; ++i)
        buffer[ty + i * TILE_H][tx]  = src.clampedRead(x,y + i * TILE_H);



    __syncthreads();


    T *kernel = d_Kernel;

    //convolve along y axis
    //    if(ty > RADIUS && ty < TILE_H2 - RADIUS)
    //    {
    //        int oy = ty - RADIUS;

    for(int i = 0; i < Y_ELEMENTS; ++i)
    {
//        int gx = x;
//        int gy = y + i * TILE_H;
        int lx = tx;
        int ly = ty + i * TILE_H;

        if(ly < RADIUS || ly >= TILE_H2 - RADIUS)
            continue;

        T sum = 0;
#pragma unroll
        for (int j=-RADIUS;j<=RADIUS;j++)
        {
            int kernelIndex = j + RADIUS;
            sum += buffer[ly + j][lx] * kernel[kernelIndex];
        }
        buffer2[ly - RADIUS][lx] = sum;
    }



    __syncthreads();

    for(int i = 0; i < Y_ELEMENTS; ++i)
    {
        int gx = x;
        int gy = y + i * TILE_H;

        int lx = tx;
        int ly = ty + i * TILE_H;

        if(ly < RADIUS || ly >= TILE_H2 - RADIUS)
            continue;

        if(lx < RADIUS || lx >= TILE_W - RADIUS)
            continue;

        T sum = 0;
#pragma unroll
        for (int j=-RADIUS;j<=RADIUS;j++)
        {
            int kernelIndex = j + RADIUS;
            sum += buffer2[ly - RADIUS][lx + j] * kernel[kernelIndex];
        }

//        if(dst.inImage(gx,gy))
//            dst(g,yp) = sum;
        dst.clampedWrite(gx,gy,sum);
    }



#if 0

    for(int i =0; i < Y_ELEMENTS; ++i){
        T sum = 0;
#pragma unroll
        for (int j=-RADIUS;j<=RADIUS;j++){
            int kernelIndex = j + RADIUS;
            sum += buffer2[ty][tx + RADIUS + j] * kernel[kernelIndex];
        }

        if(dst.inImage(xp,yp))
            dst(xp,yp) = sum;
        yp += BLOCK_H;
        ty += BLOCK_H;
    }
#endif
}

template<typename T, int RADIUS>
inline
void convolve4(ImageView<T> src, ImageView<T> dst){
    int w = src.width;
    int h = src.height;

    const int BLOCK_W = 32;
    const int BLOCK_H = 16;
    const int Y_ELEMENTS = 2;
    dim3 blocks(
                Saiga::iDivUp(w, BLOCK_W - 2 * RADIUS),
                Saiga::iDivUp(h, BLOCK_H * Y_ELEMENTS - 2 * RADIUS),
                1
                );

    //    dim3 blocks(Saiga::CUDA::getBlockCount(w, BLOCK_W), Saiga::CUDA::getBlockCount(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);

    singlePassConvolve4<T,RADIUS,BLOCK_W,BLOCK_H,Y_ELEMENTS> <<<blocks, threads>>>(src,dst);
}

void convolveSinglePassSeparate(ImageView<float> src, ImageView<float> dst, Saiga::array_view<float> kernel, int radius){
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_Kernel, kernel.data(), kernel.size()*sizeof(float),0,cudaMemcpyDeviceToDevice));

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


void convolveSinglePassSeparate2(ImageView<float> src, ImageView<float> dst, Saiga::array_view<float> kernel, int radius){
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_Kernel, kernel.data(), kernel.size()*sizeof(float),0,cudaMemcpyDeviceToDevice));
    switch (radius){
    case 3: CUDA::convolve2<float,3>(src,dst); break;
    case 4: CUDA::convolve2<float,4>(src,dst); break;
    }
}

void convolveSinglePassSeparate3(ImageView<float> src, ImageView<float> dst, Saiga::array_view<float> kernel, int radius){
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_Kernel, kernel.data(), kernel.size()*sizeof(float),0,cudaMemcpyDeviceToDevice));
    switch (radius){
    case 3: CUDA::convolve3<float,3>(src,dst); break;
    case 4: CUDA::convolve3<float,4>(src,dst); break;
    }
}

void convolveSinglePassSeparate4(ImageView<float> src, ImageView<float> dst, Saiga::array_view<float> kernel, int radius){
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_Kernel, kernel.data(), kernel.size()*sizeof(float),0,cudaMemcpyDeviceToDevice));
    switch (radius){
    case 3: CUDA::convolve4<float,3>(src,dst); break;
    case 4: CUDA::convolve4<float,4>(src,dst); break;
    }
}

}
}
