/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/imageProcessing/imageProcessing.h"
#include "saiga/cuda/device_helper.h"

using std::cout;
using std::endl;

namespace Saiga {
namespace CUDA {


__constant__ float d_Kernel[SAIGA_MAX_KERNEL_SIZE];


template<typename T, int RADIUS, unsigned int BLOCK_W, unsigned int BLOCK_H, unsigned int Y_ELEMENTS>
__global__ static
void d_convolveOuterLinear(ImageView<T> src, ImageView<T> dst)
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
        src.clampToEdge(gy,gx);
        buffer[y][x] = src(gy,gx);
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

        if(dst.inImage(yp,xp))
            dst(yp,xp) = sum;
        yp += BLOCK_H;
        ty += BLOCK_H;
    }
}

template<typename T, int RADIUS>
inline
void convolveOuterLinear(ImageView<T> src, ImageView<T> dst){
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

    d_convolveOuterLinear<T,RADIUS,BLOCK_W,BLOCK_H,Y_ELEMENTS> <<<blocks, threads>>>(src,dst);
}


template<typename T, int RADIUS, unsigned int BLOCK_W, unsigned int BLOCK_H, unsigned int Y_ELEMENTS>
__global__ static
void d_convolveOuterHalo(ImageView<T> src, ImageView<T> dst)
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
//    const unsigned int y_tile = blockIdx.y * BLOCK_H2;

    int blockStartX = blockIdx.x*BLOCK_W - RADIUS;
    int blockStartY = blockIdx.y*BLOCK_H2 - RADIUS;

    const int blockSizeX = BLOCK_W + 2*RADIUS;
//    const int blockSizeY = BLOCK_H2 + 2*RADIUS;

    //copy main data
    for(int i = 0; i < Y_ELEMENTS; ++i)
    {
        buffer[ty + i * BLOCK_H + RADIUS][tx + RADIUS]  = src.clampedRead(y + i * BLOCK_H,x);
    }

    //top and bottom halo
    if(warp_lane < 4)
    {
        const unsigned int num_warps = 4;
        for(int i = warp_lane; i < RADIUS; i+=num_warps)
        {
            buffer[i][lane_id + RADIUS]  =
                    src.clampedRead(blockStartY + i,x_tile + lane_id);

            buffer[BLOCK_H2 + RADIUS + i][lane_id + RADIUS]  =
                    src.clampedRead(blockStartY + BLOCK_H2 + RADIUS  + i,x_tile + lane_id);
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
                        src.clampedRead(blockStartY + i,blockStartX + local_lane_id);
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
                        src.clampedRead(blockStartY + i,blockStartX + local_lane_id + RADIUS + BLOCK_W);
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

        if(dst.inImage(yp,xp))
            dst(yp,xp) = sum;
        yp += BLOCK_H;
        ty += BLOCK_H;
    }
}

template<typename T, int RADIUS>
inline
void convolveOuterHalo(ImageView<T> src, ImageView<T> dst){
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

    d_convolveOuterHalo<T,RADIUS,BLOCK_W,BLOCK_H,Y_ELEMENTS> <<<blocks, threads>>>(src,dst);
}


template<typename T, int RADIUS, unsigned int BLOCK_W, unsigned int BLOCK_H, unsigned int Y_ELEMENTS>
__global__ static
void d_convolveInner(ImageView<T> src, ImageView<T> dst)
{
    const unsigned int TILE_H = BLOCK_H;
    const unsigned int TILE_W = BLOCK_W;

    const unsigned int TILE_H2 = TILE_H * Y_ELEMENTS;
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
//    int t = tx + ty * BLOCK_W;

    int x_tile = blockIdx.x * (TILE_W - 2 * RADIUS) - RADIUS;
    int y_tile = blockIdx.y * (TILE_H2 - 2 * RADIUS) - RADIUS;

    int x = x_tile + tx;
    int y = y_tile + ty;


    __shared__ T buffer[TILE_H2][TILE_W];
    __shared__ T buffer2[TILE_H2 - RADIUS * 2][TILE_W];



    //copy main data
    for(int i = 0; i < Y_ELEMENTS; ++i)
        buffer[ty + i * TILE_H][tx]  = src.clampedRead(y + i * TILE_H,x);



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
        dst.clampedWrite(gy,gx,sum);
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

template<typename T, int RADIUS, bool LOW_OCC = false>
inline
void convolveInner(ImageView<T> src, ImageView<T> dst){
    int w = src.width;
    int h = src.height;


    const int BLOCK_W = LOW_OCC ? 64 : 32;
    const int BLOCK_H = LOW_OCC ? 8 : 16;
    const int Y_ELEMENTS = LOW_OCC ? 4 : 2;
    dim3 blocks(
                Saiga::iDivUp(w, BLOCK_W - 2 * RADIUS),
                Saiga::iDivUp(h, BLOCK_H * Y_ELEMENTS - 2 * RADIUS),
                1
                );

    //    dim3 blocks(Saiga::CUDA::getBlockCount(w, BLOCK_W), Saiga::CUDA::getBlockCount(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);

    d_convolveInner<T,RADIUS,BLOCK_W,BLOCK_H,Y_ELEMENTS> <<<blocks, threads>>>(src,dst);
}



void convolveSinglePassSeparateOuterLinear(ImageView<float> src, ImageView<float> dst, Saiga::array_view<float> kernel, int radius){
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_Kernel, kernel.data(), kernel.size()*sizeof(float),0,cudaMemcpyDeviceToDevice));
    switch (radius){
    case 1: CUDA::convolveOuterLinear<float,1>(src,dst); break;
    case 2: CUDA::convolveOuterLinear<float,2>(src,dst); break;
    case 3: CUDA::convolveOuterLinear<float,3>(src,dst); break;
    case 4: CUDA::convolveOuterLinear<float,4>(src,dst); break;
    case 5: CUDA::convolveOuterLinear<float,5>(src,dst); break;
    case 6: CUDA::convolveOuterLinear<float,6>(src,dst); break;
    case 7: CUDA::convolveOuterLinear<float,7>(src,dst); break;
    case 8: CUDA::convolveOuterLinear<float,8>(src,dst); break;
    case 9: CUDA::convolveOuterLinear<float,9>(src,dst); break;
    }
}

void convolveSinglePassSeparateOuterHalo(ImageView<float> src, ImageView<float> dst, Saiga::array_view<float> kernel, int radius){
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_Kernel, kernel.data(), kernel.size()*sizeof(float),0,cudaMemcpyDeviceToDevice));
    switch (radius){
    case 1: CUDA::convolveOuterHalo<float,1>(src,dst); break;
    case 2: CUDA::convolveOuterHalo<float,2>(src,dst); break;
    case 3: CUDA::convolveOuterHalo<float,3>(src,dst); break;
    case 4: CUDA::convolveOuterHalo<float,4>(src,dst); break;
    case 5: CUDA::convolveOuterHalo<float,5>(src,dst); break;
    case 6: CUDA::convolveOuterHalo<float,6>(src,dst); break;
    case 7: CUDA::convolveOuterHalo<float,7>(src,dst); break;
    case 8: CUDA::convolveOuterHalo<float,8>(src,dst); break;
    }
}

void convolveSinglePassSeparateInner(ImageView<float> src, ImageView<float> dst, Saiga::array_view<float> kernel, int radius){
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_Kernel, kernel.data(), kernel.size()*sizeof(float),0,cudaMemcpyDeviceToDevice));
    switch (radius){
    case 1: CUDA::convolveInner<float,1>(src,dst); break;
    case 2: CUDA::convolveInner<float,2>(src,dst); break;
    case 3: CUDA::convolveInner<float,3>(src,dst); break;
    case 4: CUDA::convolveInner<float,4>(src,dst); break;
    case 5: CUDA::convolveInner<float,5>(src,dst); break;
    case 6: CUDA::convolveInner<float,6>(src,dst); break;
    case 7: CUDA::convolveInner<float,7>(src,dst); break;
    case 8: CUDA::convolveInner<float,8>(src,dst); break;
    }
}


void convolveSinglePassSeparateInner75(ImageView<float> src, ImageView<float> dst, Saiga::array_view<float> kernel, int radius){
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_Kernel, kernel.data(), kernel.size()*sizeof(float),0,cudaMemcpyDeviceToDevice));
    switch (radius){
    case 1: CUDA::convolveInner<float,1,true>(src,dst); break;
    case 2: CUDA::convolveInner<float,2,true>(src,dst); break;
    case 3: CUDA::convolveInner<float,3,true>(src,dst); break;
    case 4: CUDA::convolveInner<float,4,true>(src,dst); break;
    case 5: CUDA::convolveInner<float,5,true>(src,dst); break;
    case 6: CUDA::convolveInner<float,6,true>(src,dst); break;
    case 7: CUDA::convolveInner<float,7,true>(src,dst); break;
    case 8: CUDA::convolveInner<float,8,true>(src,dst); break;
    }
}

}
}
