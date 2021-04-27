/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/imageProcessing/imageProcessing.h"
#include "saiga/cuda/shfl_helper.h"


namespace Saiga
{
namespace CUDA
{
//todo maybe change
static __constant__ float d_Kernel[SAIGA_MAX_KERNEL_SIZE];


template <typename T, int RADIUS, unsigned int BLOCK_W, unsigned int BLOCK_H, unsigned int Y_ELEMENTS>
__global__ static void d_convolveOuterLinear(ImageView<T> src, ImageView<T> dst)
{
    const unsigned BLOCK_H2 = BLOCK_H * Y_ELEMENTS;

    // for radius = 4: elements = (32+8) * (16+8) = 960 = 3840
    __shared__ T buffer[BLOCK_H2 + 2 * RADIUS][BLOCK_W + 2 * RADIUS];
    // for radius = 4: elements = (32+8) * (16) = 640 = 2560
    __shared__ T buffer2[BLOCK_H2][BLOCK_W + 2 * RADIUS];
    // total s mem per block = 6400
    // with 512 threads per block smem per sm: 25600 -> 100% occ


    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int t  = tx + ty * BLOCK_W;
    int xp = blockIdx.x * BLOCK_W + tx;
    int yp = blockIdx.y * BLOCK_H2 + ty;


    int blockStartX = blockIdx.x * BLOCK_W - RADIUS;
    int blockStartY = blockIdx.y * BLOCK_H2 - RADIUS;

    const int blockSizeX = BLOCK_W + 2 * RADIUS;
    const int blockSizeY = BLOCK_H2 + 2 * RADIUS;

    // fill buffer
    for (int i = t; i < blockSizeX * blockSizeY; i += (BLOCK_W * BLOCK_H))
    {
        int x  = i % blockSizeX;
        int y  = i / blockSizeX;
        int gx = x + blockStartX;
        int gy = y + blockStartY;
        src.clampToEdge(gy, gx);
        buffer[y][x] = src(gy, gx);
    }

    __syncthreads();


    T* kernel = d_Kernel;

    for (int i = t; i < blockSizeX * BLOCK_H2; i += (BLOCK_W * BLOCK_H))
    {
        int x = i % blockSizeX;
        int y = i / blockSizeX;
        T sum = 0;
#pragma unroll
        for (int j = -RADIUS; j <= RADIUS; j++)
        {
            int kernelIndex = j + RADIUS;
            sum += buffer[y + RADIUS + j][x] * kernel[kernelIndex];
        }
        buffer2[y][x] = sum;
    }

    __syncthreads();

    for (int i = 0; i < Y_ELEMENTS; ++i)
    {
        T sum = 0;
#pragma unroll
        for (int j = -RADIUS; j <= RADIUS; j++)
        {
            int kernelIndex = j + RADIUS;
            sum += buffer2[ty][tx + RADIUS + j] * kernel[kernelIndex];
        }

        if (dst.inImage(yp, xp)) dst(yp, xp) = sum;
        yp += BLOCK_H;
        ty += BLOCK_H;
    }
}

template <typename T, int RADIUS>
inline void convolveOuterLinear(ImageView<T> src, ImageView<T> dst)
{
    int w = src.width;
    int h = src.height;

    const int BLOCK_W    = 32;
    const int BLOCK_H    = 16;
    const int Y_ELEMENTS = 2;
    dim3 blocks(Saiga::iDivUp(w, BLOCK_W), Saiga::iDivUp(h, BLOCK_H * Y_ELEMENTS), 1);

    //    dim3 blocks(Saiga::CUDA::getBlockCount(w, BLOCK_W), Saiga::CUDA::getBlockCount(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);

    d_convolveOuterLinear<T, RADIUS, BLOCK_W, BLOCK_H, Y_ELEMENTS><<<blocks, threads>>>(src, dst);
}


template <typename T, int RADIUS, unsigned int BLOCK_W, unsigned int BLOCK_H, unsigned int Y_ELEMENTS>
__global__ static void d_convolveOuterHalo(ImageView<T> src, ImageView<T> dst)
{
    const unsigned int BLOCK_H2        = BLOCK_H * Y_ELEMENTS;
    const unsigned int WARPS_PER_BLOCK = BLOCK_W * BLOCK_H / 32;  // 16
    static_assert(WARPS_PER_BLOCK == 16, "warps per block wrong");



    // for radius = 4: elements = (32+8) * (16+8) = 960 = 3840
    __shared__ T buffer[BLOCK_H2 + 2 * RADIUS][BLOCK_W + 2 * RADIUS];
    // for radius = 4: elements = (32+8) * (16) = 640 = 2560
    __shared__ T buffer2[BLOCK_H2][BLOCK_W + 2 * RADIUS];
    // total s mem per block = 6400
    // with 512 threads per block smem per sm: 25600 -> 100% occ


    int tx                       = threadIdx.x;
    int ty                       = threadIdx.y;
    int t                        = tx + ty * BLOCK_W;
    const unsigned int warp_lane = t / 32;
    const unsigned int lane_id   = t & 31;

    int xp = blockIdx.x * BLOCK_W + tx;
    int yp = blockIdx.y * BLOCK_H2 + ty;
    int x  = xp;
    int y  = yp;

    const unsigned int x_tile = blockIdx.x * BLOCK_W;
    //    const unsigned int y_tile = blockIdx.y * BLOCK_H2;

    int blockStartX = blockIdx.x * BLOCK_W - RADIUS;
    int blockStartY = blockIdx.y * BLOCK_H2 - RADIUS;

    const int blockSizeX = BLOCK_W + 2 * RADIUS;
    //    const int blockSizeY = BLOCK_H2 + 2*RADIUS;

    // copy main data
    for (int i = 0; i < Y_ELEMENTS; ++i)
    {
        buffer[ty + i * BLOCK_H + RADIUS][tx + RADIUS] = src.clampedRead(y + i * BLOCK_H, x);
    }

    // top and bottom halo
    if (warp_lane < 4)
    {
        const unsigned int num_warps = 4;
        for (int i = warp_lane; i < RADIUS; i += num_warps)
        {
            buffer[i][lane_id + RADIUS] = src.clampedRead(blockStartY + i, x_tile + lane_id);

            buffer[BLOCK_H2 + RADIUS + i][lane_id + RADIUS] =
                src.clampedRead(blockStartY + BLOCK_H2 + RADIUS + i, x_tile + lane_id);
        }
    }

    const unsigned int side_halo_rows_per_warp = 32 / RADIUS;

    int local_warp_id = lane_id / RADIUS;
    int local_lane_id = lane_id % RADIUS;

    // left halo
    if (warp_lane >= 4 && warp_lane < 10)
    {
        const unsigned int num_warps = 6;
        int wid                      = warp_lane - 4;
        int rows                     = BLOCK_H2 + 2 * RADIUS;

        for (int i = wid * side_halo_rows_per_warp + local_warp_id; i < rows; i += num_warps * side_halo_rows_per_warp)
        {
            if (local_warp_id < side_halo_rows_per_warp)
            {
                buffer[i][local_lane_id] = src.clampedRead(blockStartY + i, blockStartX + local_lane_id);
            }
        }
    }

    // right halo
    if (warp_lane >= 10 && warp_lane < 16)
    {
        const unsigned int num_warps = 6;
        int wid                      = warp_lane - 10;
        int rows                     = BLOCK_H2 + 2 * RADIUS;

        for (int i = wid * side_halo_rows_per_warp + local_warp_id; i < rows; i += num_warps * side_halo_rows_per_warp)
        {
            if (local_warp_id < side_halo_rows_per_warp)
            {
                buffer[i][local_lane_id + RADIUS + BLOCK_W] =
                    src.clampedRead(blockStartY + i, blockStartX + local_lane_id + RADIUS + BLOCK_W);
            }
        }
    }

    __syncthreads();


    T* kernel = d_Kernel;

    for (int i = t; i < blockSizeX * BLOCK_H2; i += (BLOCK_W * BLOCK_H))
    {
        int x = i % blockSizeX;
        int y = i / blockSizeX;
        T sum = 0;
#pragma unroll
        for (int j = -RADIUS; j <= RADIUS; j++)
        {
            int kernelIndex = j + RADIUS;
            sum += buffer[y + RADIUS + j][x] * kernel[kernelIndex];
        }
        buffer2[y][x] = sum;
    }

    __syncthreads();

    for (int i = 0; i < Y_ELEMENTS; ++i)
    {
        T sum = 0;
#pragma unroll
        for (int j = -RADIUS; j <= RADIUS; j++)
        {
            int kernelIndex = j + RADIUS;
            sum += buffer2[ty][tx + RADIUS + j] * kernel[kernelIndex];
        }

        if (dst.inImage(yp, xp)) dst(yp, xp) = sum;
        yp += BLOCK_H;
        ty += BLOCK_H;
    }
}

template <typename T, int RADIUS>
inline void convolveOuterHalo(ImageView<T> src, ImageView<T> dst)
{
    int w = src.width;
    int h = src.height;

    const int BLOCK_W    = 32;
    const int BLOCK_H    = 16;
    const int Y_ELEMENTS = 2;
    dim3 blocks(Saiga::iDivUp(w, BLOCK_W), Saiga::iDivUp(h, BLOCK_H * Y_ELEMENTS), 1);

    //    dim3 blocks(Saiga::CUDA::getBlockCount(w, BLOCK_W), Saiga::CUDA::getBlockCount(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);

    d_convolveOuterHalo<T, RADIUS, BLOCK_W, BLOCK_H, Y_ELEMENTS><<<blocks, threads>>>(src, dst);
}


template <typename T, int RADIUS, unsigned int BLOCK_W, unsigned int BLOCK_H, unsigned int Y_ELEMENTS>
__global__ static void d_convolveInner(ImageView<T> src, ImageView<T> dst)
{
    const unsigned int TILE_H = BLOCK_H;
    const unsigned int TILE_W = BLOCK_W;

    const unsigned int TILE_H2 = TILE_H * Y_ELEMENTS;
    const unsigned int tx      = threadIdx.x;
    const unsigned int ty      = threadIdx.y;
    //    int t = tx + ty * BLOCK_W;

    int x_tile = blockIdx.x * (TILE_W - 2 * RADIUS) - RADIUS;
    int y_tile = blockIdx.y * (TILE_H2 - 2 * RADIUS) - RADIUS;

    int x = x_tile + tx;
    int y = y_tile + ty;


    __shared__ T buffer[TILE_H2][TILE_W];
    __shared__ T buffer2[TILE_H2 - RADIUS * 2][TILE_W];



    // copy main data
    for (int i = 0; i < Y_ELEMENTS; ++i) buffer[ty + i * TILE_H][tx] = src.clampedRead(y + i * TILE_H, x);



    __syncthreads();


    T* kernel = d_Kernel;

    // convolve along y axis
    //    if(ty > RADIUS && ty < TILE_H2 - RADIUS)
    //    {
    //        int oy = ty - RADIUS;

    for (int i = 0; i < Y_ELEMENTS; ++i)
    {
        //        int gx = x;
        //        int gy = y + i * TILE_H;
        int lx = tx;
        int ly = ty + i * TILE_H;

        if (ly < RADIUS || ly >= TILE_H2 - RADIUS) continue;

        T sum = 0;
#pragma unroll
        for (int j = -RADIUS; j <= RADIUS; j++)
        {
            int kernelIndex = j + RADIUS;
            sum += buffer[ly + j][lx] * kernel[kernelIndex];
        }
        buffer2[ly - RADIUS][lx] = sum;
    }



    __syncthreads();

    for (int i = 0; i < Y_ELEMENTS; ++i)
    {
        int gx = x;
        int gy = y + i * TILE_H;

        int lx = tx;
        int ly = ty + i * TILE_H;

        if (ly < RADIUS || ly >= TILE_H2 - RADIUS) continue;

        if (lx < RADIUS || lx >= TILE_W - RADIUS) continue;

        T sum = 0;
#pragma unroll
        for (int j = -RADIUS; j <= RADIUS; j++)
        {
            int kernelIndex = j + RADIUS;
            sum += buffer2[ly - RADIUS][lx + j] * kernel[kernelIndex];
        }

        //        if(dst.inImage(gx,gy))
        //            dst(g,yp) = sum;
        dst.clampedWrite(gy, gx, sum);
    }



#if 0

    for(int i =0; i < Y_ELEMENTS; ++i){
        T sum = 0;
#    pragma unroll
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

template <typename T, int RADIUS, bool LOW_OCC = false>
inline void convolveInner(ImageView<T> src, ImageView<T> dst)
{
    int w = src.width;
    int h = src.height;


    const int BLOCK_W    = LOW_OCC ? 64 : 32;
    const int BLOCK_H    = LOW_OCC ? 8 : 16;
    const int Y_ELEMENTS = LOW_OCC ? 4 : 2;
    dim3 blocks(Saiga::iDivUp(w, BLOCK_W - 2 * RADIUS), Saiga::iDivUp(h, BLOCK_H * Y_ELEMENTS - 2 * RADIUS), 1);

    //    dim3 blocks(Saiga::CUDA::getBlockCount(w, BLOCK_W), Saiga::CUDA::getBlockCount(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);

    d_convolveInner<T, RADIUS, BLOCK_W, BLOCK_H, Y_ELEMENTS><<<blocks, threads>>>(src, dst);
}



template <typename T, int RADIUS, unsigned int BLOCK_W, unsigned int BLOCK_H, unsigned int Y_ELEMENTS>
__global__ static void d_convolveInnerShuffle(ImageView<T> src, ImageView<T> dst)
{
    const unsigned int TILE_H = BLOCK_H;
    const unsigned int TILE_W = BLOCK_W;

    const unsigned int TILE_H2 = TILE_H * Y_ELEMENTS;
    const unsigned int tx      = threadIdx.x;
    const unsigned int ty      = threadIdx.y;
    //    int t = tx + ty * BLOCK_W;

    unsigned int lane_id = threadIdx.x % 32;

    int x_tile = blockIdx.x * (TILE_W - 2 * RADIUS) - RADIUS;
    int y_tile = blockIdx.y * (TILE_H2 - 2 * RADIUS) - RADIUS;

    int x = x_tile + tx;
    int y = y_tile + ty;



    //    __shared__ T buffer[TILE_H2][TILE_W];
    //    __shared__ T buffer2[TILE_H2][TILE_W - RADIUS * 2 + 1];
    __shared__ T buffer2[TILE_H2][TILE_W - RADIUS * 2];
    //    __shared__ T buffer2[TILE_W - RADIUS * 2][TILE_H2];


    T localElements[Y_ELEMENTS];
    for (int i = 0; i < Y_ELEMENTS; ++i)
    {
        localElements[i] = src.clampedRead(y + i * TILE_H, x);
    }

    // conv row

    T* kernel = d_Kernel;


    for (int i = 0; i < Y_ELEMENTS; ++i)
    {
        int lx = tx;
        int ly = ty + i * TILE_H;
        T sum  = 0;
#pragma unroll
        for (int j = -RADIUS; j <= RADIUS; j++)
        {
            int kernelIndex = j + RADIUS;
            auto value      = shfl(localElements[i], lane_id + j);

            sum += value * kernel[kernelIndex];
        }

        if (lx < RADIUS || lx >= TILE_W - RADIUS) continue;

        buffer2[ly][lx - RADIUS] = sum;
        //        buffer2[lx- RADIUS][ly] = sum;
    }



    __syncthreads();

    // conv col

    for (int i = 0; i < Y_ELEMENTS; ++i)
    {
        int gx = x;
        int gy = y + i * TILE_H;

        int lx = tx;
        int ly = ty + i * TILE_H;

        if (ly < RADIUS || ly >= TILE_H2 - RADIUS) continue;

        if (lx < RADIUS || lx >= TILE_W - RADIUS) continue;

        T sum = 0;
#if 1
#    pragma unroll
        for (int j = -RADIUS; j <= RADIUS; j++)
        {
            int kernelIndex = j + RADIUS;
            auto value      = buffer2[ly + j][lx - RADIUS];
            //            auto value = buffer2[lx - RADIUS][ly + j];
            sum += value * kernel[kernelIndex];
        }
#endif
        dst.clampedWrite(gy, gx, sum);
    }
}

// | ---- BLOCK_W * X_ELEMENTS * vectorSize ---- |
// [ x x x x x x x x x x x x x x x x x x x x x x ]
//


template <typename T, int RADIUS, unsigned int BLOCK_W, unsigned int BLOCK_H, unsigned int X_ELEMENTS,
          unsigned int Y_ELEMENTS, typename VectorType = int2>
//__launch_bounds__(BLOCK_W* BLOCK_H, 3)
     __global__ static void d_convolveInnerShuffle2(ImageView<T> src, ImageView<T> dst)
{
    const unsigned int TILE_W = BLOCK_W;
    const unsigned int TILE_H = BLOCK_H;

    const unsigned int TILE_W2 = TILE_W * X_ELEMENTS;
    const unsigned int TILE_H2 = TILE_H * Y_ELEMENTS;
    const unsigned int tx      = threadIdx.x;
    const unsigned int ty      = threadIdx.y;
    //    int t = tx + ty * BLOCK_W;

    // static_assert( sizeof(VectorType) / sizeof(T) == X_ELEMENTS);

    unsigned int lane_id = threadIdx.x % 32;

    // start position of tile
    int x_tile = blockIdx.x * (TILE_W2 - 2 * RADIUS) - RADIUS;
    int y_tile = blockIdx.y * (TILE_H2 - 2 * RADIUS) - RADIUS;

    // global position of thread
    int x = x_tile + tx * X_ELEMENTS;
    int y = y_tile + ty;


    T* kernel = d_Kernel;


    // for vec4 radius 8:
    // (16 * Y_ELEMENTS) * (32 - 4) * 16
    // Y 3 -> 21504  100 occ
    // Y 4 -> 28672  75 occ
    // Y 5 -> 35840  50 occ
    // Y 6 -> 43008  50 occ
    // Y 8 -> 57344  failed
    __shared__ VectorType buffer2[TILE_H2][TILE_W - 2 * RADIUS / X_ELEMENTS];


    // own element + left and right radius
    VectorType localElements[Y_ELEMENTS][1 + 2 * RADIUS / X_ELEMENTS];  // 5

    // without this unroll we get a strange compile error
#pragma unroll
    for (int i = 0; i < Y_ELEMENTS; ++i)
    {
        int rowId = y + i * TILE_H;
        rowId     = std::min(rowId, src.height - 1);
        rowId     = std::max(0, rowId);
        int colId = std::max(0, x);

        int xb = Saiga::iAlignUp(src.width, X_ELEMENTS) - X_ELEMENTS;
        colId  = std::min(colId, xb);


        T* row = src.rowPtr(rowId);
        CUDA_ASSERT(size_t(row) % sizeof(VectorType) == 0);
        T* elem = row + colId;
        //        if(rowId == 0)
        //            printf("%d \n",colId);
        T* localElementsT = reinterpret_cast<T*>(localElements[i]);


        // center of localElements
        // the left and right of the center will be filled by shuffles
        VectorType& myValue = localElements[i][RADIUS / X_ELEMENTS];  //[i][2]

        // load own value from global memory (note: this is the only global memory read)
        CUDA_ASSERT(size_t(elem) % sizeof(VectorType) == 0);
        myValue = reinterpret_cast<VectorType*>(elem)[0];

        if (x < 0)
        {
            for (int k = 0; k < X_ELEMENTS; ++k)
            {
                localElementsT[RADIUS + k] = localElementsT[RADIUS];
            }
        }
        if (x >= src.width)
        {
            for (int k = 0; k < X_ELEMENTS; ++k)
            {
                localElementsT[RADIUS + k] = localElementsT[RADIUS + X_ELEMENTS - 1];
            }
        }


        // shuffle left
        for (int j = 0; j < RADIUS / X_ELEMENTS; ++j)
        {
            localElements[i][j] = shfl(myValue, lane_id + j - RADIUS / X_ELEMENTS);
        }

        // shuffle right
        for (int j = 0; j < RADIUS / X_ELEMENTS; ++j)
        {
            localElements[i][j + RADIUS / X_ELEMENTS + 1] = shfl(myValue, lane_id + j + 1);
        }



        T sum[X_ELEMENTS];
#pragma unroll
        for (int j = 0; j < X_ELEMENTS; ++j)
        {
            sum[j] = 0;
        }

#pragma unroll
        for (int j = -RADIUS; j <= RADIUS; j++)
        {
            T kernelValue = kernel[j + RADIUS];
#pragma unroll
            for (int k = 0; k < X_ELEMENTS; ++k)
            {
                sum[k] += localElementsT[RADIUS + j + k] * kernelValue;
            }
        }

        // write to shared memory if this thread is 'inner' (not in the halo)
        int lx = tx;
        int ly = ty + i * TILE_H;


        // continue if this thread is not a 'inner thread'
        if (lx < RADIUS / X_ELEMENTS || lx >= TILE_W - RADIUS / X_ELEMENTS) continue;

        if (x >= src.width) continue;

        //        if(lx >= RADIUS / X_ELEMENTS && lx < TILE_W - RADIUS / X_ELEMENTS)
        {
            if (rowId <= RADIUS && colId == 508)
            {
                //                printf("sum row %d %d %f \n",x,y,sum[0]);
                if (y == 1)
                    for (int k = 0; k < X_ELEMENTS + 2 * RADIUS; ++k)
                    {
                        //                   printf("localElementsT %d %f \n",k,localElementsT[k]);
                    }
            }
            buffer2[ly][lx - RADIUS / X_ELEMENTS] = reinterpret_cast<VectorType*>(sum)[0];
        }
        //           return;
    }


    // the only sync in this kernel
    __syncthreads();

#pragma unroll
    for (int i = 0; i < Y_ELEMENTS; ++i)
    {
        int rowId = y + i * TILE_H;
        rowId     = std::min(rowId, src.height - 1);
        rowId     = std::max(0, rowId);


        int colId = std::max(0, x);
        int xb    = Saiga::iAlignUp(src.width, X_ELEMENTS) - X_ELEMENTS;
        colId     = std::min(colId, xb);
        // colId = std::min(colId,src.width - X_ELEMENTS);


        // continue if this thread is not a 'inner thread'
        int lx = tx;
        int ly = ty + i * TILE_H;
        if (lx < RADIUS / X_ELEMENTS || lx >= TILE_W - RADIUS / X_ELEMENTS) continue;
        if (ly < RADIUS || ly >= TILE_H2 - RADIUS) continue;

        // continue if this thread is not in image
        if (x >= src.width || y + i * TILE_H >= src.height) continue;



        T* row  = dst.rowPtr(rowId);
        T* elem = row + colId;


        T sum[X_ELEMENTS];
        for (int j = 0; j < X_ELEMENTS; ++j)
        {
            sum[j] = 0;
        }

        // simple row convolution in shared memory
        for (int j = -RADIUS; j <= RADIUS; j++)
        {
            T kernelValue     = kernel[j + RADIUS];
            VectorType valueV = buffer2[ly + j][lx - RADIUS / X_ELEMENTS];
            for (int k = 0; k < X_ELEMENTS; ++k)
            {
                auto v = reinterpret_cast<T*>(&valueV)[k];
                sum[k] += v * kernelValue;
            }
        }

        reinterpret_cast<VectorType*>(elem)[0] = reinterpret_cast<VectorType*>(sum)[0];
    }
}


template <typename T, int RADIUS, typename VectorType = int>
inline void convolveInnerShuffle(ImageView<T> src, ImageView<T> dst)
{
    int w = src.width;
    int h = src.height;
    //    int p = src.pitchBytes;


    const int BLOCK_W = 32;
    const int BLOCK_H = 16;

    const int X_ELEMENTS = sizeof(VectorType) / sizeof(T);
    const int Y_ELEMENTS = 4;

    dim3 blocks(Saiga::iDivUp(w, BLOCK_W * X_ELEMENTS - 2 * RADIUS),
                Saiga::iDivUp(h, BLOCK_H * Y_ELEMENTS - 2 * RADIUS), 1);
    dim3 threads(BLOCK_W, BLOCK_H);


    if (sizeof(VectorType) >= 8)
        cudaFuncSetSharedMemConfig(
            d_convolveInnerShuffle2<T, RADIUS, BLOCK_W, BLOCK_H, X_ELEMENTS, Y_ELEMENTS, VectorType>,
            cudaSharedMemBankSizeEightByte);
    else
        cudaFuncSetSharedMemConfig(
            d_convolveInnerShuffle2<T, RADIUS, BLOCK_W, BLOCK_H, X_ELEMENTS, Y_ELEMENTS, VectorType>,
            cudaSharedMemBankSizeFourByte);

    d_convolveInnerShuffle2<T, RADIUS, BLOCK_W, BLOCK_H, X_ELEMENTS, Y_ELEMENTS, VectorType>
        <<<blocks, threads>>>(src, dst);

    //    d_convolveInnerShuffle3<T,12,BLOCK_W,BLOCK_H,4,Y_ELEMENTS,int4> <<<blocks, threads>>>(src,dst);
    CUDA_SYNC_CHECK_ERROR();
}

void convolveSinglePassSeparateOuterLinear(ImageView<float> src, ImageView<float> dst, Saiga::ArrayView<float> kernel,
                                           int radius)
{
    CHECK_CUDA_ERROR(
        cudaMemcpyToSymbol(d_Kernel, kernel.data(), kernel.size() * sizeof(float), 0, cudaMemcpyDeviceToDevice));
    switch (radius)
    {
        case 1:
            CUDA::convolveOuterLinear<float, 1>(src, dst);
            break;
        case 2:
            CUDA::convolveOuterLinear<float, 2>(src, dst);
            break;
        case 3:
            CUDA::convolveOuterLinear<float, 3>(src, dst);
            break;
        case 4:
            CUDA::convolveOuterLinear<float, 4>(src, dst);
            break;
        case 5:
            CUDA::convolveOuterLinear<float, 5>(src, dst);
            break;
        case 6:
            CUDA::convolveOuterLinear<float, 6>(src, dst);
            break;
        case 7:
            CUDA::convolveOuterLinear<float, 7>(src, dst);
            break;
        case 8:
            CUDA::convolveOuterLinear<float, 8>(src, dst);
            break;
        case 9:
            CUDA::convolveOuterLinear<float, 9>(src, dst);
            break;
    }
}

void convolveSinglePassSeparateOuterHalo(ImageView<float> src, ImageView<float> dst, Saiga::ArrayView<float> kernel,
                                         int radius)
{
    CHECK_CUDA_ERROR(
        cudaMemcpyToSymbol(d_Kernel, kernel.data(), kernel.size() * sizeof(float), 0, cudaMemcpyDeviceToDevice));
    switch (radius)
    {
        case 1:
            CUDA::convolveOuterHalo<float, 1>(src, dst);
            break;
        case 2:
            CUDA::convolveOuterHalo<float, 2>(src, dst);
            break;
        case 3:
            CUDA::convolveOuterHalo<float, 3>(src, dst);
            break;
        case 4:
            CUDA::convolveOuterHalo<float, 4>(src, dst);
            break;
        case 5:
            CUDA::convolveOuterHalo<float, 5>(src, dst);
            break;
        case 6:
            CUDA::convolveOuterHalo<float, 6>(src, dst);
            break;
        case 7:
            CUDA::convolveOuterHalo<float, 7>(src, dst);
            break;
        case 8:
            CUDA::convolveOuterHalo<float, 8>(src, dst);
            break;
    }
}

void convolveSinglePassSeparateInner(ImageView<float> src, ImageView<float> dst, Saiga::ArrayView<float> kernel,
                                     int radius)
{
    CHECK_CUDA_ERROR(
        cudaMemcpyToSymbol(d_Kernel, kernel.data(), kernel.size() * sizeof(float), 0, cudaMemcpyDeviceToDevice));
    switch (radius)
    {
        case 1:
            CUDA::convolveInner<float, 1>(src, dst);
            break;
        case 2:
            CUDA::convolveInner<float, 2>(src, dst);
            break;
        case 3:
            CUDA::convolveInner<float, 3>(src, dst);
            break;
        case 4:
            CUDA::convolveInner<float, 4>(src, dst);
            break;
        case 5:
            CUDA::convolveInner<float, 5>(src, dst);
            break;
        case 6:
            CUDA::convolveInner<float, 6>(src, dst);
            break;
        case 7:
            CUDA::convolveInner<float, 7>(src, dst);
            break;
        case 8:
            CUDA::convolveInner<float, 8>(src, dst);
            break;
    }
}


void convolveSinglePassSeparateInner75(ImageView<float> src, ImageView<float> dst, Saiga::ArrayView<float> kernel,
                                       int radius)
{
    CHECK_CUDA_ERROR(
        cudaMemcpyToSymbol(d_Kernel, kernel.data(), kernel.size() * sizeof(float), 0, cudaMemcpyDeviceToDevice));
    switch (radius)
    {
        case 1:
            CUDA::convolveInner<float, 1, true>(src, dst);
            break;
        case 2:
            CUDA::convolveInner<float, 2, true>(src, dst);
            break;
        case 3:
            CUDA::convolveInner<float, 3, true>(src, dst);
            break;
        case 4:
            CUDA::convolveInner<float, 4, true>(src, dst);
            break;
        case 5:
            CUDA::convolveInner<float, 5, true>(src, dst);
            break;
        case 6:
            CUDA::convolveInner<float, 6, true>(src, dst);
            break;
        case 7:
            CUDA::convolveInner<float, 7, true>(src, dst);
            break;
        case 8:
            CUDA::convolveInner<float, 8, true>(src, dst);
            break;
    }
}


void convolveSinglePassSeparateInnerShuffle(ImageView<float> src, ImageView<float> dst, Saiga::ArrayView<float> kernel,
                                            int radius)
{
    CHECK_CUDA_ERROR(
        cudaMemcpyToSymbol(d_Kernel, kernel.data(), kernel.size() * sizeof(float), 0, cudaMemcpyDeviceToDevice));
    switch (radius)
    {
        case 0:
            CUDA::convolveInnerShuffle<float, 0, int>(src, dst);
            break;

        case 1:
            CUDA::convolveInnerShuffle<float, 1, int>(src, dst);
            break;
        case 2:
            CUDA::convolveInnerShuffle<float, 2, int2>(src, dst);
            break;
        case 3:
            CUDA::convolveInnerShuffle<float, 3, int>(src, dst);
            break;
        case 4:
            CUDA::convolveInnerShuffle<float, 4, int4>(src, dst);
            break;

        case 5:
            CUDA::convolveInnerShuffle<float, 5, int>(src, dst);
            break;
        case 6:
            CUDA::convolveInnerShuffle<float, 6, int2>(src, dst);
            break;
        case 7:
            CUDA::convolveInnerShuffle<float, 7, int>(src, dst);
            break;
        case 8:
            CUDA::convolveInnerShuffle<float, 8, int4>(src, dst);
            break;

        case 9:
            CUDA::convolveInnerShuffle<float, 9, int>(src, dst);
            break;
        case 10:
            CUDA::convolveInnerShuffle<float, 10, int2>(src, dst);
            break;
        case 11:
            CUDA::convolveInnerShuffle<float, 11, int>(src, dst);
            break;
        case 12:
            CUDA::convolveInnerShuffle<float, 12, int2>(src, dst);
            break;

        case 13:
            CUDA::convolveInnerShuffle<float, 13, int>(src, dst);
            break;
        case 14:
            CUDA::convolveInnerShuffle<float, 14, int2>(src, dst);
            break;
        case 15:
            CUDA::convolveInnerShuffle<float, 15, int>(src, dst);
            break;
        case 16:
            CUDA::convolveInnerShuffle<float, 16, int4>(src, dst);
            break;

            //    case 17: CUDA::convolveInnerShuffle<float,17,int>(src,dst); break;
            //    case 18: CUDA::convolveInnerShuffle<float,18,int2>(src,dst); break;
            //    case 19: CUDA::convolveInnerShuffle<float,19,int>(src,dst); break;
        case 20:
            CUDA::convolveInnerShuffle<float, 20, int4>(src, dst);
            break;

            //    case 21: CUDA::convolveInnerShuffle<float,21,int>(src,dst); break;
            //    case 22: CUDA::convolveInnerShuffle<float,22,int2>(src,dst); break;
            //    case 23: CUDA::convolveInnerShuffle<float,23,int>(src,dst); break;
        case 24:
            CUDA::convolveInnerShuffle<float, 24, int4>(src, dst);
            break;
    }
}

}  // namespace CUDA
}  // namespace Saiga
