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
#include "saiga/time/performanceMeasure.h"

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
//    const unsigned int t = ty * TILE_W + tx;

    const unsigned int x_tile = blockIdx.x * TILE_W;
    const unsigned int y_tile = blockIdx.y * TILE_H;

    const unsigned int x = x_tile + tx;
    const unsigned int y = y_tile + ty;

    float sum = 0;
    for(int dy = -1; dy <= 1; ++dy){
        for(int dx = -1; dx <= 1; ++dx){
            int gx = x + dx;
            int gy = y + dy;
            src.clampToEdge(gy,gx);
            sum += src(gy,gx);
        }
    }
    dst(y,x) = sum;
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
        src.clampToEdge(gy,gx);
        sbuffer[y][x] = src(gy,gx);
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
    dst(y,x) = sum;
}



template<unsigned int TILE_W, unsigned int TILE_H>
__global__ static
void d_convolve3x3Shared2(ImageView<float> src, ImageView<float> dst)
{
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int t = ty * TILE_W + tx;
    const unsigned int warp_lane = t / 32;
    const unsigned int lane_id = t & 31;

    const unsigned int x_tile = blockIdx.x * TILE_W;
    const unsigned int y_tile = blockIdx.y * TILE_H * 2;

    const unsigned int x = x_tile + tx;
    const unsigned int y = y_tile + ty;

    __shared__ float sbuffer[TILE_H * 2 + 2][TILE_W + 2];


    //copy main data
    for(int i = 0; i < 2; ++i)
        sbuffer[ty + i * TILE_H + 1][tx + 1]  = src.clampedRead(y + i * TILE_H,x);

    //top halo
    if(warp_lane == 0){
        sbuffer[0][lane_id + 1]  = src.clampedRead(y_tile - 1,x_tile + lane_id);
    }

    //bottom
    if(warp_lane == 1){
        sbuffer[TILE_H * 2 + 1][lane_id + 1]  = src.clampedRead(y_tile + TILE_H * 2,x_tile + lane_id);
    }

    //left
    if(warp_lane == 2){
        sbuffer[lane_id + 1][0]  = src.clampedRead(y_tile + lane_id,x_tile - 1);
    }

    //right
    if(warp_lane == 3){
        sbuffer[lane_id + 1][TILE_W + 1]  = src.clampedRead(y_tile + lane_id,x_tile + TILE_W);
    }

    //corners
    if(warp_lane == 4){
        if(lane_id == 0) sbuffer[0][0]  = src.clampedRead(y_tile - 1,x_tile - 1);
        if(lane_id == 1) sbuffer[0][TILE_W + 1]  = src.clampedRead(y_tile - 1,x_tile + TILE_W);
        if(lane_id == 2) sbuffer[TILE_H * 2 + 1][0]  = src.clampedRead(y_tile + TILE_H * 2,x_tile - 1);
        if(lane_id == 3) sbuffer[TILE_H * 2 + 1][TILE_W + 1]  = src.clampedRead(y_tile + TILE_H * 2,x_tile + TILE_W);
    }

    __syncthreads();

    for(int i = 0; i < 2; ++i)
    {
        float sum = 0;
#if 1
        for(int dy = -1; dy <= 1; ++dy){
            for(int dx = -1; dx <= 1; ++dx){
                int x = tx + 1 + dx;
                int y = ty + 1 + dy + i * TILE_H;
                sum += sbuffer[y][x];
            }
        }
#endif

        dst(y + i * TILE_H,x) = sum;
    }
}

template<unsigned int TILE_W, unsigned int TILE_H, unsigned int Y_ELEMENTS = 2>
__global__ static
void d_convolve3x3Shared3(ImageView<float> src, ImageView<float> dst)
{
    const unsigned int TILE_H2 = TILE_H * Y_ELEMENTS;
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    int x_tile = blockIdx.x * (TILE_W - 2) - 1;
    int y_tile = blockIdx.y * (TILE_H2 - 2) - 1;

    int x = x_tile + tx;
    int y = y_tile + ty;

    __shared__ float sbuffer[TILE_H2][TILE_W];

    //copy main data
    for(int i = 0; i < Y_ELEMENTS; ++i)
        sbuffer[ty + i * TILE_H][tx]  = src.clampedRead(y + i * TILE_H,x);

    __syncthreads();


    for(int i = 0; i < Y_ELEMENTS; ++i)
    {
        int gx = x;
        int gy = y + i * TILE_H;
        int lx = tx;
        int ly = ty + i * TILE_H;

        if(!dst.inImage(gy,gx))
            continue;

        if(lx > 0 && lx < TILE_W - 1 && ly > 0 && ly < TILE_H2 - 1)
        {
            float sum = 0;
            for(int dy = -1; dy <= 1; ++dy){
                for(int dx = -1; dx <= 1; ++dx){
                    sum += sbuffer[ly + dy][lx + dx];
                }
            }
            dst(gy,gx) = sum;
        }
    }
}



template<unsigned int TILE_W, unsigned int TILE_H>
__global__ static
void d_copySharedSync(ImageView<float> src, ImageView<float> dst)
{
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
//    const unsigned int t = ty * TILE_W + tx;
    const unsigned int x_tile = blockIdx.x * TILE_W;
    const unsigned int y_tile = blockIdx.y * TILE_H;
    const unsigned int x = x_tile + tx;
    const unsigned int y = y_tile + ty;

    __shared__ float sbuffer[TILE_H][TILE_W];

    sbuffer[ty][tx]  = src(y,x);
    __syncthreads();
    dst(y,x) = sbuffer[ty][tx];
}


template<unsigned int TILE_W, unsigned int TILE_H, unsigned int Y_ELEMENTS = 2>
__global__ static
void d_copySharedSync2(ImageView<float> src, ImageView<float> dst)
{
    const unsigned int TILE_H2 = TILE_H * Y_ELEMENTS;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
//    const unsigned int t = ty * TILE_W + tx;
    const unsigned int x_tile = blockIdx.x * TILE_W;
    const unsigned int y_tile = blockIdx.y * TILE_H2;
    const unsigned int x = x_tile + tx;
    const unsigned int y = y_tile + ty;

    __shared__ float sbuffer[TILE_H2][TILE_W];

    for(int i = 0; i < Y_ELEMENTS; ++i)
        sbuffer[ty + i * TILE_H][tx]  = src.clampedRead(y + i * TILE_H,x);

    __syncthreads();

    for(int i = 0; i < Y_ELEMENTS; ++i)
        dst.clampedWrite(y + i * TILE_H,x, sbuffer[ty + i * TILE_H][tx]);
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
                h_imgSrc(y,x) = (rand()%3) - 1;
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
                        h_imgSrc.clampToEdge(gy,gx);
                        sum += h_imgSrc(gy,gx);
                    }
                }
                h_imgDst(y,x) = sum;
            }
        }
        h_ref = h_dest;
    }

    int its = 50;

    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            const int TILE_W = 128;
            const int TILE_H = 1;
            dim3 blocks(
                        Saiga::iDivUp(w, TILE_W),
                        Saiga::iDivUp(h, TILE_H),
                        1
                        );
            dim3 threads(TILE_W,TILE_H);
            d_convolve3x3<TILE_W,TILE_H><<<blocks,threads>>>(imgSrc,imgDst);
        });
        pth.addMeassurement("d_convolve3x3", st.median);
        h_dest = dest;
        SAIGA_ASSERT(h_dest == h_ref);
    }

    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            const int TILE_W = 32;
            const int TILE_H = 16;
            dim3 blocks(
                        Saiga::iDivUp(w, TILE_W),
                        Saiga::iDivUp(h, TILE_H),
                        1
                        );
            dim3 threads(TILE_W,TILE_H);
            d_convolve3x3Shared<TILE_W,TILE_H><<<blocks,threads>>>(imgSrc,imgDst);
        });
        pth.addMeassurement("d_convolve3x3Shared", st.median);
        h_dest = dest;
        SAIGA_ASSERT(h_dest == h_ref);
    }

    {
        dest = tmp;
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            const int TILE_W = 32;
            const int TILE_H = 16;
            dim3 blocks(
                        Saiga::iDivUp(w, TILE_W),
                        Saiga::iDivUp(h, TILE_H * 2),
                        1
                        );
            dim3 threads(TILE_W,TILE_H);
            d_convolve3x3Shared2<TILE_W,TILE_H><<<blocks,threads>>>(imgSrc,imgDst);
        });
        pth.addMeassurement("d_convolve3x3Shared2", st.median);
        h_dest = dest;
        SAIGA_ASSERT(h_dest == h_ref);
    }

    {
        dest = tmp;
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            const int TILE_W = 32;
            const int TILE_H = 8;
            const int Y_ELEMENTS = 4;
            dim3 blocks(
                        Saiga::iDivUp(w, TILE_W - 2),
                        Saiga::iDivUp(h, TILE_H * Y_ELEMENTS - 2),
                        1
                        );
            dim3 threads(TILE_W,TILE_H);
            d_convolve3x3Shared3<TILE_W,TILE_H,Y_ELEMENTS><<<blocks,threads>>>(imgSrc,imgDst);
        });
        pth.addMeassurement("d_convolve3x3Shared3", st.median);
        h_dest = dest;
        SAIGA_ASSERT(h_dest == h_ref);
    }


    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            const int TILE_W = 32;
            const int TILE_H = 16;
            dim3 blocks(
                        Saiga::iDivUp(w, TILE_W),
                        Saiga::iDivUp(h, TILE_H),
                        1
                        );
            dim3 threads(TILE_W,TILE_H);
            d_copySharedSync<TILE_W,TILE_H><<<blocks,threads>>>(imgSrc,imgDst);
        });
        pth.addMeassurement("d_copySharedSync", st.median);
        h_dest = dest;
        //        SAIGA_ASSERT(h_dest == h_ref);
    }

    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            const int TILE_W = 32;
            const int TILE_H = 16;
            dim3 blocks(
                        Saiga::iDivUp(w, TILE_W),
                        Saiga::iDivUp(h, TILE_H * 2),
                        1
                        );
            dim3 threads(TILE_W,TILE_H);
            d_copySharedSync2<TILE_W,TILE_H><<<blocks,threads>>>(imgSrc,imgDst);
        });
        pth.addMeassurement("d_copySharedSync2", st.median);
        h_dest = dest;
        //        SAIGA_ASSERT(h_dest == h_ref);
    }

    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            cudaMemcpy(thrust::raw_pointer_cast(dest.data()),thrust::raw_pointer_cast(src.data()),N * sizeof(int),cudaMemcpyDeviceToDevice);
        });
        pth.addMeassurement("cudaMemcpy", st.median);
    }

    CUDA_SYNC_CHECK_ERROR();

}

}
}
