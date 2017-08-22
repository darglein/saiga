/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/imageProcessing/difference.h"

namespace Saiga {
namespace CUDA {


template<int BLOCK_W, int BLOCK_H, int ROWS_PER_THREAD = 1>
__global__
static void d_subtract(ImageView<float> src1, ImageView<float> src2, ImageView<float> dst, int h)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x*BLOCK_W + tx;
    int y = blockIdx.y*BLOCK_H + ty;


    if(x >= dst.width)
        return;

#pragma unroll
    for(int i = 0; i < ROWS_PER_THREAD; ++i, y+=h){
        if(y < dst.height){
            dst(x,y) = src1(x,y) - src2(x,y);
        }
    }

}


void subtract(ImageView<float> src1, ImageView<float> src2, ImageView<float> dst){
    SAIGA_ASSERT(src1.width == dst.width && src1.height == dst.height);

    const int ROWS_PER_THREAD = 2;
    const int BLOCK_W = 128;
    const int BLOCK_H = 1;
    int w = dst.width;
    int h = iDivUp(dst.height,ROWS_PER_THREAD);
    dim3 blocks(iDivUp(w, BLOCK_W), iDivUp(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);
    d_subtract<BLOCK_W,BLOCK_H,ROWS_PER_THREAD> <<<blocks, threads>>>(src1,src2,dst,h);
}


template<typename T, int BLOCK_W, int BLOCK_H, int ROWS_PER_THREAD = 1>
__global__ void d_subtractMulti(
        ImageArrayView<float> src, ImageArrayView<float> dst, int h)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x*BLOCK_W + tx;
    int ys = blockIdx.y*BLOCK_H + ty;

    int height = dst.imgStart.height;

    if(x >= src.imgStart.width)
        return;


    T lastVals[ROWS_PER_THREAD];


    int y = ys;
#pragma unroll
    for(int i = 0; i < ROWS_PER_THREAD; ++i, y+=h){
        if(y < height){
            lastVals[i] = src[0](x,y);
        }
    }

    for(int i = 0; i < dst.n; ++i){
        int y = ys;
#pragma unroll
        for(int j = 0; j < ROWS_PER_THREAD; ++j, y+=h){
            if(y < height){
                T nextVal = src[i+1](x,y);
                dst[i](x,y) = nextVal - lastVals[j];
                lastVals[j] = nextVal;
            }
        }
    }
}

void subtractMulti(ImageArrayView<float> src, ImageArrayView<float> dst){
    //    SAIGA_ASSERT(src1.width == dst.width && src1.height == dst.height);

    SAIGA_ASSERT(src.n == dst.n + 1);
    const int ROWS_PER_THREAD = 1;
    const int BLOCK_W = 128;
    const int BLOCK_H = 1;
    int w = dst[0].width;
    int h = iDivUp(dst[0].height,ROWS_PER_THREAD);
    dim3 blocks(iDivUp(w, BLOCK_W), iDivUp(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);
    d_subtractMulti<float,BLOCK_W,BLOCK_H,ROWS_PER_THREAD> <<<blocks, threads>>>(src,dst,h);
}


}
}


