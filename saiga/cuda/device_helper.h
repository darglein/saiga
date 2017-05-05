#pragma once

#include "cudaHelper.h"


namespace CUDA{



//inline __device__ int get_grid_size()
//{
//    return blockDim.x * gridDim.x;
//}


//inline __device__ int get_global_index()
//{
//    unsigned int globalThreadNum = blockIdx.x * blockDim.x + threadIdx.x;
//    return globalThreadNum;
//}
//inline __device__ int get_global_index_2D()
//{
//    int threadid = threadIdx.y * blockDim.x + threadIdx.x;
//    int blockNumInGrid = blockIdx.x + gridDim.x  * blockIdx.y;
//    int block_width = blockDim.x * blockDim.y;
//    int globalThreadNum = blockNumInGrid * block_width + threadid;
//    return globalThreadNum;
//}




}
