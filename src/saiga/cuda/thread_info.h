/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/common.h"

namespace Saiga {
namespace CUDA{

/**
 * A simple helper struct that can be used at the beginning of each kernel to compute some usefull variables.
 * Don't worry about variables that you are not going going to use, because they are optimized away :).
 *
 * If you know the number of threads per block at compile time pass it as the first template argument.
 * LOCAL_WARP_SIZE can be used for example in partial warp reductions.
 * LOCAL_WARP_SIZE must be one of these values: 1,2,4,8,16,32
 *
 * Note: In some libraries you will find the __mul24 for these index computations, but from cuda 8 api:
 *  "_[u]mul24 are legacy intrinsic functions that have no longer any reason to be used"
 *
 * Note: It is important to use unsigend ints here, because then the compiler can replace expensive integer divisions
 * with fast bit shifts. The same counts when accessing these variables. Either use "auto" or "unsigend int".
 */
template<unsigned int THREADS_PER_BLOCK = 0, unsigned int LOCAL_WARP_SIZE=WARP_SIZE>
struct ThreadInfo{

    unsigned int local_thread_id;   // local thread id in the block PTX: %tid.x
    unsigned int thread_id;         // global thread index
    unsigned int block_id;          // id of this block PTX: %ctaid.x
    unsigned int lane_id;           // thread index within the warp
    unsigned int warp_id;           // global warp index

    unsigned int threads_per_block;
    unsigned int warp_lane;         // warp index within the block
    unsigned int num_blocks;        //number of blocks in that grid PTX: %nctaid.x
    unsigned int num_warps_block;   // total number of active warps
    unsigned int num_warps;         // total number of active warps
    unsigned int grid_size;         //total number of threads in the grid

    __device__
    ThreadInfo(){
        if(THREADS_PER_BLOCK > 0){
            threads_per_block = THREADS_PER_BLOCK;
        }else{
            threads_per_block = blockDim.x;
        }
        local_thread_id = threadIdx.x;
        block_id        = blockIdx.x;
        num_blocks      = gridDim.x;

        //Note: Beccause of this line LOCAL_WARP_SIZE must be a power of 2
        lane_id         = local_thread_id & (LOCAL_WARP_SIZE-1);
        thread_id       = threads_per_block * block_id + local_thread_id;

        grid_size       = num_blocks * threads_per_block;

        warp_id         = thread_id   / LOCAL_WARP_SIZE;
        warp_lane       = local_thread_id / LOCAL_WARP_SIZE;
        num_warps_block = threads_per_block / LOCAL_WARP_SIZE;
        num_warps       = num_warps_block * num_blocks;
    }
};

}
}
