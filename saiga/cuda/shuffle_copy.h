#ifndef SHUFFLE_COPY_H
#define SHUFFLE_COPY_H

#include "shfl_helper.h"

namespace CUDA {



template<int G, int SIZE, typename VectorType=int4>
__device__ inline
void loadShuffle(VectorType* globalStart, VectorType* localStart, int lane, int globalOffset, int Nvectors){
    int bytesPerCycle = G * sizeof(VectorType);
    int cycles = SIZE / bytesPerCycle;
    //    int vectorsPerCycle = bytesPerCycle / sizeof(VectorType);
    int vectorsPerElement = SIZE / sizeof(VectorType);

    VectorType* global = reinterpret_cast<VectorType*>(globalStart);
    VectorType* local = reinterpret_cast<VectorType*>(localStart);

    VectorType l[G];
    VectorType tmp;

    for(int g = 0 ; g < G ; ++g){

        for(int c = 0 ; c < cycles; ++c){
            auto globalIdx = globalOffset + lane + c * G + g * vectorsPerElement;

            if(globalIdx < Nvectors){
                //                printf("read %d, %d \n",globalIdx,lane);
                tmp = global[globalIdx];
            }

            //broadcast loaded value to all threads in this warp
            for(int s = 0 ; s < G ; ++s){
                l[s] = CUDA::shfl(tmp,s,G);
            }

            //this thread now has the correct value
            if(lane == g){
                for(int s = 0 ; s < G ; ++s){
                    local[s + c * G] = l[s];
                }
            }
        }

    }
}


template<int G, int SIZE, typename VectorType=int4>
__device__ inline
void storeShuffle(VectorType* globalStart, VectorType* localStart, int lane, int globalOffset, int Nvectors){
    int bytesPerCycle = G * sizeof(VectorType);
    int cycles = SIZE / bytesPerCycle;
    //    int vectorsPerCycle = bytesPerCycle / sizeof(VectorType);
    int vectorsPerElement = SIZE / sizeof(VectorType);

    VectorType* global = reinterpret_cast<VectorType*>(globalStart);
    VectorType* local = reinterpret_cast<VectorType*>(localStart);


    VectorType l[G];
    VectorType tmp;

    for(int g = 0 ; g < G ; ++g){

        for(int c = 0 ; c < cycles; ++c){

            //this thread now has the correct value
            if(lane == g){
                for(int s = 0 ; s < G ; ++s){
                    l[s] = local[s + c * G];
                }
            }

            //broadcast loaded value to all threads in this warp
            for(int s = 0 ; s < G ; ++s){
                l[s] = CUDA::shfl(l[s],g,G);
            }

            //this loop does the same as "tmp = l[lane]", but with static indexing
            //so all the locals can be held in registers
            //https://stackoverflow.com/questions/44117704/why-is-local-memory-used-in-this-simple-loop
            for(int i = 0 ; i < G ; ++i){
                if(i <= lane)
                    tmp = l[i];
            }

            auto globalIdx = globalOffset + lane + c * G + g * vectorsPerElement;
            if(globalIdx < Nvectors)
                global[globalIdx] = tmp;
        }

    }
}


}

#endif // SHUFFLE_COPY_H
