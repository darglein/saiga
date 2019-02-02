/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/cuda.h"
#include "saiga/cuda/event.h"

namespace Saiga
{
namespace CUDA
{
/**
 * A c++ class for meassuring CUDA command times
 *
 * Usage Example:
 *
 * {
 *      CudaScopedTimerPrint tim("helloKernel");
 *      helloKernel<<<1,1>>>();
 * }
 *
 */
class SAIGA_CUDA_API CudaScopedTimer
{
   public:
    CudaScopedTimer(float& time, cudaStream_t stream = 0);
    ~CudaScopedTimer();

   private:
    float& time;
    CudaEvent start, stop;
    cudaStream_t stream;
};



class SAIGA_CUDA_API CudaScopedTimerPrint
{
   public:
    CudaScopedTimerPrint(const std::string& name, cudaStream_t stream = 0);
    ~CudaScopedTimerPrint();

   private:
    std::string name;
    CudaEvent start, stop;
    cudaStream_t stream;
};


}  // namespace CUDA
}  // namespace Saiga
