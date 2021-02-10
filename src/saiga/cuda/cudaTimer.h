/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/cuda.h"
#include "saiga/cuda/event.h"

#include <iostream>
#include <string>

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
class SAIGA_CUDA_API ScopedTimer
{
   public:
    ScopedTimer(float& time, cudaStream_t stream = 0) : time(time), stream(stream) { start.record(stream); }
    ~ScopedTimer()
    {
        stop.record(stream);
        stop.synchronize();

        time = CudaEvent::elapsedTime(start, stop);
    }

   private:
    float& time;
    CudaEvent start, stop;
    cudaStream_t stream;
};



class SAIGA_CUDA_API ScopedTimerPrint
{
   public:
    ScopedTimerPrint(const std::string& name, cudaStream_t stream = 0) : name(name), stream(stream)
    {
        start.record(stream);
    }
    ~ScopedTimerPrint()
    {
        stop.record(stream);
        stop.synchronize();

        auto time = CudaEvent::elapsedTime(start, stop);

        std::cout << name << " : " << time << "ms." << std::endl;
    }

   private:
    std::string name;
    CudaEvent start, stop;
    cudaStream_t stream;
};

/**
 * Asynchronous CUDA GPU timer.
 *
 * Meassuers the time of the CUDA calls between startTimer and stopTimer.
 * These calls do not empty the GPU command queue and return immediately.
 *
 */
class SAIGA_CUDA_API MultiFrameTimer
{
   public:
    MultiFrameTimer(cudaStream_t stream = 0) : stream(stream) {}
    ~MultiFrameTimer() {}

    void startTimer() { events[queryBackBuffer][0].record(stream); }
    float stopTimer()
    {
        events[queryBackBuffer][1].record(stream);


        time = -1;

        // Skip first iteration, because calling elapsed time on an events without a previous record
        // results in an error.
        if (n > 0)
        {
            time = CudaEvent::elapsedTime(events[queryFrontBuffer][0], events[queryFrontBuffer][1]);
        }
        swap();
        n++;
        return time;
    }

    float getTimeMS() { return time; }

   private:
    CudaEvent events[2][2];
    cudaStream_t stream;
    int queryBackBuffer = 0, queryFrontBuffer = 1;
    float time = -1;
    int n      = 0;

    void swap() { std::swap(queryBackBuffer, queryFrontBuffer); }
};



}  // namespace CUDA
}  // namespace Saiga
