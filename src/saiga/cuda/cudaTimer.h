/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/time/timer.h"
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


    void startTimer() { events[queryFrontBuffer][0].record(stream); }
    float stopTimer(bool wait_back_buffer = false)
    {
        events[queryFrontBuffer][1].record(stream);
        time = -1;

        // Skip first iteration, because calling elapsed time on an events without a previous record
        // results in an error.
        if (Valid())
        {
            if (wait_back_buffer)
            {
                events[queryBackBuffer][1].synchronize();
            }
            time = CudaEvent::elapsedTime(events[queryBackBuffer][0], events[queryBackBuffer][1]);
        }
        swap();
        n++;
        return time;
    }

    float getTimeMS() { return time; }

    bool Valid() { return n > 1; }

    std::array<CudaEvent*, 2> BackBuffer() { return {&events[queryBackBuffer][0], &events[queryBackBuffer][1]}; }
    std::array<CudaEvent*, 2> FrontBuffer() { return {&events[queryFrontBuffer][0], &events[queryFrontBuffer][1]}; }

   private:
    CudaEvent events[2][2];
    cudaStream_t stream;
    int queryFrontBuffer = 0, queryBackBuffer = 1;
    float time = -1;
    int n      = 0;

    void swap() { std::swap(queryBackBuffer, queryFrontBuffer); }
};

// Measures the time relative to a base timer
// This is used to create frame timings plots.
// All timings are measured relative to the total frame timer.
class RelativeCudaTimer : public TimestampTimer
{
   public:
    void Start() { timer.startTimer(); }
    void Stop() { timer.stopTimer(); }
    std::pair<uint64_t, uint64_t> LastMeasurement()
    {
        SAIGA_ASSERT(base_timer);
        if (timer.Valid())
        {
            timer.FrontBuffer()[1]->synchronize();
            float begin_ms = CUDA::CudaEvent::elapsedTime(*base_timer->BackBuffer()[0], *timer.FrontBuffer()[0]);
            float end_ms   = CUDA::CudaEvent::elapsedTime(*base_timer->BackBuffer()[0], *timer.FrontBuffer()[1]);

            std::pair<uint64_t, uint64_t> m(begin_ms * (1000 * 1000), end_ms * (1000 * 1000));
            return m;
        }
        return {0, 0};
    }

    MultiFrameTimer timer;
    MultiFrameTimer* base_timer = nullptr;
};


}  // namespace CUDA
}  // namespace Saiga
