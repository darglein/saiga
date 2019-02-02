/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/cudaHelper.h"

namespace Saiga
{
namespace CUDA
{
/**
 * A simple c++ wrapper for cuda events
 *
 * Usage Example:
 *
 * CudaEvent start, stop;
 * start.record();
 * // ...
 * stop.record();
 * stop.synchronize();
 * float time = CudaEvent::elapsedTime(start,stop);
 *
 */
class SAIGA_CUDA_API CudaEvent
{
   public:
    cudaEvent_t event;


    CudaEvent() { cudaEventCreate(&event); }

    ~CudaEvent() { cudaEventDestroy(event); }

    // Place this event into the command stream
    void record(cudaStream_t stream = 0) { cudaEventRecord(event, stream); }

    // Wait until this event is completed
    void synchronize() { cudaEventSynchronize(event); }

    // Test if the event is completed (returns immediately)
    bool isCompleted() { return cudaEventQuery(event) == cudaSuccess; }



    static float elapsedTime(CudaEvent& first, CudaEvent& second)
    {
        float time;
        cudaEventElapsedTime(&time, first, second);
        return time;
    }

    operator cudaEvent_t() const { return event; }
};

}  // namespace CUDA
}  // namespace Saiga
