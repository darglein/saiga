/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/cudaTimer.h"
#include "saiga/util/assert.h"

namespace Saiga {
namespace CUDA {

CudaScopedTimer::CudaScopedTimer(float& time, cudaStream_t stream)
    : time(time), stream(stream)
{
    start.record(stream);
}

CudaScopedTimer::~CudaScopedTimer()
{
    stop.record(stream);
    stop.synchronize();

    time = CudaEvent::elapsedTime(start,stop);
}


CudaScopedTimerPrint::CudaScopedTimerPrint(const std::string &name, cudaStream_t stream)
    : name(name), stream(stream)
{
      start.record(stream);
}

CudaScopedTimerPrint::~CudaScopedTimerPrint()
{
    stop.record(stream);
    stop.synchronize();

    auto time = CudaEvent::elapsedTime(start,stop);

    std::cout << name << " : " << time << "ms." << std::endl;
}

}
}
