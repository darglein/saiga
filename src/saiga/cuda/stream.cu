/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "stream.h"

#ifndef _WIN32
#include <nvToolsExtCudaRt.h>
#endif




namespace Saiga
{
namespace CUDA
{
CudaStream::CudaStream()
{
    cudaStreamCreate(&stream);
}

CudaStream::~CudaStream()
{
    cudaStreamDestroy(stream);
}

void CudaStream::waitForEvent(cudaEvent_t event)
{
    cudaStreamWaitEvent(stream, event, 0);
}

void CudaStream::synchronize()
{
    cudaStreamSynchronize(stream);
}


cudaStream_t CudaStream::legacyStream()
{
    return cudaStreamLegacy;
}

cudaStream_t CudaStream::perThreadStream()
{
    return cudaStreamPerThread;
}

Saiga::CUDA::CudaStream::operator cudaStream_t() const
{
    return stream;
}

void CudaStream::setName(const std::string& name) { 
#ifndef _WIN32
	nvtxNameCudaStreamA(stream, name.c_str());
#endif
}

}  // namespace CUDA
}  // namespace Saiga
