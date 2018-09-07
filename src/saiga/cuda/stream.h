/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/cudaHelper.h"


namespace Saiga {
namespace CUDA{

/**
 * A simple c++ wrapper for cuda streams
 *
 * Usage:
 *
 * Saiga::CUDA::CudaStream stream;
 * cudaMemcpyAsync(d_slice.data(),h_slice.data(),slizeSize,cudaMemcpyHostToDevice,stream);
 *
 */
class SAIGA_GLOBAL CudaStream
{
public:
    cudaStream_t stream;


    CudaStream()
    {
        cudaStreamCreate(&stream);
    }

    ~CudaStream()
    {
        cudaStreamDestroy(stream);
    }


    operator cudaStream_t() const { return stream; }

    static cudaStream_t legacyStream()
    {
        return cudaStreamLegacy;
    }

    static cudaStream_t perThreadStream()
    {
        return cudaStreamPerThread;
    }
};

}
}
