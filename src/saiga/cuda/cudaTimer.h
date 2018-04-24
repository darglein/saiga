/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/cudaHelper.h"

namespace Saiga {
namespace CUDA {


class SAIGA_GLOBAL CudaScopedTimer{
    float& time;
    cudaEvent_t start, stop;
public:
    CudaScopedTimer(float& time);
    ~CudaScopedTimer();
};



class SAIGA_GLOBAL CudaScopedTimerPrint {
    std::string name;
    cudaEvent_t start, stop;
public:
    CudaScopedTimerPrint(const std::string &name);
    ~CudaScopedTimerPrint();
};


}
}
