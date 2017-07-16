#pragma once

#include "saiga/cuda/cudaHelper.h"

namespace Saiga {
namespace CUDA {


class SAIGA_GLOBAL CudaScopedTimer{
public:
    float& time;
    cudaEvent_t start, stop;
    CudaScopedTimer(float& time);
    ~CudaScopedTimer();
};



class SAIGA_GLOBAL CudaScopedTimerPrint {
public:
    std::string name;
    cudaEvent_t start, stop;
    CudaScopedTimerPrint(const std::string &name);
    ~CudaScopedTimerPrint();
};


}
}
