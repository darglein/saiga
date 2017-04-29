#pragma once

#include "saiga/cuda/cudaHelper.h"

namespace CUDA {

class SAIGA_GLOBAL CudaScopedTimerPrint{
public:
    std::string name;
    cudaEvent_t start, stop;
    CudaScopedTimerPrint(const std::string &name);
    ~CudaScopedTimerPrint();
};


}
