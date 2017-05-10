#pragma once

#include <saiga/config.h>
#include <iostream>

namespace CUDA {

class SAIGA_GLOBAL PerformanceTestHelper{

    std::string name;
    size_t bytesReadWrite;
public:
    PerformanceTestHelper(const std::string& name, size_t bytesReadWrite);
    ~PerformanceTestHelper();
    void addMeassurement(const std::string& name, float timeMS);

};


}


