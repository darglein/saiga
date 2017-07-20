/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>
#include <iostream>

namespace Saiga {
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
}
