/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/time/performanceMeasure.h"
#include "saiga/core/time/timer.h"

#include <iostream>

namespace Saiga
{
namespace CUDA
{
class SAIGA_CUDA_API PerformanceTestHelper
{
    std::string name;
    size_t bytesReadWrite;

   public:
    PerformanceTestHelper(const std::string& name, size_t bytesReadWrite);
    ~PerformanceTestHelper();
    void addMeassurement(const std::string& name, float timeMS);

    float bandwidth(float timeMS) const;

    // usefull when the size changes but you don't want to create a new testhelper
    void updateBytes(size_t bytesReadWrite);
};

}  // namespace CUDA
}  // namespace Saiga
