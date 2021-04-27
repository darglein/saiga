/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/tests/test_helper.h"

#include <iomanip>

namespace Saiga
{
namespace CUDA
{
PerformanceTestHelper::PerformanceTestHelper(const std::string& name, size_t bytesReadWrite)
    : name(name), bytesReadWrite(bytesReadWrite)
{
    std::cout << ">>>> Starting Test " << name << ". " << std::endl;
    std::cout << ">>>> Total amount of memory reads and writes: " << bytesReadWrite << " bytes" << std::endl;
    using std::left;
    using std::setw;
    std::cout << setw(40) << left << "Name" << setw(15) << left << "Time (ms)" << setw(15) << left << "Bandwidth (GB/s)"
              << std::endl;
}

PerformanceTestHelper::~PerformanceTestHelper()
{
    std::cout << ">>>> Test " << name << " finished." << std::endl << std::endl;
}

void PerformanceTestHelper::addMeassurement(const std::string& name, float timeMS)
{
    using std::left;
    using std::setw;
    float bandWidth = bandwidth(timeMS);
    std::cout << setw(40) << left << name << setw(15) << left << timeMS << setw(15) << left << bandWidth << std::endl;
}

float PerformanceTestHelper::bandwidth(float timeMS) const
{
    float bandWidth = double(bytesReadWrite) / (timeMS / 1000.0) / (1000 * 1000 * 1000);
    return bandWidth;
}

void PerformanceTestHelper::updateBytes(size_t _bytesReadWrite)
{
    bytesReadWrite = _bytesReadWrite;
}

}  // namespace CUDA
}  // namespace Saiga
