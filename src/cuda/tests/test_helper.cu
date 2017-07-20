/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/tests/test_helper.h"
#include <iomanip>

namespace Saiga {
namespace CUDA{

using std::cout;
using std::endl;

PerformanceTestHelper::PerformanceTestHelper(const std::string& name, size_t bytesReadWrite)
    : name(name), bytesReadWrite(bytesReadWrite){
    cout << ">>>> Starting Test " << name << ". " << endl;
    cout << ">>>> Total amount of memory reads and writes: " << bytesReadWrite << " bytes" << endl;
    using std::setw;
    using std::left;
    std::cout
            << setw(40) << left << "Name"
            << setw(15) << left << "Time (ms)"
            << setw(15) << left << "Bandwidth (GB/s)"
            << std::endl;
}

PerformanceTestHelper::~PerformanceTestHelper(){
    cout << ">>>> Test " << name << " finished." << endl << endl;
}

void PerformanceTestHelper::addMeassurement(const std::string& name, float timeMS){
    using std::setw;
    using std::left;
    float bandWidth = double(bytesReadWrite) / (timeMS / 1000.0) / (1000*1000*1000);
    std::cout
            << setw(40) << left << name
            << setw(15) << left << timeMS
            << setw(15) << left << bandWidth
            << endl ;
}

}
}
