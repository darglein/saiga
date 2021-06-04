/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "floatingPoint.h"

#include "internal/noGraphicsAPI.h"

#include <fstream>
#include <iostream>
#include <string>


#if defined(__i386__) || defined(__x86_64__)
#    include "xmmintrin.h"

namespace Saiga
{
namespace FP
{
enum SSECSR : unsigned int
{
    //    Pnemonic	Bit Location	Description
    //    FZ	bit 15	Flush To Zero
    //    R+	bit 14	Round Positive
    //    R-	bit 13	Round Negative
    //    RZ	bits 13 and 14	Round To Zero
    //    RN	bits 13 and 14 are 0	Round To Nearest
    //    PM	bit 12	Precision Mask
    //    UM	bit 11	Underflow Mask
    //    OM	bit 10	Overflow Mask
    //    ZM	bit 9	Divide By Zero Mask
    //    DM	bit 8	Denormal Mask
    //    IM	bit 7	Invalid Operation Mask
    //    DAZ	bit 6	Denormals Are Zero
    //    PE	bit 5	Precision Flag
    //    UE	bit 4	Underflow Flag
    //    OE	bit 3	Overflow Flag
    //    ZE	bit 2	Divide By Zero Flag
    //    DE	bit 1	Denormal Flag
    //    IE	bit 0	Invalid Operation Flag
    FZ  = 15,
    RZ  = 14,
    RN  = 13,
    PM  = 12,
    UM  = 11,
    OM  = 10,
    ZM  = 9,
    DM  = 8,
    IM  = 7,
    DAZ = 6,
    PE  = 5,
    UE  = 4,
    OE  = 3,
    ZE  = 2,
    DE  = 1,
    IE  = 0,
};

void resetSSECSR()
{
    unsigned int csr = _mm_getcsr();
    csr &= ~(1 << DAZ);
    csr &= ~(1 << FZ);
    csr &= ~(1 << RZ);
    csr &= ~(1 << RN);
    _mm_setcsr(csr);
}


bool checkSSECSR()
{
    unsigned int csr = _mm_getcsr();
    //    for(int i = 0 ; i < 32 ; ++i){
    //        std::cout << i << " " << ((csr>>i)&1) << std::endl;
    //    }

    if (((csr >> FZ) & 1) != 0) return false;
    if (((csr >> DAZ) & 1) != 0) return false;
    if (((csr >> RZ) & 1) != 0) return false;
    if (((csr >> RN) & 1) != 0) return false;
    return true;
}

void breakSSECSR()
{
    unsigned int csr = _mm_getcsr();
    csr |= (1 << DAZ);
    csr |= (1 << RZ);
    _mm_setcsr(csr);
}

void printCPUInfo()
{
    // TODO: Windows
    std::string line;
    std::ifstream myfile("/proc/cpuinfo");
    if (myfile.is_open())
    {
        while (getline(myfile, line))
        {
            std::cout << line << std::endl;
        }
        myfile.close();
    }
}

}  // namespace FP
}  // namespace Saiga

#else


namespace Saiga
{
namespace FP
{
void resetSSECSR() {}

bool checkSSECSR()
{
    return true;
}

void printCPUInfo()
{
    // TODO: Windows
    std::string line;
    std::ifstream myfile("/proc/cpuinfo");
    if (myfile.is_open())
    {
        while (getline(myfile, line))
        {
            std::cout << line << std::endl;
        }
        myfile.close();
    }
}

}  // namespace FP
}  // namespace Saiga
#endif
