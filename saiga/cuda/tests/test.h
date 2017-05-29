#pragma once

#include <saiga/config.h>


namespace CUDA {

SAIGA_GLOBAL extern void occupancyTest();
SAIGA_GLOBAL extern void randomAccessTest();
SAIGA_GLOBAL extern void coalescedCopyTest();
SAIGA_GLOBAL extern void dotTest();
SAIGA_GLOBAL extern void recursionTest();


SAIGA_GLOBAL extern void bandwidthTest();
SAIGA_GLOBAL extern void scanTest();

SAIGA_GLOBAL extern void reduceTest();


SAIGA_GLOBAL extern void testCuda();
SAIGA_GLOBAL extern void testThrust();


}


