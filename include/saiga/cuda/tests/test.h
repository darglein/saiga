/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>

namespace Saiga {
namespace CUDA {

SAIGA_GLOBAL extern void occupancyTest();
SAIGA_GLOBAL extern void randomAccessTest();
SAIGA_GLOBAL extern void coalescedCopyTest();
SAIGA_GLOBAL extern void dotTest();
SAIGA_GLOBAL extern void recursionTest();


SAIGA_GLOBAL extern void bandwidthTest();
SAIGA_GLOBAL extern void scanTest();

SAIGA_GLOBAL extern void reduceTest();
SAIGA_GLOBAL extern void warpStrideLoopTest();

SAIGA_GLOBAL extern void convolutionTest();
SAIGA_GLOBAL extern void imageProcessingTest();

SAIGA_GLOBAL extern void testCuda();
SAIGA_GLOBAL extern void testThrust();

}
}
