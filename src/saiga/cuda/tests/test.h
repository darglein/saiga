/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

namespace Saiga
{
namespace CUDA
{
SAIGA_CUDA_API extern void occupancyTest();
SAIGA_CUDA_API extern void coalescedCopyTest();
SAIGA_CUDA_API extern void dotTest();
SAIGA_CUDA_API extern void recursionTest();

SAIGA_CUDA_API extern void scanTest();

SAIGA_CUDA_API extern void reduceTest();
SAIGA_CUDA_API extern void warpStrideLoopTest();

SAIGA_CUDA_API extern void convolutionTest();
SAIGA_CUDA_API extern void convolutionTest3x3();

SAIGA_CUDA_API extern void imageProcessingTest();

SAIGA_CUDA_API extern void testCuda();
SAIGA_CUDA_API extern void testThrust();


}  // namespace CUDA
}  // namespace Saiga
