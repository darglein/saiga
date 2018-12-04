/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/cudaHelper.h"

#include <cusparse.h>

#include <cublas_v2.h>

namespace Saiga
{
namespace CUDA
{
SAIGA_GLOBAL extern cusparseHandle_t cusparseHandle;
SAIGA_GLOBAL extern cublasHandle_t cublashandle;

// Only initializes when not initialized yet.
SAIGA_GLOBAL extern void initBLASSPARSE();
SAIGA_GLOBAL extern void destroyBLASSPARSE();
SAIGA_GLOBAL extern bool isBLASSPARSEInitialized();

SAIGA_GLOBAL extern void runBLASSPARSETests();

}  // namespace CUDA
}  // namespace Saiga
