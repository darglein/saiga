/**
 * Copyright (c) 2021 Darius Rückert
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
SAIGA_CUDA_API extern cusparseHandle_t cusparseHandle;
SAIGA_CUDA_API extern cublasHandle_t cublashandle;

// Only initializes when not initialized yet.
SAIGA_CUDA_API extern void initBLASSPARSE();
SAIGA_CUDA_API extern void destroyBLASSPARSE();
SAIGA_CUDA_API extern bool isBLASSPARSEInitialized();

SAIGA_CUDA_API extern void runBLASSPARSETests();

}  // namespace CUDA
}  // namespace Saiga
