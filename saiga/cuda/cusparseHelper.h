#pragma once

#include "saiga/cuda/cudaHelper.h"

#include <cublas_v2.h>
#include <cusparse.h>



namespace CUDA{


SAIGA_GLOBAL extern cusparseHandle_t cusparseHandle;
SAIGA_GLOBAL extern cublasHandle_t cublashandle;

//Only initializes when not initialized yet.
SAIGA_GLOBAL extern void initBLASSPARSE();
SAIGA_GLOBAL extern void destroyBLASSPARSE();
SAIGA_GLOBAL extern bool isBLASSPARSEInitialized();

SAIGA_GLOBAL extern void runBLASSPARSETests();

}
