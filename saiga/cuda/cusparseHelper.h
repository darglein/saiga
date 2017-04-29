#pragma once

#include "saiga/cuda/cudaHelper.h"

#include <cublas_v2.h>
#include <cusparse.h>



namespace CUDA{


SAIGA_GLOBAL extern cusparseHandle_t cusparseHandle;
SAIGA_GLOBAL extern cublasHandle_t cublashandle;


SAIGA_GLOBAL extern void initBLASSPARSE();
SAIGA_GLOBAL extern void destroyBLASSPARSE();
SAIGA_GLOBAL extern void runBLASSPARSETests();
}
