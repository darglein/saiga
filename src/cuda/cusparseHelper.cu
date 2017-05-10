#include "saiga/cuda/cusparseHelper.h"

#include "saiga/util/assert.h"

using std::cout;
using std::endl;

namespace CUDA {

cusparseHandle_t cusparseHandle = 0;
cublasHandle_t cublashandle = 0;

void initBLASSPARSE(){
    if(!isBLASSPARSEInitialized()){
        cublasCreate(&cublashandle);
        cusparseCreate(&cusparseHandle);
    }
}

void destroyBLASSPARSE(){
    if(isBLASSPARSEInitialized()){
        cusparseDestroy(cusparseHandle);
        cublasDestroy(cublashandle);
        cusparseHandle = 0;
        cublashandle = 0;
    }
}

bool isBLASSPARSEInitialized(){
    return cusparseHandle != 0;
}

extern void testCuBLAS();
extern void testCuSparse();
void runBLASSPARSETests(){
    testCuBLAS();
    testCuSparse();
}

}
