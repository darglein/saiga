#include "saiga/cuda/cusparseHelper.h"

#include "saiga/util/assert.h"

using std::cout;
using std::endl;

namespace CUDA {

cusparseHandle_t cusparseHandle = 0;
cublasHandle_t cublashandle = 0;

void initBLASSPARSE(){
    cublasCreate(&cublashandle);
    cusparseCreate(&cusparseHandle);
}

void destroyBLASSPARSE(){
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublashandle);
}

extern void testCuBLAS();
extern void testCuSparse();
void runBLASSPARSETests(){
    testCuBLAS();
    testCuSparse();
}

}
