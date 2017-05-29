#include "saiga/util/crash.h"

#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/cusparseHelper.h"
#include "saiga/cuda/tests/test.h"
#include "saiga/tests/test.h"

int main(int argc, char *argv[]) {

    catchSegFaults();


    {
        //CUDA tests
        CUDA::initCUDA();
        CUDA::initBLASSPARSE();

        CUDA::occupancyTest();
        CUDA::randomAccessTest();
        CUDA::coalescedCopyTest();
        CUDA::dotTest();
        CUDA::recursionTest();


        CUDA::bandwidthTest();
        CUDA::scanTest();

        CUDA::reduceTest();

        CUDA::testCuda();
        CUDA::testThrust();

        CUDA::destroyBLASSPARSE();
        CUDA::destroyCUDA();
    }



    Tests::fpTest();

}
