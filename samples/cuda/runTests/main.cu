/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/geometry/clipping.h"
#include "saiga/core/tests/test.h"
#include "saiga/core/util/commandLineArguments.h"
#include "saiga/core/util/crash.h"
#include "saiga/cuda/CudaInfo.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/cusparseHelper.h"
#include "saiga/cuda/random.h"
#include "saiga/cuda/tests/test.h"
using namespace Saiga;

int main(int argc, char* argv[])
{
    catchSegFaults();
    {
        // CUDA tests
        CUDA::initCUDA();

        Saiga::CUDA::testCuda();
        Saiga::CUDA::testThrust();


        CUDA::randomTest();

        //        return 0;
        //        CUDA::occupancyTest();
        //        CUDA::randomAccessTest();
        //        CUDA::coalescedCopyTest();
        //        CUDA::recursionTest();


        CUDA::scanTest();

        CUDA::reduceTest();

        //        CUDA::testCuda();
        //        CUDA::testThrust();

        //        CUDA::destroyBLASSPARSE();
        CUDA::destroyCUDA();
    }



    //    Tests::fpTest();
}
