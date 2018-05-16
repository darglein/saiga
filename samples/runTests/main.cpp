/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/util/crash.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/cusparseHelper.h"
#include "saiga/cuda/tests/test.h"
#include "saiga/cuda/random.h"
#include "saiga/tests/test.h"
#include "saiga/geometry/clipping.h"

using namespace Saiga;

int main(int argc, char *argv[]) {

    catchSegFaults();


    /*
    Triangle trit;
    trit.a = vec3(-2,0,0);
    trit.b = vec3(2,0,0);
    trit.c = vec3(2,2,0);

    auto tri = Polygon::toPolygon(trit);

    AABB bb(vec3(-1,-1,-1),vec3(1,1,1));

    PolygonType res = Clipping::clipPolygonAABB(tri,bb);

    for(auto p : res){
        cout << p << endl;
    }

    cout << Clipping::clipTriAABBtoBox(trit,bb) << endl;
    */

//    return 0;

    {
        //CUDA tests
        CUDA::initCUDA();

        Saiga::CUDA::testCuda();
        Saiga::CUDA::testThrust();

        CUDA::initBLASSPARSE();

//        CUDA::imageProcessingTest();
//        CUDA::inverseTest();
//        CUDA::warpStrideLoopTest();
//        CUDA::convolutionTest();
//        CUDA::convolutionTest3x3();
//        CUDA::dotTest();
        CUDA::bandwidthTest();
        return 0;
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

        CUDA::destroyBLASSPARSE();
        CUDA::destroyCUDA();
    }



//    Tests::fpTest();

}
