#include "saiga/util/crash.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/cusparseHelper.h"
#include "saiga/cuda/tests/test.h"
#include "saiga/tests/test.h"
#include "saiga/geometry/clipping.h"

using namespace Saiga;

int main(int argc, char *argv[]) {

    catchSegFaults();


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

//    return 0;

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
