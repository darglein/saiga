/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/geometry/clipping.h"
#include "saiga/core/tests/test.h"
#include "saiga/core/util/commandLineArguments.h"
#include "saiga/core/util/crash.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/CudaInfo.h"
#include "saiga/cuda/cusparseHelper.h"
#include "saiga/cuda/random.h"
#include "saiga/cuda/tests/test.h"
using namespace Saiga;

int main(int argc, char* argv[])
{
    catchSegFaults();

    CommandLineArguments arguments;

    arguments.arguments = std::vector<CommandLineArguments::CLA>{
        {"", 'f', "", "lasgld", true, false},
        {"", 'r', "", "lasgld", true, false},
        {"", 'g', "", "lasgld", true, false},
        {"asdf", 0, "djhlgsg", "lasgld", false, false},
    };


    struct asdf
    {
        std::string long_name;
        //        char short_name = 0;
        //        std::string defaultValue = "";
        bool flag;
    };

    asdf a = {"sdgl", true};


    //    CommandLineArguments::CLA asdf = {std::string(""),'f',std::string(""),true,false};

    arguments.parse(argc, argv);

    return 0;

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
