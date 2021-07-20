/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/cuda/imageProcessing/NppiHelper.h"
//
#include "saiga/core/camera/all.h"
#include "saiga/core/framework/framework.h"
#include "saiga/core/geometry/all.h"
#include "saiga/core/image/all.h"
#include "saiga/core/math/Eigen_Compile_Checker.h"
#include "saiga/core/model/all.h"
#include "saiga/cuda/CudaInfo.h"
#include "saiga/cuda/imageProcessing/image.h"
#include "saiga/cuda/thrust_helper.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"


namespace Saiga
{
//__global__ static void addFive(Ray ray, ArrayView<Triangle> triangles, ArrayView<float> output)
//{
//    int tid = blockDim.x * blockIdx.x + threadIdx.x;
//    if (tid >= triangles.size()) return;
//    auto t = triangles[tid];
//}

TEST(BVH, IntersectionRayTriangle)
{
    initSaigaSampleNoWindow();

    int w = 500;
    int h = 500;

    PerspectiveCamera camera;
    camera.setProj(60.0f, 1, 0.1f, 50.0f, true);
    camera.setView(vec3(0, 3, 6), vec3(0, 0, 0), vec3(0, 1, 0));

    auto mesh = UnifiedModel("teapot.obj").mesh[0];



    auto triangles = mesh.TriangleSoup();


    AccelerationStructure::ObjectMedianBVH bf(triangles);

    std::cout << "Num triangles = " << triangles.size() << std::endl;


    TemplatedImage<ucvec3> img(w, h);

    {
#pragma omp parallel for
        for (int i = 0; i < h; ++i)
        {
            for (int j = 0; j < w; ++j)
            {
                img(i, j) = ucvec3(255, 0, 0);


                Ray ray = camera.PixelRay(vec2(j, i), w, h, false);

                auto inter = bf.getClosest(ray);
                if (inter && !inter.backFace)
                {
                    img(i, j) = ucvec3(0, 255, 0);
                }
            }
        }
    }
    img.save("raytracing.png");
}


}  // namespace Saiga
