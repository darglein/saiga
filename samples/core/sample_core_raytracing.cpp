/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/Core.h"

using namespace Saiga;

int main(int argc, char* args[])
{
    initSaigaSampleNoWindow();

    int w = 500;
    int h = 500;

    PerspectiveCamera camera;
    camera.setProj(60.0f, 1, 0.1f, 50.0f, true);
    camera.setView(vec3(0, 3, 6), vec3(0, 0, 0), vec3(0, 1, 0));

    //    ObjModelLoader loader("teapot.obj");



    auto mesh = UnifiedModel("teapot.obj").mesh[0];
    //    loader.toTriangleMesh(mesh);


    auto triangles = mesh.TriangleSoup();


    AccelerationStructure::ObjectMedianBVH bf(triangles);

    std::cout << "Num triangles = " << triangles.size() << std::endl;


    TemplatedImage<ucvec3> img(w, h);

    {
        SAIGA_BLOCK_TIMER();
#pragma omp parallel for
        for (int i = 0; i < h; ++i)
        {
            for (int j = 0; j < w; ++j)
            {
                img(i, j) = ucvec3(255, 0, 0);

                //                vec3 dir = camera.inverseprojectToWorldSpace(vec2(j, i), 1, w, h);
                //                Ray ray(normalize(dir), camera.getPosition());
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
