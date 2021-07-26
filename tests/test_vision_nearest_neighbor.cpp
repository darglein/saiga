/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/geometry/all.h"
#include "saiga/core/math/Eigen_Compile_Checker.h"
#include "saiga/core/math/all.h"
#include "saiga/core/util/Align.h"
#include "saiga/core/util/ConsoleColor.h"
#include "saiga/core/util/table.h"
#include "saiga/vision/reconstruction/MeshToTSDF.h"

#include "gtest/gtest.h"

namespace Saiga
{
TEST(NearestNeighbor, PointToMesh)
{
    // Compute the distance of a point to the mesh
    std::vector<Triangle> mesh;

    for (int i = 0; i < 10000; ++i)
    {
        Triangle t;
        t.a = Random::MatrixUniform<vec3>();
        t.b = t.a + Random::MatrixGauss<vec3>(0, 0.1);
        t.c = t.a + Random::MatrixGauss<vec3>(0, 0.1);
        mesh.push_back(t);
    }


    std::vector<vec3> points;
    for (int i = 0; i < 1000; ++i)
    {
        points.push_back(Random::MatrixUniform<vec3>(-2, 2));
    }
    std::vector<float> result_bf(points.size()), result_bvh(points.size());

    AccelerationStructure::ObjectMedianBVH bvh(mesh);


    {
        SAIGA_BLOCK_TIMER("Brute Force");
        for (int i = 0; i < points.size(); ++i)
        {
            result_bf[i] = Distance(mesh, points[i]);
        }
    }

    {
        SAIGA_BLOCK_TIMER("BVH");
        for (int i = 0; i < points.size(); ++i)
        {
            result_bvh[i] = bvh.ClosestPoint(points[i]).first;
        }
    }
    EXPECT_EQ(result_bf, result_bvh);
}


}  // namespace Saiga
