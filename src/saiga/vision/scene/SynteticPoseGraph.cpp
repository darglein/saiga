/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "SynteticPoseGraph.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/vision/util/Random.h"
namespace Saiga
{
namespace SyntheticPoseGraph
{
PoseGraph CreateLinearPoseGraph(int num_vertices, int num_connections)
{
    PoseGraph pg;

    SE3 current = SE3();
    for (int i = 0; i < num_vertices; ++i)
    {
        PoseVertex v;
        v.SetPose(current);

        v.constant = i == 0;
        pg.vertices.push_back(v);
        current.translation() += Vec3(0, 0, 1);
    }



    for (int i = 0; i < num_vertices; ++i)
    {
        for (int j = 1; j <= num_connections && i + j < num_vertices; ++j)
        {
            pg.AddVertexEdge(i, i + j, 1.0);
        }
    }
    return pg;
}

}  // namespace SyntheticPoseGraph
}  // namespace Saiga
