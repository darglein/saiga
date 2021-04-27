/**
 * Copyright (c) 2021 Darius Rückert
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
PoseGraph Linear(int num_vertices, int num_connections)
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
    pg.sortEdges();
    return pg;
}

PoseGraph Circle(double radius, int num_vertices, int num_connections)
{
    PoseGraph pg;

    for (int i = 0; i < num_vertices; ++i)
    {
        double alpha = double(i) / num_vertices;

        Vec3 position(radius * sin(alpha * 2 * M_PI), 0, radius * cos(alpha * 2 * M_PI));

        SE3 t;
        t.so3()         = onb(position.normalized().cross(Vec3(0, -1, 0)), Vec3(0, 1, 0));
        t.translation() = position;


        PoseVertex v;
        v.constant = i == 0;
        v.SetPose(t);
        pg.vertices.push_back(v);
    }

    Vec3 offset = Vec3::Random() * 5.0;
    for (auto& v : pg.vertices)
    {
        v.T_w_i.se3().translation() += offset;
    }


    for (int i = 0; i < num_vertices; ++i)
    {
        for (int j = 1; j <= num_connections; ++j)
        {
            pg.AddVertexEdge(i, (i + j) % pg.vertices.size(), 1.0);
        }
    }
    pg.sortEdges();

    return pg;
}

PoseGraph CircleWithDrift(double radius, int num_vertices, int num_connections, double sigma, double sigma_scale)
{
    auto pg = Circle(radius, num_vertices, num_connections);
    pg.edges.clear();
    auto pg_cpy = pg;

    pg.fixScale = sigma_scale == 0;

    double scale_drift = 1 + sigma_scale;


    // vertices
    for (int i = 1; i < (int)pg.vertices.size(); ++i)
    {
        SE3 drift = Sophus::se3_expd(Sophus::Vector6d::Random() * sigma);
        DSim3 drift_sim(drift, 1);

        auto T_i_prev = drift_sim * (pg_cpy.vertices[i].Sim3Pose().inverse() * pg_cpy.vertices[i - 1].Sim3Pose());
        T_i_prev.se3().translation() *= pow(scale_drift, i);
        pg.vertices[i].T_w_i = pg.vertices[i - 1].T_w_i * T_i_prev.inverse();
    }


    // Add all relative edges (without the loop edges)
    for (int i = 0; i < num_vertices; ++i)
    {
        for (int j = 1; j <= num_connections && i + j < num_vertices; ++j)
        {
            pg.AddVertexEdge(i, i + j, 1.0);
        }
    }

    {
        // transform last pose  to origin
        SE3 drift = Sophus::se3_expd(Sophus::Vector6d::Random() * sigma);
        DSim3 drift_sim(drift, 1);

        auto T_last_first = drift_sim * (pg_cpy.vertices.back().Sim3Pose().inverse() * pg_cpy.vertices[0].Sim3Pose());
        pg.vertices.back().T_w_i    = pg.vertices[0].T_w_i * T_last_first.inverse();
        pg.vertices.back().constant = true;

        pg.vertices.back().T_w_i.scale() = 1.0 / pow(scale_drift, num_vertices - 1);
    }

    pg.sortEdges();

    return pg;
}

}  // namespace SyntheticPoseGraph
}  // namespace Saiga
