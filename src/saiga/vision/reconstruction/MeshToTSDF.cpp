/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "MeshToTSDF.h"

#include "saiga/core/geometry/all.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/discreteProbabilityDistribution.h"
#include "saiga/vision/util/Random.h"

#include "MarchingCubes.h"
#include "fstream"
namespace Saiga
{
std::vector<vec3> MeshToPointCloud(const std::vector<Triangle>& triangles, int N)
{
    std::vector<float> areas;
    for (auto& t : triangles)
    {
        areas.push_back(t.Area());
    }

    DiscreteProbabilityDistribution<float> dis(areas);

    std::vector<vec3> points;

    for (int i = 0; i < N; ++i)
    {
        auto t = dis.sample();

        auto p = triangles[t].RandomPointOnSurface();
        points.push_back(p);
    }
    return points;
}

float Distance(const std::vector<Triangle>& triangles, const vec3& p)
{
    float dis = std::numeric_limits<float>::infinity();
    for (auto& t : triangles)
    {
        dis = std::min(dis, t.Distance(p));
    }
    return dis;
}


std::shared_ptr<SparseTSDF> MeshToTSDF(const std::vector<Triangle>& triangles, float voxel_size, int r)
{
    std::shared_ptr<SparseTSDF> tsdf = std::make_shared<SparseTSDF>(voxel_size);

    {
        auto points = MeshToPointCloud(triangles, 10000000);

        ProgressBar bar(std::cout, "M2TSDF Allocate", points.size());
        for (auto& p : points)
        {
            tsdf->AllocateAroundPoint(p, r);
            bar.addProgress(1);
        }

        for (auto& t : triangles)
        {
            tsdf->AllocateAroundPoint(t.a);
            tsdf->AllocateAroundPoint(t.b);
            tsdf->AllocateAroundPoint(t.c);
        }
    }

    {
        ProgressBar bar(std::cout, "M2TSDF Compute Unsigned Distance", tsdf->current_blocks);
#pragma omp parallel for
        for (int i = 0; i < tsdf->current_blocks; ++i)
        {
            auto& b = tsdf->blocks[i];
            for (int i = 0; i < tsdf->VOXEL_BLOCK_SIZE; ++i)
            {
                for (int j = 0; j < tsdf->VOXEL_BLOCK_SIZE; ++j)
                {
                    for (int k = 0; k < tsdf->VOXEL_BLOCK_SIZE; ++k)
                    {
                        vec3 global_pos = tsdf->GlobalPosition(b.index, i, j, k);

                        float dis;
                        {
                            dis = Distance(triangles, global_pos);
                        }
                        b.data[i][j][k].distance = -dis;
                        b.data[i][j][k].weight   = 1;
                    }
                }
            }
            bar.addProgress(1);
        }
    }


    auto triangles_cpy = triangles;
    //    for (auto& t : triangles_cpy)
    //    {
    //        t.ScaleUniform(1.000125);
    //    }
    AccelerationStructure::ObjectMedianBVH bvh(triangles_cpy);



    {
        std::vector<Vec3> directions = {Vec3(1, 0, 0),  Vec3(0, 1, 0),  Vec3(0, 0, 1),
                                        Vec3(-1, 0, 0), Vec3(0, -1, 0), Vec3(0, 0, -1)};

        for (int i = 0; i < 100; ++i)
        {
            Vec3 d   = Random::MatrixUniform<Vec3>(-1, 1);
            double l = d.norm();

            if (l > 0.01)
            {
                directions.push_back(d / l);
            }
        }

        ProgressBar bar(std::cout, "M2TSDF Compute Sign", tsdf->current_blocks);
#pragma omp parallel for
        for (int i = 0; i < tsdf->current_blocks; ++i)
        {
            auto& b = tsdf->blocks[i];
            auto id = b.index;
            for (int i = 0; i < tsdf->VOXEL_BLOCK_SIZE; ++i)
            {
                for (int j = 0; j < tsdf->VOXEL_BLOCK_SIZE; ++j)
                {
                    for (int k = 0; k < tsdf->VOXEL_BLOCK_SIZE; ++k)
                    {
                        Vec3 global_pos = tsdf->GlobalPosition(id, i, j, k).cast<double>();
                        auto& cell      = b.data[i][j][k];

                        for (auto& d : directions)
                        {
                            Ray r;
                            r.direction = d.cast<float>();
                            r.origin    = global_pos.cast<float>();
                            if (bvh.getAll(r).empty())
                            {
                                cell.distance = std::abs(cell.distance);
                                break;
                            }
                        }
                    }
                }
            }
            bar.addProgress(1);
        }
    }


    return tsdf;
}


}  // namespace Saiga
