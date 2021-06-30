/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "MeshToTSDF.h"

#include "saiga/core/geometry/all.h"
#include "saiga/core/geometry/kdtree.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/discreteProbabilityDistribution.h"
#include "saiga/vision/util/Random.h"

#include "MarchingCubes.h"
#include "fstream"

#include <algorithm>
namespace Saiga
{
SimplePointCloud RandomSampleMesh(const std::vector<Triangle>& triangles, const std::vector<float>& weights, int N)
{
    std::vector<double> sample_prob(triangles.size());

#pragma omp parallel for
    for (int i = 0; i < triangles.size(); ++i)
    {
        float prob     = triangles[i].Area() * weights[i];
        sample_prob[i] = prob;
    }

    DiscreteProbabilityDistribution<double> dis(sample_prob);

    SimplePointCloud points(N);

#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        auto t    = dis.sample();
        auto& tri = triangles[t];

        SimpleVertex v;
        v.normal         = tri.normal();
        v.bary           = tri.RandomBarycentric();
        v.position       = tri.InterpolateBarycentric(v.bary);
        v.triangle_index = t;
        v.radius         = sample_prob[t];
        points[i]        = v;
    }
    return points;
}

SimplePointCloud ReducePointsPoissonDisc(const SimplePointCloud& mesh_points, float radius)
{
    std::vector<int> used(mesh_points.size(), 0);


    std::vector<vec3> positions;
    positions.reserve(mesh_points.size());
    for (auto m : mesh_points)
    {
        positions.push_back(m.position);
    }

    KDTree<3, vec3> tree(positions);
#pragma omp parallel for
    for (int i = 0; i < mesh_points.size(); ++i)
    {
        auto& p = mesh_points[i].position;
        auto ps = tree.RadiusSearch(p, mesh_points[i].radius);

        bool found = false;
        for (auto pi : ps)
        {
            if (used[pi])
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            used[i] = true;
        }
    }

    SimplePointCloud result;
    result.reserve(mesh_points.size());

    for (int i = 0; i < mesh_points.size(); ++i)
    {
        if (used[i])
        {
            result.push_back(mesh_points[i]);
        }
    }
    return result;
}

SimplePointCloud MeshToPointCloudPoissonDisc2(const std::vector<Triangle>& triangles, const std::vector<float>& weights,
                                              int max_samples, float radius)
{
    std::vector<int> num_samples_per_triangle(triangles.size(), 0);

    {
        ScopedTimerPrintLine tim("Compute initial");
        double total_area = 0;
        for (int i = 0; i < triangles.size(); ++i)
        {
            total_area += triangles[i].Area() * weights[i];
        }
        for (int i = 0; i < triangles.size(); ++i)
        {
            double ratio                = triangles[i].Area() * weights[i] / total_area;
            num_samples_per_triangle[i] = std::round(ratio * max_samples);
        }
    }


    std::vector<SimplePointCloud> samples_per_triangle(triangles.size());

    {
        ScopedTimerPrintLine tim("Sample + Local Reduce");
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < triangles.size(); ++i)
        {
            int n     = num_samples_per_triangle[i];
            auto& tri = triangles[i];

            SimplePointCloud points(n);
            for (int j = 0; j < n; ++j)
            {
                SimpleVertex v;
                v.normal         = tri.normal();
                v.bary           = tri.RandomBarycentric();
                v.position       = tri.InterpolateBarycentric(v.bary);
                v.triangle_index = i;
                v.radius         = (1.f / (weights[i] + 1e-10)) * radius;
                points[j]        = v;
            }

            samples_per_triangle[i] = ReducePointsPoissonDisc(points, radius);
        }
    }

    SimplePointCloud result;


    {
        ScopedTimerPrintLine tim("Insert");
        result.reserve(max_samples);
        for (auto& points : samples_per_triangle)
        {
            result.insert(result.end(), points.begin(), points.end());
        }
    }

    {
        ScopedTimerPrintLine tim("Shuffle");
        std::shuffle(result.begin(), result.end(), Random::generator());
    }

    ScopedTimerPrintLine tim("Global Reduce");
    return ReducePointsPoissonDisc(result, radius);
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
        std::vector<float> weights(triangles.size(), 1);
        auto points = RandomSampleMesh(triangles, weights, 1000000);
        ProgressBar bar(std::cout, "M2TSDF Allocate V", triangles.size());
        for (auto& t : triangles)
        {
            tsdf->AllocateAroundPoint(t.a, 1);
            tsdf->AllocateAroundPoint(t.b, 1);
            tsdf->AllocateAroundPoint(t.c, 1);
            bar.addProgress(1);
        }
        ProgressBar bar2(std::cout, "M2TSDF Allocate P1", points.size());
        for (auto& p : points)
        {
            tsdf->AllocateAroundPoint(p.position, 1);
            bar2.addProgress(1);
        }
    }

    {
        for (int i = 0; i < r - 1; ++i)
        {
            std::vector<ivec3> current_blocks;
            for (auto bi = 0; bi < tsdf->current_blocks; ++bi)
            {
                current_blocks.push_back(tsdf->blocks[bi].index);
            }
            ProgressBar bar(std::cout, "M2TSDF Expand", current_blocks.size());

            for (auto block_id : current_blocks)
            {
                for (int z = -1; z <= 1; ++z)
                {
                    for (int y = -1; y <= 1; ++y)
                    {
                        for (int x = -1; x <= 1; ++x)
                        {
                            ivec3 current_id = ivec3(x, y, z) + block_id;
                            tsdf->InsertBlock(current_id);
                        }
                    }
                }
                bar.addProgress(1);
            }
        }
        std::cout << "Allocated blocks = " << tsdf->current_blocks << std::endl;
    }



    AccelerationStructure::ObjectMedianBVH bvh(triangles);
    bvh.triangle_epsilon = 0;
    {
        ProgressBar bar(std::cout, "M2TSDF Compute Unsigned Distance", tsdf->current_blocks);
#pragma omp parallel for schedule(dynamic)
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

                        //                        float dis  = Distance(triangles, global_pos);
                        float dis = bvh.ClosestPoint(global_pos).first;

                        // SAIGA_ASSERT(dis == dis2);
                        //                        if (dis != dis2)
                        //                        {
                        //                            std::cout << ":o " << dis << " " << dis2 << std::endl;
                        //                        }

                        b.data[i][j][k].distance = -dis;
                        b.data[i][j][k].weight   = 1;
                    }
                }
            }
            bar.addProgress(1);
        }
    }


    //    auto triangles_cpy = triangles;
    //    for (auto& t : triangles_cpy)
    //    {
    //        t.ScaleUniform(1.000125);
    //    }



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
#pragma omp parallel for schedule(dynamic)
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
