/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */



#include "saiga/core/geometry/all.h"
#include "saiga/core/time/all.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/reconstruction/MarchingCubes.h"

#include "gtest/gtest.h"

#include <fstream>
#include <unsupported/Eigen/CXX11/Tensor>

#include "compare_numbers.h"
namespace Saiga
{
#ifndef WIN32

class PoissonTest
{
   public:
    PoissonTest()
    {
        long w = 100;

        normal_grid.resize(w, w, w);
        normal_grid.setZero();

        divergence_grid.resize(w, w, w);
        divergence_grid.setZero();


        sdf_grid.resize(w, w, w);
        sdf_grid.setZero();


        ui.resize(w, w, w);

        uo.resize(w, w, w);
    }

    void smoothNormals()
    {
        for (int i = 0; i < 10000; ++i)
        {
            // Sample point on sphere
            Vec3 n = Random::sphericalRand(r);
            Vec3 p = (n + c) * (1.0 / voxel_size);

            // Compute grid point
            ivec3 cell_index = (p).array().round().cast<int>();
            normal_grid(cell_index(2), cell_index(1), cell_index(0)) += n.normalized();
        }
    }

    void computeDivergence()
    {
        // Compute only on inner voxels for now
        for (int i = 1; i < normal_grid.dimension(0) - 1; ++i)
        {
            for (int j = 1; j < normal_grid.dimension(1) - 1; ++j)
            {
                for (int k = 1; k < normal_grid.dimension(2) - 1; ++k)
                {
                    // central difference
                    auto dx = normal_grid(i, j, k + 1).x() - normal_grid(i, j, k - 1).x();
                    auto dy = normal_grid(i, j + 1, k).y() - normal_grid(i, j - 1, k).y();
                    auto dz = normal_grid(i + 1, j, k).z() - normal_grid(i - 1, j, k).z();

                    //                    auto dx = normal_grid(i, j, k + 1).z() - normal_grid(i, j, k - 1).z();
                    //                    auto dy = normal_grid(i, j + 1, k).y() - normal_grid(i, j - 1, k).y();
                    //                    auto dz = normal_grid(i + 1, j, k).x() - normal_grid(i - 1, j, k).x();

                    divergence_grid(i, j, k) = (dx + dy + dz) / (2.0 * voxel_size);
                }
            }
        }
    }



    void jacobiStep()
    {
        double voxel_size = 1;
        const float dx2   = voxel_size * voxel_size;
        const float dy2   = voxel_size * voxel_size;
        const float dz2   = voxel_size * voxel_size;

        // compute constant coeffs
        float ne = 2.f * (dx2 * dy2 + dy2 * dz2 + dz2 * dx2);
        float ax = dy2 * dz2 / ne;
        float ay = dz2 * dx2 / ne;
        float az = dx2 * dy2 / ne;
        float af = dx2 * dy2 * dz2 / ne;

        // Compute only on inner voxels for now
        for (int i = 1; i < ui.dimension(0) - 1; ++i)
        {
            for (int j = 1; j < ui.dimension(1) - 1; ++j)
            {
                for (int k = 1; k < ui.dimension(2) - 1; ++k)
                {
                    auto center = divergence_grid(i, j, k);

                    auto sx = ax * (ui(i, j, k - 1) + ui(i, j, k + 1));
                    auto sy = ay * (ui(i, j - 1, k) + ui(i, j + 1, k));
                    auto sz = az * (ui(i - 1, j, k) + ui(i + 1, j, k));

                    uo(i, j, k) = sx + sy + sz - af * center;
                }
            }
        }
    }

    double computeChange()
    {
        double sum = 0;
        for (int i = 1; i < ui.dimension(0) - 1; ++i)
        {
            for (int j = 1; j < ui.dimension(1) - 1; ++j)
            {
                for (int k = 1; k < ui.dimension(2) - 1; ++k)
                {
                    auto diff = ui(i, j, k) - uo(i, j, k);
                    sum += diff * diff;
                }
            }
        }
        return sum;
    }

    void solveJacobi()
    {
        ui.setZero();

        for (int i = 0; i < 10; ++i)
        {
            uo.setZero();
            jacobiStep();

            std::cout << "It " << i << " change: " << computeChange() << std::endl;
            ui = uo;
        }

        sdf_grid = uo;

        Eigen::array<long, 3> offsets = {50, 50, 0};
        Eigen::array<long, 3> extents = {1, 1, 100};

        Eigen::Tensor<double, 3> slice = sdf_grid.slice(offsets, extents).eval();

        std::cout << slice.dimensions() << std::endl;
        std::cout << slice << std::endl;
    }

    void extractSurface()
    {
        // ==================================================================
        // extract surface
        std::vector<std::array<vec3, 3>> triangle_soup;
        for (int i = 0; i < 100 - 1; ++i)
        {
            for (int j = 0; j < 100 - 1; ++j)
            {
                for (int k = 0; k < 100 - 1; ++k)
                {
                    std::array<std::pair<vec3, float>, 8> cell;
                    cell[0] = {vec3(k, j, i), sdf_grid(i, j, k)};
                    cell[1] = {vec3(k + 1, j, i), sdf_grid(i, j, k + 1)};
                    cell[2] = {vec3(k + 1, j, i + 1), sdf_grid(i + 1, j, k + 1)};
                    cell[3] = {vec3(k, j, i + 1), sdf_grid(i + 1, j, k)};
                    cell[4] = {vec3(k, j + 1, i), sdf_grid(i, j + 1, k)};
                    cell[5] = {vec3(k + 1, j + 1, i), sdf_grid(i, j + 1, k + 1)};
                    cell[6] = {vec3(k + 1, j + 1, i + 1), sdf_grid(i + 1, j + 1, k + 1)};
                    cell[7] = {vec3(k, j + 1, i + 1), sdf_grid(i + 1, j + 1, k)};

                    auto [triangles, count] = MarchingCubes(cell, 0);
                    for (int c = 0; c < count; ++c)
                    {
                        triangle_soup.push_back(triangles[c]);
                    }
                }
            }
        }

        TriangleMesh<VertexNC, uint32_t> mesh;
        for (auto& t : triangle_soup)
        {
            VertexNC tri[3];
            for (int i = 0; i < 3; ++i)
            {
                tri[i].position.head<3>() = t[i].cast<float>();
                tri[i].color              = vec4(1, 1, 1, 1);
            }
            mesh.addTriangle(tri);
        }

        std::ofstream strm("poisson.off");
        saveMeshOff(mesh, strm);
    }


    Eigen::Tensor<double, 3> ui;
    Eigen::Tensor<double, 3> uo;

    Eigen::Tensor<Vec3, 3> normal_grid;
    Eigen::Tensor<double, 3> divergence_grid;
    Eigen::Tensor<double, 3> sdf_grid;


    double voxel_size = 0.01;

    // sphere
    Vec3 c   = Vec3(0.5, 0.5, 0.5);
    double r = 0.3;
};


TEST(PoissonSurfaceReconstruction, Grid)
{
    PoissonTest test;
    test.smoothNormals();
    test.computeDivergence();
    test.solveJacobi();
    test.extractSurface();
}


#endif
}  // namespace Saiga
