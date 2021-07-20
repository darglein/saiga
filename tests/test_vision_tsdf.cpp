/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/core/Core.h"
#include "saiga/core/model/model_loader_ply.h"
#include "saiga/vision/reconstruction/MarchingCubes.h"
#include "saiga/vision/reconstruction/SparseTSDF.h"
#include "saiga/vision/reconstruction/VoxelFusion.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"

namespace Saiga
{
std::shared_ptr<SparseTSDF> CreateSphereTSDF(vec3 position, float radius, float voxel_size, float truncation_distance)
{
    std::shared_ptr<SparseTSDF> tsdf = std::make_shared<SparseTSDF>(voxel_size);

    float dis    = radius + truncation_distance;
    vec3 min_pos = position - vec3::Ones() * dis;
    vec3 max_pos = position + vec3::Ones() * dis;

    auto bmin = tsdf->GetBlockIndex(min_pos);
    auto bmax = tsdf->GetBlockIndex(max_pos);

    for (int z = bmin.z(); z <= bmax.z(); ++z)
    {
        for (int y = bmin.y(); y <= bmax.y(); ++y)
        {
            for (int x = bmin.x(); x <= bmax.x(); ++x)
            {
                ivec3 bid(x, y, z);
                tsdf->InsertBlock(bid);
            }
        }
    }

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
                    vec3 global_pos = tsdf->GlobalPosition(id, i, j, k);
                    auto& cell      = b.data[i][j][k];

                    float d       = (global_pos - position).norm() - radius;
                    cell.distance = d;
                    cell.weight   = 1;
                }
            }
        }
    }



    return tsdf;
}

class TSDFTest
{
   public:
    TSDFTest()
    {
        tsdf = CreateSphereTSDF(sphere.pos, sphere.r, 0.05, 1);
        std::cout << *tsdf << std::endl;
    }
    Sphere sphere = Sphere(vec3(0, 0, 0), 0.5);
    std::shared_ptr<SparseTSDF> tsdf;
    UnifiedMesh mesh;

    BlockSparseGrid<int, 4> test;
    BlockSparseGrid<int, 8> test2;
};

std::unique_ptr<TSDFTest> test;

TEST(TSDF, Create)
{
    test = std::make_unique<TSDFTest>();



    auto tris  = test->tsdf->ExtractSurface(0, 4, 0, 1, true);
    test->mesh = test->tsdf->CreateMesh(tris, false);

    {
        std::ofstream strm("tsdf_sphere.off");
        // saveMeshOff(test->mesh, strm);
    }

//    PLYLoader::save("tsdf_sphere.ply", test->mesh);

    //    exit(0);
    // Test if mesh vertices are on the sphere
//    for (auto v : test->mesh.vertices)
//    {
//        EXPECT_NEAR(test->sphere.sdf(v.position.head<3>()), 0, 0.001);
//    }
}

TEST(TSDF, InsertRemoveBlock)
{
    {
        SparseTSDF tsdf(1, 1000, 1000);
        tsdf.InsertBlock({0, 0, 0});
        EXPECT_TRUE(tsdf.GetBlock({0, 0, 0}));
        EXPECT_EQ(tsdf.Bounds(), iRect<3>(ivec3(0, 0, 0), ivec3(1, 1, 1)));
        tsdf.InsertBlock({2, 1, -1});
        EXPECT_EQ(tsdf.Bounds(), iRect<3>(ivec3(0, 0, -1), ivec3(3, 2, 1)));

        EXPECT_TRUE(tsdf.GetBlock({2, 1, -1}));
        tsdf.EraseBlockWithHole({2, 1, -1}, tsdf.H({2, 1, -1}));
        EXPECT_FALSE(tsdf.GetBlock({2, 1, -1}));
    }

    {
        SparseTSDF tsdf(1, 1, 1);
        tsdf.InsertBlock({0, 0, 0});
        tsdf.InsertBlock({2, 1, -1});
        tsdf.InsertBlock({4, 1, -1});
        tsdf.InsertBlock({1, 1, -1});

        EXPECT_TRUE(tsdf.EraseBlock({2, 1, -1}));
        EXPECT_EQ(tsdf.Bounds(), iRect<3>(ivec3(0, 0, -1), ivec3(5, 2, 1)));
        EXPECT_FALSE(tsdf.GetBlock({2, 1, -1}));

        EXPECT_TRUE(tsdf.EraseBlock({4, 1, -1}));
        EXPECT_EQ(tsdf.Bounds(), iRect<3>(ivec3(0, 0, -1), ivec3(2, 2, 1)));
        EXPECT_FALSE(tsdf.GetBlock({4, 1, -1}));

        EXPECT_EQ(tsdf.current_blocks, tsdf.NumBlocksInRect(tsdf.Bounds()));

        EXPECT_EQ(tsdf.current_blocks, 2);
        tsdf.EraseEmptyBlocks();
        EXPECT_EQ(tsdf.current_blocks, 0);
    }
}

TEST(TSDF, Crop)
{
    Random::setSeed(394765346);

    for (int k = 0; k < 100; ++k)
    {
        SparseTSDF tsdf(1, 500, 500);

        for (int i = 0; i < 1000; ++i)
        {
            ivec3 r(Random::uniformInt(-10, 11), Random::uniformInt(-10, 11), Random::uniformInt(-10, 11));
            tsdf.InsertBlock(r);
        }

        ivec3 pos(Random::uniformInt(-10, 11), Random::uniformInt(-10, 11), Random::uniformInt(-10, 11));
        iRect<3> rect(pos);
        rect.Expand(Random::uniformInt(0, 20));


        //        (ivec3(-3, -2, -4), ivec3(2, 7, 100));

        int before_in_rect = tsdf.NumBlocksInRect(rect);
        EXPECT_LT(before_in_rect, tsdf.current_blocks);

        tsdf.CropToRect(rect);

        EXPECT_EQ(tsdf.current_blocks, tsdf.NumBlocksInRect(rect));
        EXPECT_EQ(before_in_rect, tsdf.NumBlocksInRect(rect));
    }
}


TEST(TSDF, VirtualVoxelIndex)
{
    {
        SparseTSDF tsdf(2, 1000, 1000);

        // equal
        EXPECT_EQ(tsdf.VirtualVoxelIndex({0, 0, 0}), ivec3(0, 0, 0));
        EXPECT_EQ(tsdf.VirtualVoxelIndex({4, 2, -2}), ivec3(2, 1, -1));

        // rounding
        EXPECT_EQ(tsdf.VirtualVoxelIndex({0.49, 0.99, 0.49}), ivec3(0, 0, 0));
        EXPECT_EQ(tsdf.VirtualVoxelIndex({-0.49, -0.49, -0.49}), ivec3(0, 0, 0));

        EXPECT_EQ(tsdf.VirtualVoxelIndex({0.49, 0.52, 1.51}), ivec3(0, 0, 1));


        EXPECT_EQ(tsdf.VirtualVoxelIndex({1.01, -1.01, 2.99}), ivec3(1, -1, 1));
        EXPECT_EQ(tsdf.VirtualVoxelIndex({1.01, -1.01, 3.01}), ivec3(1, -1, 2));
    }


    {
        SparseTSDF tsdf(2, 1000, 1000);
        EXPECT_EQ(tsdf.GetBlockIndex(ivec3{0, 0, 0}), ivec3(0, 0, 0));
        EXPECT_EQ(tsdf.GetBlockIndex(ivec3{8, 0, 0}), ivec3(1, 0, 0));
        EXPECT_EQ(tsdf.GetBlockIndex(ivec3{0, 0, -1}), ivec3(0, 0, -1));
        EXPECT_EQ(tsdf.GetBlockIndex(ivec3{-8, 0, -1}), ivec3(-1, 0, -1));
        EXPECT_EQ(tsdf.GetBlockIndex(ivec3{-8, -9, 8}), ivec3(-1, -2, 1));
    }
}


TEST(TSDF, GetVoxel)
{
    SparseTSDF tsdf(1, 1000, 1000);
    tsdf.InsertBlock({0, 0, 0});
    tsdf.InsertBlock({-1, 0, 0});
    tsdf.GetVoxel({-1, 0, 0});
    tsdf.GetVoxel({-5, 3, 7});
}

TEST(TSDF, TrilinearInterpolation)
{
    SparseTSDF tsdf(1, 1000, 1000);

    auto elements = tsdf.TrilinearAccess(vec3(-0.25, 0.5, -1.5));
    EXPECT_EQ(elements[0].first, ivec3(-1, 0, -2));
    EXPECT_EQ(elements[0].second, 0.25 * 0.5 * 0.5);

    EXPECT_EQ(elements[7].first, ivec3(0, 1, -1));
    EXPECT_EQ(elements[7].second, 0.75 * 0.5 * 0.5);

    auto b = tsdf.InsertBlock({0, 0, 0});

    tsdf.SetForAll(0, 1);

    b->data[0][0][0].distance = 0;
    b->data[0][0][1].distance = 0;
    b->data[0][1][0].distance = 0;
    b->data[0][1][1].distance = 0;

    b->data[1][0][0].distance = 1;
    b->data[1][0][1].distance = 1;
    b->data[1][1][0].distance = 1;
    b->data[1][1][1].distance = 1;
    b->data[3][3][3].weight   = 0;

    SparseTSDF::Voxel v;

    EXPECT_TRUE(tsdf.TrilinearAccess(vec3(0, 0, 0), v, 0));
    EXPECT_EQ(v.distance, 0);

    EXPECT_TRUE(tsdf.TrilinearAccess(vec3(0, 0, 1), v, 0));
    EXPECT_EQ(v.distance, 1);

    EXPECT_TRUE(tsdf.TrilinearAccess(vec3(0, 0, 0.25), v, 0));
    EXPECT_EQ(v.distance, 0.25);

    EXPECT_FALSE(tsdf.TrilinearAccess(vec3(-0.0001, 0, 0.25), v, 0));
    EXPECT_EQ(v.distance, 0);

    EXPECT_FALSE(tsdf.TrilinearAccess(vec3(3, 3, 3), v, 0));
    EXPECT_EQ(v.distance, 0);
}
TEST(TSDF, BasicFunctions)
{
    SparseTSDF tsdf(1, 1000, 1000);
    tsdf.InsertBlock({0, 0, 0});
    EXPECT_EQ(tsdf.current_blocks, 1);

    auto block = tsdf.GetBlock({0, 0, 0});
    EXPECT_TRUE(block);
}

TEST(TSDF, Trace)
{
    int w = 50;
    int h = 50;
    TemplatedImage<ucvec3> rgb_image(h, w);
    rgb_image.makeZero();
    auto result_view = rgb_image.getImageView();

    TemplatedImage<ucvec3> rgb_image2(h, w);
    rgb_image2.makeZero();

    //    SE3 view        = test->scene.images[0].V;
    SE3 model(Quat::Identity(), Vec3(0, 0, -2));
    IntrinsicsPinholed K(w, h, w / 2, h / 2, 0);
    Vec3 camera_pos = model.translation();

    auto tris = test->mesh.TriangleSoup();
    AccelerationStructure::ObjectMedianBVH bvh(tris);


    std::vector<double> errors;
    for (auto y : result_view.rowRange())
    {
        for (auto x : result_view.colRange())
        {
            Vec3 dir = K.unproject(Vec2(x, y), 1).normalized();
            dir      = model.so3() * dir;

            Ray ray(dir.cast<float>(), camera_pos.cast<float>());

            float t_max = 3;
            float t     = test->tsdf->RaySurfaceIntersection<2>(ray.origin, ray.direction, 0, t_max,
                                                            test->tsdf->voxel_size * 3, 0);


            if (t < t_max)
            {
                vec3 p_trace = ray.origin + ray.direction * t;

                float error = std::abs(test->sphere.sdf(p_trace));
                errors.push_back(error);

                vec3 n  = test->tsdf->TrilinearNormal(p_trace, 0);
                float c = std::max(n.dot(vec3(0, 0, -1)), 0.f) * 255;

                rgb_image(y, x) = ucvec3(c, c, c);
            }

            auto inter = bvh.getClosest(ray);

            if (inter.valid)
            {
                auto tri = tris[inter.triangleIndex];
                vec3 n   = tri.normal();
                float c  = std::max(n.dot(vec3(0, 0, -1)), 0.f) * 255;

                rgb_image2(y, x) = ucvec3(c, c, c);
            }

            EXPECT_EQ(t < t_max, inter.valid);
        }
    }

    Statistics stats(errors);
    EXPECT_LT(stats.max, 0.002);
    EXPECT_LT(stats.mean, 0.001);

    rgb_image.save("tsdf_trace.png");
    rgb_image2.save("tsdf_trace2.png");
}

}  // namespace Saiga

int main()
{
    Saiga::initSaigaSampleNoWindow();
    testing::InitGoogleTest();

    return RUN_ALL_TESTS();
}
