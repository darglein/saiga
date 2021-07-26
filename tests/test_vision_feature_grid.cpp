/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */



#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/util/FeatureGrid.h"
#include "saiga/vision/util/FeatureGrid2.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"
namespace Saiga
{
struct FeatureGridTest
{
    FeatureGridTest()
    {
        // This are the camera parameters from the EuRoC dataset.
        intr = IntrinsicsPinholed(458.654, 457.296, 367.215, 248.375, 0);

        Vector<double, 8> c;
        //        c << -0.0284351, -2.47131, 1.7389, -0.145427, -2.26192, 1.63544, 0.00128128, -0.000454588;
        c << -0.283408, 0.0739591, 0, 0, 0, 0, 0.00019359, 1.76187e-05;
        dis = Distortion(c);
        w   = 752;
        h   = 480;
    }



    int w, h;
    IntrinsicsPinholed intr;
    Distortion dis;

    FeatureGridBounds<double, 24, 32> grid_bounds;
    FeatureGridBounds2<double, 20> grid_bounds2;

    FeatureGrid<24, 32> grid;
    FeatureGrid2 grid2;
};

inline FeatureGridTest test;

TEST(FeatureGridBounds, Create)
{
    std::cout << std::setprecision(20);
    test.grid_bounds.computeFromIntrinsicsDist(test.w, test.h, test.intr, test.dis, 5000);
    test.grid_bounds2.computeFromIntrinsicsDist(test.w, test.h, test.intr, test.dis, 5000);

    // Test Bounds
    Vec2 ref_bmin(-135.795, -92.8749);
    Vec2 ref_bmax(895.507, 565.497);
    ExpectCloseRelative(ref_bmin, test.grid_bounds.bmin, 0.5, false);
    ExpectCloseRelative(ref_bmax, test.grid_bounds.bmax, 0.5, false);
    ExpectCloseRelative(ref_bmin, test.grid_bounds2.bmin, 0.5, false);
    ExpectCloseRelative(ref_bmax, test.grid_bounds2.bmax, 0.5, false);

    // Test if cells cover complete area
    Vec2 ref_size   = ref_bmax.array() - ref_bmin.array();
    Vec2 grid_size  = test.grid_bounds.cellSize.array() * Vec2(test.grid_bounds.Cols, test.grid_bounds.Rows).array();
    Vec2 grid_size2 = test.grid_bounds2.cellSize.array() * Vec2(test.grid_bounds2.Cols, test.grid_bounds2.Rows).array();

    // the grid size must be large than ref but not more than one cell
    EXPECT_GE(grid_size(0), ref_size(0));
    EXPECT_GE(grid_size(1), ref_size(1));

    EXPECT_GE(grid_size2(0), ref_size(0));
    EXPECT_GE(grid_size2(1), ref_size(1));

    EXPECT_LT(grid_size(0), ref_size(0) + test.grid_bounds.cellSize(0));
    EXPECT_LT(grid_size(1), ref_size(1) + test.grid_bounds.cellSize(1));

    EXPECT_LT(grid_size2(0), ref_size(0) + test.grid_bounds2.cellSize(0));
    EXPECT_LT(grid_size2(1), ref_size(1) + test.grid_bounds2.cellSize(1));
}

TEST(FeatureGrid, Create)
{
    Saiga::Random::setSeed(953454);
    srand(3947435);

    // Add a few random points
    Vec2 ref_bmin(-135.795, -92.8749);
    Vec2 ref_bmax(895.507, 565.497);


    std::vector<KeyPoint<double>> keypoints;

    for (int i = 0; i < 500; ++i)
    {
        double x = Random::sampleDouble(ref_bmin(0) - 50, ref_bmax(0) + 50);
        double y = Random::sampleDouble(ref_bmin(1) - 50, ref_bmax(1) + 50);
        keypoints.emplace_back(x, y);
    }

    auto permutation  = test.grid.create(test.grid_bounds, keypoints);
    auto permutation2 = test.grid2.create(test.grid_bounds2, keypoints);


    std::vector<KeyPoint<double>> permuted_keypoints(keypoints.size());
    std::vector<KeyPoint<double>> permuted_keypoints2(keypoints.size());

    for (int i = 0; i < keypoints.size(); ++i)
    {
        permuted_keypoints[permutation[i]]   = keypoints[i];
        permuted_keypoints2[permutation2[i]] = keypoints[i];
    }


    // Go for each cell over all keypoints and check if they project to
    // this cell
    for (int i = 0; i < test.grid.Rows; ++i)
    {
        for (int j = 0; j < test.grid.Cols; ++j)
        {
            for (int c : test.grid.cellIt({j, i}))
            {
                std::pair<int, int> cid2;
                test.grid_bounds.cell(permuted_keypoints[c].point, cid2);
                SAIGA_ASSERT(cid2.first == j && cid2.second == i);
            }
        }
    }


    // Create more points and do a radius search
    for (int i = 0; i < 500; ++i)
    {
        double r = 40;
        double x = Random::sampleDouble(ref_bmin(0) - 50, ref_bmax(0) + 50);
        double y = Random::sampleDouble(ref_bmin(1) - 50, ref_bmax(1) + 50);
        //        double x = 334.35918311191995;
        //        double y = 544.07839070422017;
        Vec2 position(x, y);

        //        if (!test.grid_bounds.inImage(position)) return;

        // use brute force search;
        std::vector<int> ref, ref2;
        for (int i = 0; i < permuted_keypoints.size(); ++i)
        {
            auto kp = permuted_keypoints[i];
            if (!test.grid_bounds.inImage(kp.point)) continue;
            if ((kp.point - position).squaredNorm() < r * r)
            {
                ref.push_back(i);
            }

            auto kp2 = permuted_keypoints2[i];
            if (!test.grid_bounds2.inImage(kp2.point)) continue;
            if ((kp2.point - position).squaredNorm() < r * r)
            {
                ref2.push_back(i);
            }
        }
        std::sort(ref.begin(), ref.end());
        std::sort(ref2.begin(), ref2.end());


        {
            std::vector<int> grid_search;
            auto [cellMin, cellMax] = test.grid_bounds.minMaxCellWithRadius(position, r);
            for (auto cx : Range(cellMin.first, cellMax.first + 1))
            {
                for (auto cy : Range(cellMin.second, cellMax.second + 1))
                {
                    for (auto pid : test.grid.cellIt({cx, cy}))
                    {
                        auto& kp = permuted_keypoints[pid];
                        //                    std::cout << "test " << pid << " " << kp.point.transpose() << std::endl;

                        if ((kp.point - position).squaredNorm() < r * r)
                        {
                            grid_search.push_back(pid);
                            //                        std::cout << "found" << std::endl;
                        }
                    }
                }
            }
            std::sort(grid_search.begin(), grid_search.end());
            EXPECT_EQ(ref, grid_search) << x << "  " << y;
        }

        {
            std::vector<int> grid_search2;
            auto [cellMin, cellMax] = test.grid_bounds2.minMaxCellWithRadius(position, r);
            for (auto cx : Range(cellMin.first, cellMax.first + 1))
            {
                for (auto cy : Range(cellMin.second, cellMax.second + 1))
                {
                    for (auto pid : test.grid2.cellIt({cx, cy}))
                    {
                        auto& kp = permuted_keypoints2[pid];
                        //                    std::cout << "test " << pid << " " << kp.point.transpose() << std::endl;

                        if ((kp.point - position).squaredNorm() < r * r)
                        {
                            grid_search2.push_back(pid);
                            //                        std::cout << "found" << std::endl;
                        }
                    }
                }
            }
            std::sort(grid_search2.begin(), grid_search2.end());
            EXPECT_EQ(ref2, grid_search2) << x << "  " << y;
        }
    }
}

}  // namespace Saiga
