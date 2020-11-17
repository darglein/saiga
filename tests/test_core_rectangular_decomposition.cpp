/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/geometry/RectangularDecomposition.h"
#include "saiga/core/math/random.h"

#include "gtest/gtest.h"

using namespace Saiga;
using namespace RectangularDecomposition;


inline std::vector<ivec3> RandomPointCloud(int N, int radius)
{
    std::vector<ivec3> points;
    for (int i = 0; i < N; ++i)
    {
        ivec3 p = ivec3::Random().array();
        p.x() %= radius;
        p.y() %= radius;
        p.z() %= radius;
        points.push_back(p);
    }
    return points;  // RemoveDuplicates(points);
}

inline std::vector<ivec3> RandomRectanglePointCloud(int N, int radius, int r_size)
{
    std::vector<Rect> rectangles;
    for (int i = 0; i < N; ++i)
    {
        ivec3 p = ivec3::Random().array();
        p.x() %= radius;
        p.y() %= radius;
        p.z() %= radius;

        ivec3 s = ivec3::Random().array().abs();
        s.x() %= r_size;
        s.y() %= r_size;
        s.z() %= r_size;

        rectangles.push_back(Rect(p, p + s));
    }


    std::vector<ivec3> points;
    for (auto r : rectangles)
    {
        for (auto p : r.ToPoints())
        {
            points.push_back(p);
        }
    }
    return RemoveDuplicates(points);
}

TEST(RectangularDecomposition, Rect)
{
    Rect rect(ivec3(-4, 5, 3), ivec3(-1, 10, 7));

    EXPECT_TRUE(rect.Contains(rect.begin));
    EXPECT_FALSE(rect.Contains(rect.end));

    EXPECT_TRUE(rect.Contains(rect));

    auto points = rect.ToPoints();

    for (auto p : points)
    {
        EXPECT_TRUE(rect.Contains(p));
    }
    EXPECT_EQ(rect.Volume(), 60);
    EXPECT_EQ(rect.Expand(1).Volume(), 210);

    EXPECT_EQ(Rect(ivec3::Zero()).Volume(), 1);
    EXPECT_EQ(Rect(ivec3::Zero()).Expand(1).Volume(), 27);
}

TEST(RectangularDecomposition, Decomposition)
{
    Random::setSeed(10587235);
    srand(390476);
    //    auto points = RandomPointCloud(20, 2);
    auto points = RandomRectanglePointCloud(100, 10, 6);
    std::cout << "Points: " << points.size() << std::endl;

    TrivialRectangularDecomposition triv_decomp;
    auto triv_result = triv_decomp.Compute(points);
    EXPECT_TRUE(triv_result.ContainsAll(points));
    EXPECT_TRUE(triv_result.RemoveFullyContained().ContainsAll(points));
    EXPECT_TRUE(triv_result.ShrinkIfPossible().ContainsAll(points));
    EXPECT_EQ(triv_result.rectangles.size(), points.size());
    std::cout << triv_result << std::endl;
    std::cout << triv_result.RemoveFullyContained() << std::endl;
    std::cout << triv_result.ShrinkIfPossible() << std::endl;

    RowMergeDecomposition rm_decomp;
    auto rm_result = rm_decomp.Compute(points);
    EXPECT_TRUE(rm_result.ContainsAll(points));
    EXPECT_TRUE(rm_result.RemoveFullyContained().ContainsAll(points));
    EXPECT_TRUE(rm_result.ShrinkIfPossible().ContainsAll(points));
    std::cout << rm_result << std::endl;
    std::cout << rm_result.RemoveFullyContained() << std::endl;
    std::cout << rm_result.ShrinkIfPossible() << std::endl;



    GrowAndShrinkDecomposition gs_decomp;
    auto gs_result = gs_decomp.Compute(points);
    EXPECT_TRUE(gs_result.ContainsAll(points));
    EXPECT_TRUE(gs_result.RemoveFullyContained().ContainsAll(points));
    EXPECT_TRUE(gs_result.ShrinkIfPossible().ContainsAll(points));
    std::cout << gs_result << std::endl;
    std::cout << gs_result.RemoveFullyContained() << std::endl;
    std::cout << gs_result.ShrinkIfPossible() << std::endl;
}
