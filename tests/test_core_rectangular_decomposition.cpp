/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/config.h"
#include "saiga/core/framework/framework.h"
#include "saiga/core/geometry/RectangularDecomposition.h"
#include "saiga/core/geometry/RectilinearOptimization.h"
#include "saiga/core/math/random.h"
#include "saiga/core/time/all.h"

#include "gtest/gtest.h"

using namespace Saiga;
using namespace RectangularDecomposition;



void CheckCover(const RectangleList& result, ArrayView<const ivec3> points, bool exact)
{
    auto cpy = result;
    DiscreteBVH bvh(cpy);

    for (auto& p : points)
    {
        std::vector<int> res;
        bvh.DistanceIntersect(Rect(p), -1, res);

        if (exact)
        {
            EXPECT_EQ(res.size(), 1);
        }
        else
        {
            EXPECT_GE(res.size(), 1);
        }
    }


    if (exact)
    {
        auto v = Volume(result);
        EXPECT_EQ(v, points.size());
    }
}
inline ivec3 RandomIvec3(int min_value, int max_value)
{
    ivec3 p;
    p.x() = Random::uniformInt(min_value, max_value);
    p.y() = Random::uniformInt(min_value, max_value);
    p.z() = Random::uniformInt(min_value, max_value);
    return p;
}


inline std::vector<ivec3> RandomPointCloud(int N, int radius)
{
    std::vector<ivec3> points;
    for (int i = 0; i < N; ++i)
    {
        ivec3 p = RandomIvec3(-radius, radius);
        points.push_back(p);
    }
    return RemoveDuplicates(points);
}

inline std::vector<ivec3> RandomRectanglePointCloud(int N, int radius, int r_size)
{
    std::vector<Rect> rectangles;
    for (int i = 0; i < N; ++i)
    {
        ivec3 p = RandomIvec3(-radius, radius);
        ivec3 s = RandomIvec3(1, r_size);
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

TEST(RectangularDecomposition, RectIntersect)
{
    std::vector<Rect> rectangles;
    for (int i = 0; i < 200; ++i)
    {
        ivec3 p = RandomIvec3(-4, 4);
        ivec3 s = RandomIvec3(1, 3);
        rectangles.push_back(Rect(p, p + s));
    }

    DiscreteBVH bvh(rectangles);

    for (auto& r1 : rectangles)
    {
        int intersecting_rectangles = 0;
        for (auto& r2 : rectangles)
        {
            bool intersec = r1.Intersect(r2);
            EXPECT_EQ(intersec, r2.Intersect(r1));
            EXPECT_EQ(intersec, r1.Distance(r2) < 0);

            intersecting_rectangles += intersec;

            auto p1s = r1.ToPoints();
            auto p2s = r2.ToPoints();

            bool intersecp1 = false;
            for (auto p1 : p1s)
            {
                if (r2.Contains(p1))
                {
                    intersecp1 = true;
                    break;
                }
            }
            EXPECT_EQ(intersec, intersecp1);


            bool intersecp2 = false;
            for (auto p2 : p2s)
            {
                if (r1.Contains(p2))
                {
                    intersecp2 = true;
                    break;
                }
            }
            EXPECT_EQ(intersec, intersecp2);
        }

        std::vector<int> result;
        bvh.DistanceIntersect(r1, -1, result);
        EXPECT_EQ(intersecting_rectangles, result.size());
    }
}

TEST(RectangularDecomposition, Shrink)
{
    {
        Rect r1(ivec3(0, 0, 0), ivec3(4, 4, 1));
        Rect r2(ivec3(2, 1, 0), ivec3(8, 3, 1));

        EXPECT_TRUE(r1.Intersect(r2));


        Rect o_in, o_out_l, o_out_r;
        EXPECT_TRUE(r1.ShrinkOtherToThis(r2, o_in, o_out_l, o_out_r));
        EXPECT_EQ(o_in.begin, r2.begin);
        EXPECT_EQ(o_in.end, ivec3(4, 3, 1));

        EXPECT_EQ(o_in.Volume() + o_out_l.Volume() + o_out_r.Volume(), r2.Volume());
    }

    {
        // Single shrink (1D) right
        Rect r1(ivec3(0, 0, 0), ivec3(4, 1, 1));
        Rect r2(ivec3(2, 0, 0), ivec3(9, 1, 1));
        EXPECT_TRUE(r1.Intersect(r2));
        Rect o_in, o_out_l, o_out_r;
        EXPECT_TRUE(r1.ShrinkOtherToThis(r2, o_in, o_out_l, o_out_r));
        EXPECT_EQ(o_in.Volume() + o_out_l.Volume() + o_out_r.Volume(), r2.Volume());

        EXPECT_EQ(o_out_l.Volume(), 0);
        EXPECT_EQ(o_out_r.Volume(), 5);
    }

    {
        // Single shrink (1D) left
        Rect r1(ivec3(0, 0, 0), ivec3(4, 1, 1));
        Rect r2(ivec3(-8, 0, 0), ivec3(1, 1, 1));
        EXPECT_TRUE(r1.Intersect(r2));
        Rect o_in, o_out_l, o_out_r;
        EXPECT_TRUE(r1.ShrinkOtherToThis(r2, o_in, o_out_l, o_out_r));
        EXPECT_EQ(o_in.Volume() + o_out_l.Volume() + o_out_r.Volume(), r2.Volume());

        EXPECT_EQ(o_out_l.Volume(), 8);
        EXPECT_EQ(o_out_r.Volume(), 0);
    }

    {
        // Double shrink (1D) (right)
        Rect r1(ivec3(0, 0, 0), ivec3(4, 1, 1));
        Rect r2(ivec3(-2, 0, 0), ivec3(9, 1, 1));
        EXPECT_TRUE(r1.Intersect(r2));
        Rect o_in, o_out_l, o_out_r;
        EXPECT_TRUE(r1.ShrinkOtherToThis(r2, o_in, o_out_l, o_out_r));
        EXPECT_EQ(o_in.Volume() + o_out_l.Volume() + o_out_r.Volume(), r2.Volume());

        EXPECT_EQ(o_out_l.Volume(), 2);
        EXPECT_EQ(o_out_r.Volume(), 5);
    }
}

TEST(RectangularDecomposition, ShrinkIfPossible)
{
    {
        RectangleList rectangles = {Rect(ivec3(0, 0, 0), ivec3(4, 4, 4)), Rect(ivec3(0, 0, 0), ivec3(4, 4, 4))};
        std::cout << to_string(rectangles) << std::endl;

        std::vector<ivec3> points;
        auto ps = rectangles.front().ToPoints();
        points.insert(points.end(), ps.begin(), ps.end());


        {
            auto cpy = rectangles;
            ShrinkIfPossible(cpy);
            EXPECT_EQ(Volume(cpy), 64);
            std::cout << to_string(cpy) << std::endl;
        }

        {
            auto cpy = rectangles;
            ShrinkIfPossible2(cpy, points);
            EXPECT_EQ(Volume(cpy), 64);
            std::cout << to_string(cpy) << std::endl;
        }
    }

    {
        RectangleList rectangles = {Rect(ivec3(0, 0, 0), ivec3(4, 4, 4)), Rect(ivec3(3, 0, 0), ivec3(7, 4, 4))};
        std::cout << to_string(rectangles) << std::endl;

        std::vector<ivec3> points;
        auto ps = Rect(ivec3(0, 0, 0), ivec3(4, 4, 4)).ToPoints();
        points.insert(points.end(), ps.begin(), ps.end());

        ps = Rect(ivec3(4, 0, 0), ivec3(7, 4, 4)).ToPoints();
        points.insert(points.end(), ps.begin(), ps.end());


        {
            auto cpy = rectangles;
            ShrinkIfPossible(cpy);
            //            EXPECT_EQ(Volume(cpy), 64);
            std::cout << to_string(cpy) << std::endl;
        }

        {
            auto cpy = rectangles;
            ShrinkIfPossible2(cpy, points);
            //            EXPECT_EQ(Volume(cpy), 64);
            std::cout << to_string(cpy) << std::endl;
        }
    }

    for (int i = 0; i < 10; ++i)
    {
        auto points     = RandomRectanglePointCloud(100, 10, 6);
        auto rectangles = DecomposeOctTree(points);
        // points.resize(points.size() / 2);
        auto old_points = points;
        points.clear();
        for (auto p : old_points)
        {
            if (Random::sampleBool(0.5)) points.push_back(p);
        }


        CheckCover(rectangles, points, false);

        auto cpy2 = rectangles;
        ShrinkIfPossible2(cpy2, points);
        RemoveEmpty(cpy2);
        CheckCover(cpy2, points, false);

        EXPECT_LE(cpy2.size(), rectangles.size());
    }
}


TEST(RectangularDecomposition, Decompose)
{
    Random::setSeed(10587235);
    srand(390476);

    for (int i = 0; i < 20; ++i)
    {
        auto points = RandomRectanglePointCloud(100, 10, 6);
        CheckCover(DecomposeTrivial(points), points, true);
        CheckCover(DecomposeRowMerge(points), points, true);
        CheckCover(DecomposeOctTree(points), points, true);
    }
}

TEST(RectangularDecomposition, MergeNeighborSave)
{
    for (int i = 0; i < 10; ++i)
    {
        auto points     = RandomRectanglePointCloud(100, 10, 6);
        auto rectangles = DecomposeTrivial(points);

        auto old_n = rectangles.size();
        MergeNeighborsSave(rectangles);
        CheckCover(rectangles, points, true);
        auto new_n = rectangles.size();
        std::cout << "merge save " << old_n << " -> " << new_n << std::endl;
    }
}

TEST(RectangularDecomposition, MergeNeighbor)
{
    VolumeCost cost({0, 8, 4, 1});
    for (int i = 0; i < 10; ++i)
    {
        auto points     = RandomRectanglePointCloud(100, 10, 6);
        auto rectangles = DecomposeTrivial(points);

        std::pair<int, float> old_n = {rectangles.size(), cost(rectangles)};
        MergeNeighbors(rectangles, cost, 100);
        CheckCover(rectangles, points, false);
        std::pair<int, float> new_n = {rectangles.size(), cost(rectangles)};
        std::cout << "merge " << old_n.first << "(" << old_n.second << ")"
                  << " -> " << new_n.first << "(" << new_n.second << ")" << std::endl;
    }
}

TEST(RectangularDecomposition, MergeShrink)
{
    VolumeCost cost({0, 8, 4, 1});
    for (int i = 0; i < 10; ++i)
    {
        auto points     = RandomRectanglePointCloud(100, 10, 6);
        auto rectangles = DecomposeTrivial(points);
        MergeNeighbors(rectangles, cost, 100);

        std::pair<int, float> old_n = {rectangles.size(), cost(rectangles)};
        MergeShrink(points, rectangles, 100, 50, cost);
        CheckCover(rectangles, points, false);
        std::pair<int, float> new_n = {rectangles.size(), cost(rectangles)};
        std::cout << "merge " << old_n.first << "(" << old_n.second << ")"
                  << " -> " << new_n.first << "(" << new_n.second << ")" << std::endl;
    }
}


TEST(RectangularDecomposition, Benchmark)
{
    Random::setSeed(3904763);
    auto points = RandomRectanglePointCloud(500, 15, 3);
    std::cout << "points " << points.size() << std::endl;

    //    TestDecomp<TrivialRectangularDecomposition>(points);
    //    TestDecomp<RowMergeDecomposition>(points);
    //    TestDecomp<OctTreeDecomposition>(points);
    //    TestDecomp<SaveMergeDecomposition>(points);
    //    TestDecomp<GrowAndShrinkDecomposition>(points);
}
