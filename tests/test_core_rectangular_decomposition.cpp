/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/config.h"
#include "saiga/core/geometry/RectangularDecomposition.h"
#include "saiga/core/math/random.h"
#include "saiga/core/time/all.h"

#include "gtest/gtest.h"

using namespace Saiga;
using namespace RectangularDecomposition;

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


#if 0
    rectangles.clear();
    rectangles.push_back(Rect(ivec3(0, 0, 0), ivec3(6, 7, 3)));
#endif

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

    for (auto& r1 : rectangles)
    {
        for (auto& r2 : rectangles)
        {
            bool intersec = r1.Intersect(r2);
            EXPECT_EQ(intersec, r2.Intersect(r1));

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
    }
}

TEST(RectangularDecomposition, Shrink)
{
    if (1)
    {
        Rect r1(ivec3(0, 0, 0), ivec3(4, 4, 1));
        Rect r2(ivec3(2, 1, 0), ivec3(8, 3, 1));

        EXPECT_TRUE(r1.Intersect(r2));


        Rect o_in, o_out_l, o_out_r;
        EXPECT_TRUE(r1.ShrinkOtherToThis(r2, o_in, o_out_l, o_out_r));

        //    std::cout << o_in << std::endl;
        //    std::cout << o_out << std::endl;

        //        EXPECT_EQ(o_out_l.begin, ivec3(4, 1, 0));
        //        EXPECT_EQ(o_out_l.end, r2.end);

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
        //        std::cout << o_in << std::endl;
        //        std::cout << o_out_l << std::endl;
        //        std::cout << o_out_r << std::endl;
    }
}


template <typename T>
void TestDecomp(ArrayView<const ivec3> points)
{
    std::cout << "[Decomp] " << typeid(T).name() << std::endl;
    T decomp_system;
    decomp_system.conv_cost_weights = {1, 2, 1, 1};
    Decomposition result;
    float time;
    {
        ScopedTimer tm(time);
        result = decomp_system.Compute(points);
    }
    EXPECT_TRUE(result.ContainsAll(points));
    //    EXPECT_TRUE(result.RemoveFullyContained().ContainsAll(points));
    //    EXPECT_TRUE(result.ShrinkIfPossible().ContainsAll(points));
    // EXPECT_EQ(triv_result.rectangles.size(), points.size());
    std::cout << result << " C = " << decomp_system.ConvolutionCost(result) << std::endl;
    //    std::cout << result.RemoveFullyContained() << std::endl;
    //    std::cout << result.ShrinkIfPossible() << std::endl;
    std::cout << "Evaluation Time: " << time << " ms." << std::endl;
    std::cout << std::endl;
}

TEST(RectangularDecomposition, Decomposition)
{
    Random::setSeed(10587235);
    srand(390476);
    //    auto points = RandomPointCloud(20, 3);
    auto points = RandomRectanglePointCloud(100, 10, 6);

    std::cout << "Points: " << points.size() << std::endl;

    ivec3 corner = points.front();
    for (auto& p : points)
    {
        corner = corner.array().min(p.array());
    }
    for (auto& p : points)
    {
        p -= corner;
    }

    TestDecomp<TrivialRectangularDecomposition>(points);
    TestDecomp<RowMergeDecomposition>(points);
    TestDecomp<OctTreeDecomposition>(points);
    TestDecomp<SaveMergeDecomposition>(points);
    //    exit(0);
    TestDecomp<GrowAndShrinkDecomposition>(points);


    //    RowMergeDecomposition rm_decomp;
    //    auto rm_result = rm_decomp.Compute(points);
    //    EXPECT_TRUE(rm_result.ContainsAll(points));
    //    EXPECT_TRUE(rm_result.RemoveFullyContained().ContainsAll(points));
    //    EXPECT_TRUE(rm_result.ShrinkIfPossible().ContainsAll(points));
    //    std::cout << rm_result << std::endl;
    //    std::cout << rm_result.RemoveFullyContained() << std::endl;
    //    std::cout << rm_result.ShrinkIfPossible() << std::endl;

    //    OctTreeDecomposition ot_decomp;
    //    auto ot_result = ot_decomp.Compute(points);
    //    EXPECT_TRUE(ot_result.ContainsAll(points));
    //    EXPECT_TRUE(ot_result.RemoveFullyContained().ContainsAll(points));
    //    EXPECT_TRUE(ot_result.ShrinkIfPossible().ContainsAll(points));
    //    std::cout << ot_result << std::endl;
    //    std::cout << ot_result.RemoveFullyContained() << std::endl;
    //    std::cout << ot_result.ShrinkIfPossible() << std::endl;



    //    GrowAndShrinkDecomposition gs_decomp;
    //    auto gs_result = gs_decomp.Compute(points);
    //    EXPECT_TRUE(gs_result.ContainsAll(points));
    //    EXPECT_TRUE(gs_result.RemoveFullyContained().ContainsAll(points));
    //    EXPECT_TRUE(gs_result.ShrinkIfPossible().ContainsAll(points));
    //    std::cout << gs_result << std::endl;
    //    std::cout << gs_result.RemoveFullyContained() << std::endl;
    //    std::cout << gs_result.ShrinkIfPossible() << std::endl;
}
