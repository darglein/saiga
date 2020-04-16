/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/framework/framework.h"
#include "saiga/core/image/image.h"
#include "saiga/core/math/Eigen_Compile_Checker.h"
#include "saiga/core/math/random.h"
#include "saiga/core/time/all.h"
#include "saiga/core/util/BinaryFile.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/reconstruction/FivePoint.h"
#include "saiga/vision/reconstruction/TwoViewReconstruction.h"
#include "saiga/vision/scene/Scene.h"
#include "saiga/vision/util/Features.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"

namespace Saiga
{
using FeatureDescriptor = DescriptorORB;



class ReconstructionTest
{
   public:
    ReconstructionTest()
    {
        auto p1 = SearchPathes::data("vision/0.features");
        auto p2 = SearchPathes::data("vision/50.features");
        SAIGA_ASSERT(!p1.empty() && !p2.empty());

        {
            BinaryFile bf(p1, std::ios_base::in);
            bf >> keys1 >> des1;
        }
        {
            BinaryFile bf(p2, std::ios_base::in);
            bf >> keys2 >> des2;
        }

        five_point_ransac_params.maxIterations     = 2000;
        double epipolarTheshold                    = 1.5 / 535.4;
        five_point_ransac_params.residualThreshold = epipolarTheshold * epipolarTheshold;
        five_point_ransac_params.reserveN          = 2000;
        five_point_ransac_params.threads           = 8;
    }

    std::vector<KeyPoint<double>> keys1, keys2;
    std::vector<FeatureDescriptor> des1, des2;
    Intrinsics4 intr = Intrinsics4(535.4, 539.2, 320.1, 247.6);

    std::vector<Vec2> points1, points2;
    std::vector<Vec2> npoints1, npoints2;

    TwoViewReconstruction tvr;

    RansacParameters five_point_ransac_params;
};

std::unique_ptr<ReconstructionTest> test;

TEST(TwoViewReconstruction, Load)
{
    auto seed        = Random::generateTimeBasedSeed();
    ransacRandomSeed = seed;
    Random::setSeed(seed);

    test = std::make_unique<ReconstructionTest>();

    EXPECT_EQ(test->keys1.size(), 1007);
    EXPECT_EQ(test->keys2.size(), 1006);

    EXPECT_EQ(test->des1.size(), 1007);
    EXPECT_EQ(test->des2.size(), 1006);
}

TEST(TwoViewReconstruction, BruteForceMatcher)
{
    BruteForceMatcher<DescriptorORB> matcher;
    matcher.matchKnn2(test->des1, test->des2);
    matcher.filterMatches(100, 0.8);
    EXPECT_EQ(matcher.matches.size(), 343);

    for (auto m : matcher.matches)
    {
        test->points1.push_back(test->keys1[m.first].point);
        test->points2.push_back(test->keys2[m.second].point);
    }

    for (int i = 0; i < test->points1.size(); ++i)
    {
        test->npoints1.push_back(test->intr.unproject2(test->points1[i]));
        test->npoints2.push_back(test->intr.unproject2(test->points2[i]));
    }
}

TEST(TwoViewReconstruction, BruteForceMatcherOMP)
{
    BruteForceMatcher<DescriptorORB> matcher;
    matcher.matchKnn2_omp(test->des1, test->des2, 20);
    matcher.filterMatches(100, 0.8);
    EXPECT_EQ(matcher.matches.size(), test->points1.size());
}


TEST(TwoViewReconstruction, EssentialMatrix)
{
    test->tvr.init(test->five_point_ransac_params);

    Quat ref_q(0.999486, -0.00403537, -0.0285828, -0.0138906);
    ref_q.normalize();

    Vec3 ref_t(0.950824, -0.0201628, 0.308601);
    ref_t.normalize();


    double ref_angle_by_depth = degrees(0.0435928);
    double ref_angle          = degrees(0.0754463);

    for (int i = 0; i < 10; ++i)
    {
        test->tvr.compute(test->npoints1, test->npoints2, test->five_point_ransac_params.threads);
        EXPECT_GT(test->tvr.inlierCount, 270);
        EXPECT_LT(test->tvr.inlierCount, 290);

        auto be = test->tvr.inlierCount;
        test->tvr.optimize(5, 1.5 / 535.4);
        auto af = test->tvr.inlierCount;
        EXPECT_LE(af, be);

        auto q1 = test->tvr.pose1().unit_quaternion();
        auto t1 = test->tvr.pose1().translation();

        ExpectCloseRelative(q1.coeffs(), Quat::Identity().coeffs(), 1e-40);
        ExpectCloseRelative(t1, Vec3::Zero(), 1e-40);

        auto q2 = test->tvr.pose2().unit_quaternion();
        auto t2 = test->tvr.pose2().translation().normalized();

        ExpectCloseRelative(t2, ref_t, 0.1, false);
        ExpectCloseRelative(q2.coeffs(), ref_q.coeffs(), 0.01, false);

        ExpectCloseRelative(degrees(test->tvr.medianAngleByDepth()), ref_angle_by_depth, 0.15);
        ExpectCloseRelative(degrees(test->tvr.medianAngle()), ref_angle, 0.15);
    }
}

}  // namespace Saiga

int main()
{
    Saiga::initSaigaSampleNoWindow();
    testing::InitGoogleTest();

    return RUN_ALL_TESTS();
}
