/**
 * Copyright (c) 2021 Darius Rückert
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
#include "saiga/vision/reconstruction/Homography.h"
#include "saiga/vision/scene/Scene.h"
#include "saiga/vision/features/Features.h"

#include <numeric>
using namespace Saiga;

using FeatureDescriptor = DescriptorORB;

void testSimple()
{
    // let's get a random H and random points and try to predict H from the point matches
    Mat3 Href = Mat3::Random();
    Href      = (1.0 / Href(2, 2)) * Href;

    int N = 4;
    AlignedVector<Vec2> points1, points2;
    for (int i = 0; i < N; ++i)
    {
        Vec2 p = Vec2::Random();
        points1.push_back(p);
        points2.push_back((Href * p.homogeneous()).hnormalized());
    }


    Mat3 H = homography(points1, points2);
    H      = (1.0 / H(2, 2)) * H;

    std::cout << Href << std::endl;
    std::cout << H << std::endl;
    std::cout << "Error: " << (Href - H).norm() << std::endl;

    for (int i = 0; i < N; ++i)
    {
        std::cout << homographyResidual(points1[i], points2[i], H) << std::endl;
    }
}

void testRansac()
{
    // let's get a random H and random points and try to predict H from the point matches
    Mat3 Href = Mat3::Random();
    Href      = (1.0 / Href(2, 2)) * Href;

    int N_inlier  = 100;
    int N_outlier = 100;
    AlignedVector<Vec2> points1, points2;
    for (int i = 0; i < N_inlier; ++i)
    {
        Vec2 p = Vec2::Random();
        points1.push_back(p);
        points2.push_back((Href * p.homogeneous()).hnormalized());
    }

    for (int i = 0; i < N_outlier; ++i)
    {
        Vec2 p1 = Vec2::Random();
        Vec2 p2 = Vec2::Random();
        points1.push_back(p1);
        points2.push_back(p2);
    }


    RansacParameters rparams;
    rparams.maxIterations     = 200;
    rparams.residualThreshold = 0.1;
    rparams.threads           = 1;

    HomographyRansac hran(rparams);

    Mat3 H;
    int inliers;
    {
        SAIGA_BLOCK_TIMER();
        inliers = hran.solve(points1, points2, H);
    }
    std::cout << "ransac inliers: " << inliers << " / " << points1.size() << std::endl;
    std::cout << "H Error: " << (Href - H).norm() << std::endl;
}


int main(int, char**)
{
    initSaigaSampleNoWindow();

    Saiga::EigenHelper::checkEigenCompabitilty<15357>();
    Saiga::Random::setSeed(45786045);


    testSimple();
    testRansac();
    return 0;


    std::vector<KeyPoint<double>> keys1, keys2;
    std::vector<FeatureDescriptor> des1, des2;

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
    }
    BruteForceMatcher<DescriptorORB> matcher;
    matcher.matchKnn2(des1, des2);
    matcher.filterMatches(100, 0.8);


    std::vector<Vec2> points1, points2;

    for (auto m : matcher.matches)
    {
        points1.push_back(keys1[m.first].point);
        points2.push_back(keys2[m.second].point);
    }



    RansacParameters rparams;
    rparams.maxIterations     = 1000;
    rparams.residualThreshold = 5;

    HomographyRansac hran(rparams);

    Mat3 H;
    auto inliers = hran.solve(points1, points2, H);

    std::cout << "ransac inliers: " << inliers << " / " << matcher.matches.size() << std::endl;

    return 0;
}
