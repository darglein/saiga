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
#include "saiga/vision/features/Features.h"
#include "saiga/vision/reconstruction/FivePoint.h"
#include "saiga/vision/reconstruction/TwoViewReconstruction.h"
#include "saiga/vision/scene/Scene.h"

#include <numeric>
using namespace Saiga;

using FeatureDescriptor = DescriptorORB;



int main(int, char**)
{
    initSaigaSampleNoWindow();

    Saiga::EigenHelper::checkEigenCompabitilty<15357>();
    Saiga::Random::setSeed(45786045);



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
    std::vector<Vec2> npoints1, npoints2;
    IntrinsicsPinholed intr(535.4, 539.2, 320.1, 247.6, 0);

    for (auto m : matcher.matches)
    {
        points1.push_back(keys1[m.first].point);
        points2.push_back(keys2[m.second].point);
    }

    for (int i = 0; i < points1.size(); ++i)
    {
        npoints1.push_back(intr.unproject2(points1[i]));
        npoints2.push_back(intr.unproject2(points2[i]));
    }


    RansacParameters rparams;
    rparams.maxIterations     = 200;
    double epipolarTheshold   = 1.5 / 535.4;
    rparams.residualThreshold = epipolarTheshold * epipolarTheshold;
    rparams.reserveN          = 2000;
    rparams.threads           = 8;


    Mat3 E;
    std::vector<int> inliers;
    std::vector<char> inlierMask;
    SE3 rel;
    int num;

    {
        //        SAIGA_BLOCK_TIMER();
        num = computeERansac(npoints1, npoints2, rparams, E, rel, inliers, inlierMask);
        SAIGA_ASSERT(num == inliers.size());
    }

    std::cout << "5 Point Ransac Inliers: " << num << " " << npoints1.size() << std::endl;


    {
        for (int i = 0; i < 1; ++i)
        {
            //            SAIGA_BLOCK_TIMER();
            FivePointRansac fran(rparams);
#pragma omp parallel num_threads(rparams.threads)
            {
                num = fran.solve(npoints1, npoints2, E, rel, inliers, inlierMask);
            }
            std::cout << num << std::endl;
        }
        SAIGA_ASSERT(num == inliers.size());



        TwoViewReconstruction tvr;
        tvr.init(rparams);
        for (int i = 0; i < 50; ++i)
        {
            //            SAIGA_BLOCK_TIMER();
            //            tvr.compute(npoints1, npoints2, rparams.threads);
            //            std::cout << tvr.inlierCount << std::endl;
        }
        std::cout << tvr.scene << std::endl;
    }

    return 0;

    {
    }


    std::cout << "5 Point Ransac Inliers: " << num << " " << npoints1.size() << std::endl;

    Scene scene;
    scene.intrinsics.push_back(intr);
    scene.images.resize(2);

    scene.images[0].intr = 0;
    scene.images[1].intr = 0;

    scene.images[0].se3 = SE3();
    scene.images[0].se3 = rel;



    for (int i = 0; i < num; ++i)
    {
        int idx = inliers[i];

        StereoImagePoint ip1;
        ip1.wp    = i;
        ip1.point = points1[idx];
        scene.images[0].stereoPoints.push_back(ip1);

        StereoImagePoint ip2;
        ip2.wp    = i;
        ip2.point = points2[idx];
        scene.images[1].stereoPoints.push_back(ip2);

        WorldPoint wp;
        wp.p = TriangulateHomogeneous<double>(SE3(), rel, npoints1[idx], npoints2[idx]);
        scene.worldPoints.push_back(wp);
    }
    scene.fixWorldPointReferences();
    SAIGA_ASSERT(scene);


    std::cout << scene << std::endl;

    scene.save("fivePoint.scene");

    return 0;
}
