/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/core/time/all.h"
#include "saiga/vision/reconstruction/P3P.h"
#include "saiga/vision/util/Random.h"

using namespace Saiga;

class PnPTest
{
   public:
    PnPTest(int inliers, int outliers)
    {
        groundTruth = Random::randomSE3();

        SE3 gtInv = groundTruth.inverse();
        for (int i = 0; i < inliers; ++i)
        {
            Vec2 ip      = Vec2::Random();
            double depth = Random::sampleDouble(0.5, 10);

            Vec3 ipn = Vec3(ip(0), ip(1), 1) * depth;

            Vec3 wp = gtInv * ipn;

            Vec2 noise = Vec2::Random() * 0.001;

            wps.push_back(wp);
            ips.push_back(ip + noise);
            outlier.push_back(false);
        }

        std::cout << "Testing PnP solvers." << std::endl;
        std::cout << "Ground Truth SE3: " << std::endl << groundTruth << std::endl << std::endl;


        test();
        testRansac();
        benchmark();
    }

    void test()
    {
        {
            P3P p3pSolver2;
            auto bestSolution = p3pSolver2.solve4(ArrayView<Vec3>(wps).head(4), ArrayView<Vec2>(ips).head(4));
            auto res          = bestSolution.value();
            std::cout << "P3P colmap PNP " << std::endl;
            std::cout << res << std::endl;
            std::cout << "Error: " << rotationalError(res, groundTruth) << std::endl << std::endl;
        }
    }

    void benchmark()
    {
        int its      = 10;
        int innerIts = 1000;
        std::cout << "Running Benchmark with " << innerIts << " inner iterations." << std::endl;

        {
            P3P p3pSolver2;
            SE3 result;
            auto res = measureObject(its, [&]() {
                for (int i = 0; i < innerIts; ++i)
                {
                    auto bestSolution = p3pSolver2.solve4(ArrayView<Vec3>(wps).head(4), ArrayView<Vec2>(ips).head(4));
                    auto res          = bestSolution.value();
                    result            = res * result;
                }
            });
            std::cout << "Median time: " << res.median << " micro seconds" << std::endl << std::endl;
        }
    }

    void testRansac()
    {
        RansacParameters params;
        params.maxIterations     = 1000;
        params.residualThreshold = 0.001;


        P3PRansac pnp(params);
        SE3 result;
        std::vector<char> inlierMask;
        std::vector<int> inliers;
        auto num = pnp.solve(wps, ips, result, inliers, inlierMask);

        std::cout << "Ransac P3P" << std::endl;
        std::cout << "Inliers: " << num << std::endl;
        std::cout << result << std::endl;

        std::cout << "Error: T/R " << translationalError(groundTruth, result) << " "
                  << rotationalError(groundTruth, result) << std::endl
                  << std::endl;
    }

    AlignedVector<Vec3> wps;
    AlignedVector<Vec2> ips;
    std::vector<bool> outlier;
    SE3 groundTruth;
};


int main(int, char**)
{
    Saiga::Random::setSeed(45786045);


    PnPTest test(100, 100);
    return 0;
}
