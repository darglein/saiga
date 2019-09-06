/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/vision/util/Random.h"

#include "PNP.h"
#include "p3p.h"
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

            Vec2 noise = Vec2::Random() * 0.01;

            wps.push_back(wp);
            ips.push_back(ip + noise);
            outlier.push_back(false);
        }

        std::cout << "Testing PnP solvers." << std::endl;
        std::cout << "Ground Truth SE3: " << std::endl << groundTruth << std::endl << std::endl;


        test();
        testRansac();
    }

    void test()
    {
        PNP<double> pnp;
        auto res = pnp.extractSE3(pnp.dlt(wps.data(), ips.data(), 8));

        std::cout << "DLT PNP (not working) " << std::endl;
        std::cout << res << std::endl << std::endl;

        p3p p3pSolver;
        auto success = p3pSolver.solve(wps.data(), ips.data(), res);
        std::cout << "P3P PNP " << std::endl;
        std::cout << res << std::endl << std::endl;
    }

    void testRansac()
    {
        PNP<double> pnp;
        std::vector<int> inliers;
        SE3 result;
        auto num = pnp.solvePNPRansac(wps, ips, inliers, result);

        std::cout << "Ransac P3P" << std::endl;
        std::cout << "Inliers: " << num << std::endl;
        std::cout << result << std::endl;

        std::cout << "Error: T/R " << translationalError(groundTruth, result) << " "
                  << rotationalError(groundTruth, result) << std::endl;
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
