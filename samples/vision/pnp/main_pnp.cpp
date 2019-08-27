/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/vision/util/Random.h"

#include "PNP.h"
#include "p3p.h"
using namespace Saiga;


// SE3 solveP3P(const Vec3* worldPoints, const Vec2* normalizedPoints, int N)
//{
//    p3p solver;



//    double rotation_matrix[3][3], translation_matrix[3];
//    auto res = solver.solve(rotation_matrix, translation_matrix, normalizedPoints[0](0), normalizedPoints[0](1),
//                            worldPoints[0](0), worldPoints[0](1), worldPoints[0](2), normalizedPoints[1](0),
//                            normalizedPoints[1](1), worldPoints[1](0), worldPoints[1](1), worldPoints[1](2),
//                            normalizedPoints[2](0), normalizedPoints[2](1), worldPoints[2](0), worldPoints[2](1),
//                            worldPoints[2](2), normalizedPoints[3](0), normalizedPoints[3](1), worldPoints[3](0),
//                            worldPoints[3](1), worldPoints[3](2));
//    SAIGA_ASSERT(res);
//    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R(&rotation_matrix[0][0]);
//    Vec3 t = Vec3(translation_matrix[0], translation_matrix[1], translation_matrix[2]);

//    Quat q(R);
//    return SE3(q, t);
//}

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

        test();
        testRansac();
    }

    void test()
    {
        std::cout << groundTruth << std::endl;

        PNP<double> pnp;
        auto res = pnp.extractSE3(pnp.dlt(wps.data(), ips.data(), 8));
        std::cout << res << std::endl;

        p3p p3pSolver;
        auto success = p3pSolver.solve(wps.data(), ips.data(), res);
        std::cout << res << std::endl;
    }

    void testRansac()
    {
        PNP<double> pnp;
        std::vector<int> inliers;
        SE3 result;
        auto num = pnp.solvePNPRansac(wps, ips, inliers, result);
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
