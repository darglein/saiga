/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/Core.h"
#include "saiga/core/time/all.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/vision/reconstruction/PoseOptimization_Ceres.h"
#include "saiga/vision/reconstruction/RobustPoseOptimization.h"
#include "saiga/vision/reconstruction/RobustSmoothPoseOptimization.h"
#include "saiga/vision/util/Random.h"

#include "gtest/gtest.h"

using namespace Saiga;


class RPOTest
{
   public:
    RPOTest()
    {
        scene.K    = StereoCamera4Base<T>(458.654, 457.296, 367.215, 248.375, 0, 50);
        scene.pose = Random::randomSE3();

        // Add some random observations
        double h = scene.K.cy * 2.0;
        double w = scene.K.cx * 2.0;

        for (int i = 0; i < 100; ++i)
        {
            Obs o;
            o.ip = Vec2(Random::sampleDouble(0, w), Random::sampleDouble(0, h));


            double depth = Random::sampleDouble(1, 5);
            Vec3 wp      = scene.K.unproject(o.ip, depth);
            wp           = scene.pose.inverse() * wp;


            if (Random::sampleDouble(0, 1) < 0.1)
            {
                o.depth = depth;
            }

            scene.obs.push_back(o);
            scene.wps.push_back(wp);
        }

        scene.outlier.resize(scene.obs.size(), false);
        scene.pose               = Random::JitterPose(scene.pose, 0.04, 0.01);
        scene.prediction         = Random::JitterPose(scene.pose, 0.04, 0.01);
        scene.pose               = scene.prediction;
        scene.weight_rotation    = 50;
        scene.weight_translation = 100;
        std::cout << "Init : " << scene.pose << std::endl;
        std::cout << "Pred : " << scene.prediction << std::endl;
    }



    double solveSaiga()
    {
        auto cpy = scene;
        //        RobustPoseOptimization<T, false> rpo(923475094325, 981450945);
        RobustSmoothPoseOptimization<T, false> rpo(923475094325, 981450945);
        rpo.optimizePoseRobust(cpy);
        std::cout << "Saiga: " << cpy.pose << std::endl;
        return cpy.chi2();
    }


    double solveCeres()
    {
        auto cpy = scene;
        OptimizePoseCeres(cpy, true);
        std::cout << "ceres: " << cpy.pose << std::endl;
        return cpy.chi2();
    }

    void TestBasic()
    {
        auto e1 = solveSaiga();
        auto e2 = solveCeres();

        std::cout << scene.chi2() << " -> (Saiga) " << e1 << " (Ceres) " << e2 << std::endl;
        EXPECT_NEAR(e1, e2, 1);
        //        EXPECT_NEAR(solveSaiga(), solveSaigaMP(), 1e-5);
    }


    using T       = double;
    using SE3Type = Sophus::SE3<T>;
    using Vec3    = Eigen::Matrix<T, 3, 1>;
    using Vec4    = Eigen::Matrix<T, 4, 1>;
    using Obs     = ObsBase<T>;

    PoseOptimizationScene<double> scene;
};



TEST(PoseEstimation, Sim3)
{
    for (int i = 0; i < 1; ++i)
    {
        RPOTest test;
        test.TestBasic();
    }
}
