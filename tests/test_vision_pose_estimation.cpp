/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/Core.h"
#include "saiga/core/time/all.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/vision/reconstruction/PoseOptimization_Ceres.h"
#include "saiga/vision/reconstruction/RobustPoseOptimization.h"
#include "saiga/vision/util/Random.h"

#include "gtest/gtest.h"

using namespace Saiga;


class RPOTest
{
   public:
    RPOTest()
    {
        scene.K    = StereoCamera4Base<T>(458.654, 457.296, 367.215, 248.375, 50);
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
        scene.pose              = Random::JitterPose(scene.pose, 0.01, 0.02);
        scene.prediction        = Random::JitterPose(scene.pose, 0.01, 0.02);
        scene.prediction_weight = 0;
    }



    double solveSaiga()
    {
        auto cpy = scene;
        RobustPoseOptimization<T, false> rpo(923475094325, 981450945);
        rpo.optimizePoseRobust(cpy);
        return cpy.chi2();
    }


    double solveSaigaMP()
    {
        auto cpy = scene;
        RobustPoseOptimization<T, false> rpo(923475094325, 981450945);
        rpo.optimizePoseRobust(cpy.wps, cpy.obs, cpy.outlier, cpy.pose, cpy.K, 4);
        return cpy.chi2();
    }


    double solveCeres()
    {
        auto cpy = scene;
        OptimizePoseCeres(cpy, false);
        return cpy.chi2();
    }

    void TestBasic()
    {
        EXPECT_NEAR(solveSaiga(), solveCeres(), 1e-5);
        EXPECT_NEAR(solveSaiga(), solveSaigaMP(), 1e-5);
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
