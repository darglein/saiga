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
            scene.obs.push_back(o);

            double depth = Random::sampleDouble(1, 5);
            Vec3 wp      = scene.K.unproject(o.ip, depth);
            wp           = scene.pose.inverse() * wp;
            scene.wps.push_back(wp);
        }

        std::cout << "[RPOTest]" << std::endl;
        std::cout << "Target Pose: " << scene.pose << std::endl;

        scene.outlier.resize(scene.obs.size(), false);
        scene.pose              = Random::JitterPose(scene.pose, 0.01, 0.02);
        scene.prediction        = Random::JitterPose(scene.pose, 0.01, 0.02);
        scene.prediction_weight = 100000;
        //        for (auto& wp : scene.wps)
        //        {
        //            wp *= 1.2;
        //        }

        std::cout << "Prediction Pose: " << scene.prediction << std::endl;
        std::cout << "Initial RMS: " << scene.chi2() << std::endl;
    }



    void solveSaiga()
    {
        auto cpy = scene;
        RobustPoseOptimization<T, false> rpo(923475094325, 981450945);
        rpo.optimizePoseRobust(cpy);
        std::cout << cpy.pose << std::endl;
        std::cout << "solveSaiga rms: " << cpy.chi2() << std::endl;
        std::cout << "solveSaiga pred. error: " << cpy.predictionError() << std::endl;
    }


    void solveCeres()
    {
        auto cpy = scene;
        OptimizePoseCeres(cpy);
        std::cout << cpy.pose << std::endl;
        std::cout << "solveCeres rms: " << cpy.chi2() << std::endl;
        std::cout << "solveCeres pred. error: " << cpy.predictionError() << std::endl;
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
    RPOTest test;
    test.solveSaiga();
    test.solveCeres();
}
