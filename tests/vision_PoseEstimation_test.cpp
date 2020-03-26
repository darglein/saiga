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

        scene.outlier.resize(scene.obs.size(), false);
        std::cout << scene.rms() << std::endl;
        scene.pose = Random::JitterPose(scene.pose, 0.01, 0.02);
        std::cout << scene.rms() << std::endl;
    }

    void applyScale(double s)
    {
        for (auto& wp : scene.wps)
        {
            wp *= s;
        }
        std::cout << scene.rms() << std::endl;
    }

    void solveOld()
    {
        AlignedVector<int> outlier(scene.wps.size(), false);

        RobustPoseOptimization<T, false> rpo;
        int inliers = rpo.optimizePoseRobust(scene);
        std::cout << "rpo inliers: " << inliers << std::endl;

        std::cout << scene.rms() << std::endl;
    }


    void solveCeres()
    {
        AlignedVector<int> outlier(scene.wps.size(), false);

        RobustPoseOptimization<T, false> rpo;
        //        int inliers = rpo.optimizePoseRobust(scene);
        OptimizePoseCeres(scene);
        //        std::cout << "rpo inliers: " << inliers << std::endl;

        std::cout << scene.rms() << std::endl;
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
    //    test.solveOld();
    test.applyScale(20);
    test.solveCeres();
}
