/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/Core.h"
#include "saiga/core/time/all.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/vision/ceres/CeresPGO.h"
#include "saiga/vision/recursive/PGORecursive.h"
#include "saiga/vision/scene/SynteticScene.h"

#include "gtest/gtest.h"

using namespace Saiga;


bool ExpectClose(double x, double y, double max_abs_relative_difference)
{
    double absolute_difference = fabs(x - y);
    double relative_difference = absolute_difference / std::max(fabs(x), fabs(y));
    if (x == 0 || y == 0)
    {
        // If x or y is exactly zero, then relative difference doesn't have any
        // meaning. Take the absolute difference instead.
        relative_difference = absolute_difference;
    }

    EXPECT_NEAR(relative_difference, 0.0, max_abs_relative_difference);
    return relative_difference <= max_abs_relative_difference;
}


class PoseGraphOptimizationTest
{
   public:
    PoseGraphOptimizationTest()
    {
        opoptions.debugOutput            = true;
        opoptions.debug                  = false;
        opoptions.maxIterations          = 500;
        opoptions.maxIterativeIterations = 50;
        opoptions.iterativeTolerance     = 1e-10;
        opoptions.numThreads             = 1;
        opoptions.buildExplizitSchur     = true;

        buildScene();
    }

    PoseGraph solveRec()
    {
        PoseGraph cpy = scene;
        PGORec ba;
        ba.optimizationOptions = opoptions;
        ba.create(cpy);
        ba.initAndSolve();
        return cpy;
    }

    PoseGraph solveCeres()
    {
        PoseGraph cpy = scene;
        CeresPGO ba;
        ba.optimizationOptions = opoptions;
        ba.create(cpy);
        //        SAIGA_BLOCK_TIMER();
        ba.initAndSolve();
        return cpy;
    }


    void test()
    {
        auto scene1 = solveRec();
        auto scene2 = solveCeres();

        std::cout << scene.chi2() << " " << scene1.chi2() << " " << scene2.chi2() << std::endl;

        ExpectClose(scene1.chi2(), scene2.chi2(), 1e-5);
    }

    void buildScene() { scene.load(Saiga::SearchPathes::data("user/kitty_old.posegraph")); }


   private:
    PoseGraph scene;
    OptimizationOptions opoptions;
};

TEST(PoseGraph, LoadStore)
{
    SynteticScene sscene;
    sscene.numCameras     = 20;
    sscene.numImagePoints = 100;
    sscene.numWorldPoints = 100;

    PoseGraph pg1(sscene.circleSphere());
    pg1.sortEdges();
    pg1.save("test.posegraph");

    PoseGraph pg2;
    pg2.load("test.posegraph");

    // compare vertices
    for (int i = 0; i < pg1.poses.size(); ++i)
    {
        auto v1 = pg1.poses[i];
        auto v2 = pg2.poses[i];
        EXPECT_EQ(v1.constant, v2.constant);
        EXPECT_EQ(v1.se3.matrix(), v2.se3.matrix());
    }

    // compare edges
    for (int i = 0; i < pg1.edges.size(); ++i)
    {
        auto e1 = pg1.edges[i];
        auto e2 = pg2.edges[i];
        EXPECT_EQ(e1.from, e2.from);
        EXPECT_EQ(e1.to, e2.to);

        for (int j = 0; j < e1.from_pose.params().rows(); ++j)
        {
            EXPECT_EQ(e1.from_pose.params()[j], e2.from_pose.params()[j]);
            EXPECT_EQ(e1.to_pose.params()[j], e2.to_pose.params()[j]);
        }
    }
}
#if 0
TEST(PoseGraphOptimization, Default)
{
    Saiga::initSaigaSample();
    for (int i = 0; i < 1; ++i)
    {
        PoseGraphOptimizationTest test;
        test.test();
    }
}
#endif
