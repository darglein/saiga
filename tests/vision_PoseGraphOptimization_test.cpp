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
#include "saiga/vision/scene/SynteticPoseGraph.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"

namespace Saiga
{
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

        //        ExpectClose(scene1.chi2(), scene2.chi2(), 1e-5);
    }

    void buildScene() { scene.load(Saiga::SearchPathes::data("user/kitty_old.posegraph")); }


   private:
    PoseGraph scene;
    OptimizationOptions opoptions;
};

TEST(PoseGraph, LoadStore)
{
    PoseGraph pg1 = SyntheticPoseGraph::CircleWithDrift(3, 4, 3, 0.01, 0.01);

    pg1.sortEdges();
    pg1.save("test.posegraph");

    PoseGraph pg2;
    pg2.load("test.posegraph");

    // compare vertices
    for (int i = 0; i < pg1.vertices.size(); ++i)
    {
        auto v1 = pg1.vertices[i];
        auto v2 = pg2.vertices[i];
        EXPECT_EQ(v1.constant, v2.constant);
        ExpectCloseRelative(v1.T_w_i.params(), v2.T_w_i.params(), 1e-20);
    }

    // compare edges
    for (int i = 0; i < pg1.edges.size(); ++i)
    {
        auto e1 = pg1.edges[i];
        auto e2 = pg2.edges[i];
        EXPECT_EQ(e1.from, e2.from);
        EXPECT_EQ(e1.to, e2.to);

        ExpectCloseRelative(e1.T_i_j.params(), e2.T_i_j.params(), 1e-20);
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

}  // namespace Saiga
