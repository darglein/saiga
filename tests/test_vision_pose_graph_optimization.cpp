/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/config.h"
#include "saiga/core/Core.h"
#include "saiga/core/time/all.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/vision/ceres/CeresPGO.h"
#include "saiga/vision/recursive/PGORecursive.h"
#include "saiga/vision/recursive/PGOSim3Recursive.h"
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
        opoptions.debugOutput   = false;
        opoptions.debug         = false;
        opoptions.maxIterations = 20;
        opoptions.solverType    = OptimizationOptions::SolverType::Direct;
    }

    PoseGraph solveRec()
    {
        PoseGraph cpy = scene;

        if (cpy.fixScale)
        {
            PGORec ba;
            ba.optimizationOptions = opoptions;
            ba.create(cpy);
            ba.initAndSolve();
        }
        else
        {
            PGOSim3Rec ba;
            ba.optimizationOptions = opoptions;
            ba.create(cpy);
            ba.initAndSolve();
        }
        return cpy;
    }

    PoseGraph solveCeres()
    {
        PoseGraph cpy = scene;
        CeresPGO ba;
        ba.optimizationOptions = opoptions;
        ba.create(cpy);
        ba.initAndSolve();
        return cpy;
    }


    void test()
    {
        auto scene1 = solveRec();
        auto scene2 = solveCeres();

        std::cout << scene.chi2() << " -> (Saiga) " << scene1.chi2() << " (Ceres) " << scene2.chi2() << std::endl;

        ExpectClose(scene1.chi2(), scene2.chi2(), 1e-5);
    }

    void buildScene(bool with_scale_drift)
    {
        if (with_scale_drift)
        {
            scene = SyntheticPoseGraph::CircleWithDrift(5, 250, 6, 0.01, 0.005);
        }
        else
        {
            scene = SyntheticPoseGraph::CircleWithDrift(5, 250, 6, 0.01, 0);
        }
        scene.addNoise(0.01);
    }

    void buildSimpleScene(int num_cameras)
    {
        scene = SyntheticPoseGraph::Linear(num_cameras, 1);

        scene.addNoise(1.1);
    }

   private:
    PoseGraph scene;
    OptimizationOptions opoptions;
};

TEST(PoseGraph, LoadStore)
{
    PoseGraph pg1 = SyntheticPoseGraph::CircleWithDrift(5, 250, 6, 0.01, 0);

    pg1.sortEdges();
    pg1.fixScale = false;

    pg1.edges[0].weight = 5;
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
        EXPECT_EQ(e1.weight, e2.weight);

        ExpectCloseRelative(e1.T_i_j.params(), e2.T_i_j.params(), 1e-20);
    }
}

TEST(PoseGraphOptimization, SimpleLinear)
{
    PoseGraphOptimizationTest test;
    test.buildSimpleScene(4);
    test.test();

    //    exit(0);
}

TEST(PoseGraphOptimization, LoopClosingSE3)
{
    for (int i = 0; i < 5; ++i)
    {
        PoseGraphOptimizationTest test;
        test.buildScene(false);
        test.test();
    }
}
TEST(PoseGraphOptimization, LoopClosingSim3)
{
    for (int i = 0; i < 5; ++i)
    {
        PoseGraphOptimizationTest test;
        test.buildScene(true);
        test.test();
    }
}

}  // namespace Saiga
