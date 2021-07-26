/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/Core.h"
#include "saiga/core/time/all.h"
#include "saiga/core/util/table.h"
#include "saiga/vision/ceres/CeresBA.h"
#include "saiga/vision/recursive/BAPointOnly.h"
#include "saiga/vision/recursive/BARecursive.h"
#include "saiga/vision/recursive/BARecursiveRel.h"
#include "saiga/vision/scene/SynteticScene.h"
//#include "saiga/vision/scene/SynteticScene.h"


#include "gtest/gtest.h"

#include "compare_numbers.h"

namespace Saiga
{
class BundleAdjustmentTest
{
   public:
    BundleAdjustmentTest()
    {
        opoptions.debugOutput            = false;
        opoptions.debug                  = false;
        opoptions.maxIterations          = 50;
        opoptions.maxIterativeIterations = 100;
        opoptions.iterativeTolerance     = 1e-20;
        opoptions.minChi2Delta           = 1e-20;
        opoptions.numThreads             = 1;
        opoptions.buildExplizitSchur     = true;

        buildScene();
    }

    Scene solveRec(const BAOptions& options)
    {
        Scene cpy = scene;
        BARec ba;
        ba.optimizationOptions = opoptions;
        ba.baOptions           = options;
        ba.create(cpy);
        //        SAIGA_BLOCK_TIMER();
        ba.initAndSolve();
        return cpy;
    }

    Scene solveRecRel(const BAOptions& options)
    {
        Scene cpy = scene;
        BARecRel ba;
        ba.optimizationOptions = opoptions;
        ba.baOptions           = options;
        ba.create(cpy);
        //        SAIGA_BLOCK_TIMER();
        ba.initAndSolve();
        return cpy;
    }


    Scene solveRecOMP(const BAOptions& options)
    {
        Scene cpy = scene;
        BARec ba;
        ba.optimizationOptions      = opoptions;
        ba.baOptions                = options;
        ba.baOptions.helper_threads = 8;
        ba.create(cpy);
        //        SAIGA_BLOCK_TIMER();
        ba.initAndSolve();
        //        ba.initOMP();
        //        ba.solveOMP();
        return cpy;
    }

    Scene solveCeres(const BAOptions& options)
    {
        Scene cpy = scene;
        CeresBA ba;
        ba.optimizationOptions = opoptions;
        ba.baOptions           = options;
        ba.create(cpy);
        //        SAIGA_BLOCK_TIMER();
        ba.initAndSolve();
        return cpy;
    }


    void test(const BAOptions& options)
    {
        auto scene1 = solveRec(options);
        auto scene2 = solveCeres(options);
        auto scene3 = solveRecRel(options);
        //        auto scene3 = scen

        std::cout << scene.chi2(options.huberMono) << " -> " << scene1.chi2(options.huberMono) << " "
                  << scene2.chi2(options.huberMono) << std::endl;

        ExpectClose(scene1.chi2(options.huberMono), scene2.chi2(options.huberMono), 1e-1);
        ExpectClose(scene1.chi2(options.huberMono), scene3.chi2(options.huberMono), 1e-1);
    }

    void BenchmarkRecursive(const OptimizationOptions& op_options, const BAOptions& options, bool rel = false)
    {
        int its = 20;
        std::vector<double> timings;



        BARec ba;
        ba.optimizationOptions = op_options;
        ba.baOptions           = options;

        BARecRel barel;
        barel.optimizationOptions = op_options;
        barel.baOptions           = options;

        for (int i = 0; i < its; ++i)
        {
            float time;
            if (rel)
            {
                Scene cpy = scene;
                ScopedTimer tim(time);
                barel.create(cpy);
                auto res = barel.initAndSolve();
            }
            else
            {
                Scene cpy = scene;
                ScopedTimer tim(time);
                ba.create(cpy);
                auto res = ba.initAndSolve();
            }
            timings.push_back(time);
        }

        static bool first = true;
        Table tab({15, 15, 15, 15, 15, 15, 15});
        if (first)
        {
            tab << "Type"
                << "Expl."
                << "Simple LM"
                << "Helper Threads"
                << "Solver Threads"
                << "Rel Pose"
                << "Time(ms)";
            first = false;
        }
        tab << (op_options.solverType == OptimizationOptions::SolverType::Direct ? "Direct" : "Iterative")
            << op_options.buildExplizitSchur << op_options.simple_solver << options.helper_threads
            << options.solver_threads << rel << Statistics(timings).median;
    }



    void buildScene(bool with_depth = false, bool with_stereo = false)
    {
        scene = SynteticScene::CircleSphere(100, 10, 20);
        if (with_depth)
        {
            for (auto& img : scene.images)
            {
                for (auto& obs : img.stereoPoints)
                {
                    obs.depth = 1.0;
                }
            }
        }

        // 2 cm point noise
        scene.addWorldPointNoise(0.05);

        // Add 2 pixel image noise
        scene.addImagePointNoise(2.0);

        scene.addExtrinsicNoise(0.01);



        //        scene.extrinsics[0].constant = true;
    }

    Scene scene;

    OptimizationOptions opoptions;

   private:
};

TEST(Scene, LoadStore)
{
    Scene scene = SynteticScene::CircleSphere(2500, 65, 250);
    scene.save("test.scene");

    Scene scene2;
    scene2.load("test.scene");

    EXPECT_EQ(scene.intrinsics.size(), scene2.intrinsics.size());
    EXPECT_EQ(scene.images.size(), scene2.images.size());
    EXPECT_EQ(scene.worldPoints.size(), scene2.worldPoints.size());

    for (int i = 0; i < (int)std::min(scene.worldPoints.size(), scene2.worldPoints.size()); ++i)
    {
        EXPECT_EQ(scene.worldPoints[i].p, scene.worldPoints[i].p);
        EXPECT_EQ(scene.worldPoints[i].valid, scene.worldPoints[i].valid);
    }

    for (int i = 0; i < (int)std::min(scene.images.size(), scene2.images.size()); ++i)
    {
        EXPECT_EQ(scene.images[i].se3, scene.images[i].se3);
        EXPECT_EQ(scene.images[i].constant, scene.images[i].constant);
        EXPECT_EQ(scene.images[i].intr, scene.images[i].intr);
        EXPECT_EQ(scene.images[i].validPoints, scene.images[i].validPoints);
        EXPECT_EQ(scene.images[i].stereoPoints.size(), scene.images[i].stereoPoints.size());

        for (int j = 0; j < (int)std::min(scene.images[j].stereoPoints.size(), scene2.images[j].stereoPoints.size());
             ++j)
        {
            EXPECT_EQ(scene.images[i].stereoPoints[j].wp, scene.images[i].stereoPoints[j].wp);
            EXPECT_EQ(scene.images[i].stereoPoints[j].point, scene.images[i].stereoPoints[j].point);
        }
    }
}


TEST(BundleAdjustment, Empty)
{
    Scene scene;

    //    BARec ba;
    //    ba.create(scene);
    //    ba.initAndSolve();
}

TEST(BundleAdjustment, PointOnly)
{
    for (int i = 0; i < 10; ++i)
    {
        BundleAdjustmentTest test;
        test.buildScene(false);
        BAOptions options;

        for (auto& i : test.scene.images)
        {
            i.constant = true;
        }

        auto res_ceres = test.solveCeres(options);

        auto cpy = test.scene;
        BAPointOnly bapo;
        bapo.create(cpy);
        bapo.optimizationOptions = test.opoptions;
        bapo.baOptions           = options;
        bapo.initAndSolve();


        ExpectClose(res_ceres.chi2(), cpy.chi2(), 1e-1);

        std::cout << test.scene.chi2() << " -> " << res_ceres.chi2() << " " << cpy.chi2() << std::endl;
    }
}

TEST(BundleAdjustment, Default)
{
    for (int i = 0; i < 10; ++i)
    {
        BundleAdjustmentTest test;
        test.buildScene(false);
        BAOptions options;
        test.test(options);
    }
}


TEST(BundleAdjustment, DefaultParallel)
{
    BundleAdjustmentTest test;
    test.buildScene(false);
    test.opoptions.numThreads = 8;

    BAOptions options;
    auto ref1 = test.solveRec(options);
    auto ref2 = test.solveRecOMP(options);

    std::cout << ref1.chi2() << std::endl;
    std::cout << ref2.chi2() << std::endl;
    //    exit(0);
}


TEST(BundleAdjustment, DefaultDepth)
{
    for (int i = 0; i < 5; ++i)
    {
        BundleAdjustmentTest test;
        test.buildScene(true);
        BAOptions options;
        test.test(options);
    }
}


TEST(BundleAdjustment, PartialConstant)
{
    for (int i = 0; i < 10; ++i)
    {
        BundleAdjustmentTest test;
        test.scene.images[0].constant = true;
        BAOptions options;
        test.test(options);
    }

    for (int i = 0; i < 10; ++i)
    {
        BundleAdjustmentTest test;
        test.scene.images[0].constant = true;
        test.scene.images[1].constant = true;
        BAOptions options;
        test.test(options);
    }
}


TEST(BundleAdjustment, Huber)
{
    Random::setSeed(923652);
    for (int i = 0; i < 1; ++i)
    {
        BundleAdjustmentTest test;
        BAOptions options;
        options.huberMono   = 1;
        options.huberStereo = 0.1;
        //        test.test(options);
    }
}

TEST(BundleAdjustment, SLAM_LBA)
{
    //    Saiga::initSaigaSampleNoWindow();


#if 1
    Scene scene         = SynteticScene::CircleSphere(2500, 65, 250);
    int constant_images = 40;
#else
    Scene scene         = SynteticScene::CircleSphere(1800, 40, 250);
    int constant_images = 20;
#endif

    for (int i = 0; i < constant_images; ++i)
    {
        scene.images[i].constant = true;
    }
    //    std::cout << scene << std::endl;
    //    }
    //    Scene scene;
    //    scene.load("vision/local_ba.scene");
    scene.addWorldPointNoise(0.01);
    scene.addImagePointNoise(0.1);

    //    std::cout << scene << std::endl;

    OptimizationOptions local_op_options;
    local_op_options.debugOutput            = false;
    local_op_options.maxIterations          = 3;
    local_op_options.maxIterativeIterations = 30;
    local_op_options.iterativeTolerance     = 1e-20;
    local_op_options.solverType             = OptimizationOptions::SolverType::Direct;
    local_op_options.buildExplizitSchur     = true;
    local_op_options.simple_solver          = true;
    local_op_options.numThreads             = 1;

    // Huber makes (almost)  no difference
    BAOptions local_ba_options;
    local_ba_options.huberMono   = 2.4;
    local_ba_options.huberStereo = 2.8;



    BundleAdjustmentTest test;
    test.scene = scene;
    test.test(local_ba_options);


    //    test.BenchmarkRecursive("Default", local_op_options, local_ba_options);

    local_op_options.buildExplizitSchur = true;
    local_op_options.simple_solver      = true;
    local_op_options.solverType         = OptimizationOptions::SolverType::Iterative;
    test.BenchmarkRecursive(local_op_options, local_ba_options);
    test.BenchmarkRecursive(local_op_options, local_ba_options, true);

    local_op_options.buildExplizitSchur = true;
    local_op_options.simple_solver      = false;
    local_op_options.solverType         = OptimizationOptions::SolverType::Iterative;
    test.BenchmarkRecursive(local_op_options, local_ba_options);

    local_op_options.buildExplizitSchur = false;
    local_op_options.simple_solver      = true;
    local_op_options.solverType         = OptimizationOptions::SolverType::Iterative;
    test.BenchmarkRecursive(local_op_options, local_ba_options);

    local_op_options.buildExplizitSchur = false;
    local_op_options.simple_solver      = false;
    local_op_options.solverType         = OptimizationOptions::SolverType::Iterative;
    test.BenchmarkRecursive(local_op_options, local_ba_options);

    local_op_options.buildExplizitSchur = true;
    local_op_options.simple_solver      = false;
    local_op_options.solverType         = OptimizationOptions::SolverType::Direct;
    test.BenchmarkRecursive(local_op_options, local_ba_options);

    local_op_options.buildExplizitSchur = true;
    local_op_options.simple_solver      = true;
    local_op_options.solverType         = OptimizationOptions::SolverType::Direct;
    test.BenchmarkRecursive(local_op_options, local_ba_options);


    std::cout << std::endl;
    for (int i = 1; i < 8; ++i)
    {
        local_ba_options.helper_threads     = i;
        local_op_options.buildExplizitSchur = true;
        local_op_options.simple_solver      = true;
        local_op_options.solverType         = OptimizationOptions::SolverType::Iterative;
        test.BenchmarkRecursive(local_op_options, local_ba_options);
    }

    //    std::cout << std::endl;
    //    for (int i = 1; i < 8; ++i)
    //    {
    //        local_ba_options.helper_threads     = i;
    //        local_ba_options.solver_threads     = i;
    //        local_op_options.buildExplizitSchur = true;
    //        local_op_options.simple_solver      = true;
    //        local_op_options.solverType         = OptimizationOptions::SolverType::Iterative;
    //        test.BenchmarkRecursive(local_op_options, local_ba_options);
    //    }

    //    test.BenchmarkRecursive("Huber", local_op_options, local_ba_options);
}

}  // namespace Saiga
