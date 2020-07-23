/**
 * Copyright (c) 2017 Darius Rückert
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
        opoptions.maxIterations          = 15;
        opoptions.maxIterativeIterations = 50;
        opoptions.iterativeTolerance     = 1e-10;
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
        auto scene2 = solveRecRel(options);
        auto scene3 = solveCeres(options);

        std::cout << scene.chi2(options.huberMono) << " -> (Saiga) " << scene1.chi2(options.huberMono)
                  << " -> (Saiga Rel) " << scene2.chi2(options.huberMono) << " (Ceres) "
                  << scene3.chi2(options.huberMono) << std::endl;

        ExpectClose(scene2.chi2(options.huberMono), scene3.chi2(options.huberMono), 1);
    }

    void BenchmarkRecursive(const OptimizationOptions& op_options, const BAOptions& options)
    {
        int its = 20;
        std::vector<double> timings;



        BARec ba;
        ba.optimizationOptions = op_options;
        ba.baOptions           = options;


        for (int i = 0; i < its; ++i)
        {
            Scene cpy = scene;

            float time;
            {
                ScopedTimer tim(time);
                ba.create(cpy);
                auto res = ba.initAndSolve();
                //                ba.initOMP();
                //                ba.solveOMP();
            }
            timings.push_back(time);
        }

        static bool first = true;
        Table tab({15, 15, 15, 15, 15, 15});
        if (first)
        {
            tab << "Type"
                << "Expl."
                << "Simple LM"
                << "Helper Threads"
                << "Solver Threads"
                << "Time(ms)";
            first = false;
        }
        tab << (op_options.solverType == OptimizationOptions::SolverType::Direct ? "Direct" : "Iterative")
            << op_options.buildExplizitSchur << op_options.simple_solver << options.helper_threads
            << options.solver_threads << Statistics(timings).median;
    }



    void buildScene()
    {
        scene = SynteticScene::CircleSphere(100, 3, 100);

        for (auto& img : scene.images)
        {
            for (auto& obs : img.stereoPoints)
            {
                if (Random::sampleDouble(0, 1) < 0.2)
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


        scene.images.front().constant = true;


        for (int i = 1; i < (int)scene.images.size(); ++i)
        {
            auto p1 = scene.images[i - 1].se3;
            auto p2 = scene.images[i].se3;

            RelPoseConstraint rpc;
            rpc.SetRelPose(p1, p2);
            rpc.img1               = i - 1;
            rpc.img2               = i;
            rpc.weight_rotation    = 50;
            rpc.weight_translation = 100;
            scene.rel_pose_constraints.push_back(rpc);
        }



        //        scene.



        //        scene.extrinsics[0].constant = true;
    }

    Scene scene;

    OptimizationOptions opoptions;

   private:
};

TEST(BundleAdjustmentRelpose, Empty)
{
    for (int i = 0; i < 1; ++i)
    {
        BundleAdjustmentTest test;
        test.buildScene();
        test.scene.rel_pose_constraints.clear();
        BAOptions options;
        test.test(options);
    }
}


TEST(BundleAdjustmentRelpose, Default)
{
    for (int i = 0; i < 1; ++i)
    {
        BundleAdjustmentTest test;
        test.buildScene();
        BAOptions options;
        test.test(options);
    }
}

}  // namespace Saiga
