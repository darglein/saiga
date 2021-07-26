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
#include "saiga/vision/util/Random.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"

namespace Saiga
{
class BundleAdjustmentTest
{
   public:
    BundleAdjustmentTest()
    {
        opoptions.debugOutput            = true;
        opoptions.debug                  = false;
        opoptions.maxIterations          = 4;
        opoptions.maxIterativeIterations = 40;
        opoptions.iterativeTolerance     = 1e-10;
        opoptions.numThreads             = 1;
        opoptions.buildExplizitSchur     = true;
        opoptions.solverType             = OptimizationOptions::SolverType::Iterative;

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

        {
            BARecRel ba_rel;
            ba_rel.optimizationOptions = opoptions;
            ba_rel.baOptions           = options;
            ba_rel.create(cpy);
            ba_rel.initAndSolve();
        }

        if (0)
        {
            BAPointOnly bapoint;
            bapoint.create(cpy);
            bapoint.initAndSolve();
        }
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

    BARecRel ba_rel;
    void test(const BAOptions& options)
    {
        auto scene1 = solveRec(options);

        auto scene2 = solveRecRel(options);
        auto scene3 = solveCeres(options);

        scene.chi2();
        std::cout << "saiga rel" << std::endl;
        scene2.chi2();
        std::cout << "ceres rel" << std::endl;
        scene3.chi2();
        //        auto scene1 = scene2;
        //        std::cout << scene.chi2(options.huberMono) << " -> (Saiga) " << scene1.chi2(options.huberMono)
        //                  << " -> (Saiga Rel) " << scene2.chi2(options.huberMono) << " (Ceres) "
        //                  << scene3.chi2(options.huberMono) << std::endl;

        //        ExpectClose(scene2.chi2(options.huberMono), scene3.chi2(options.huberMono), 1);
    }


    void buildScene()
    {
        //        int seed = Random::rand();
        int seed = 298783728;


        Random::setSeed(seed);

        std::cout << "seed " << seed << std::endl;
        int wps  = Random::uniformInt(200, 500);
        int cams = Random::uniformInt(5, 10);
        int obs  = Random::uniformInt(50, 100);
        scene    = SynteticScene::CircleSphere(wps, cams, obs);

        std::cout << "Creating Scene " << wps << "/" << cams << "/" << obs << std::endl;

        //        exit(0);
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
            rpc.rel_pose           = Random::JitterPose(rpc.rel_pose, 0.1, 0.00);
            rpc.img1               = i - 1;
            rpc.img2               = i;
            rpc.weight_rotation    = 500;
            rpc.weight_translation = 1000;
            scene.rel_pose_constraints.push_back(rpc);
        }



        std::cout << "before " << scene.chi2() << " " << scene.images.size() << std::endl;
        scene.compress();
        std::cout << "after " << scene.chi2() << " " << scene.images.size() << std::endl;

        scene.load("test_rel.scene");
        std::cout << scene << std::endl;
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
        //        BundleAdjustmentTest test;
        //        test.scene.rel_pose_constraints.clear();
        //        BAOptions options;
        //        test.test(options);
    }
}


TEST(BundleAdjustmentRelpose, Default)
{
    for (int i = 0; i < 1; ++i)
    {
        BundleAdjustmentTest test;
        BAOptions options;
        test.test(options);
    }
}

}  // namespace Saiga
