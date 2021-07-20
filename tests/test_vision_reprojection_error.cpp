/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/Core.h"
#include "saiga/core/time/all.h"
#include "saiga/vision/ceres/CeresBA.h"
#include "saiga/vision/recursive/BAPointOnly.h"
#include "saiga/vision/recursive/BARecursive.h"
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


class BundleAdjustmentTest
{
   public:
    BundleAdjustmentTest()
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

        std::cout << scene.chi2(options.huberMono) << " " << scene1.chi2(options.huberMono) << " "
                  << scene2.chi2(options.huberMono) << std::endl;

        ExpectClose(scene1.chi2(options.huberMono), scene2.chi2(options.huberMono), 1e-5);
    }

    void buildScene()
    {
        SynteticScene sscene;
        sscene.numCameras     = 3;
        sscene.numImagePoints = 100;
        sscene.numWorldPoints = 100;
        scene                 = sscene.circleSphere();

        // 2 cm point noise
        scene.addWorldPointNoise(0.05);

        // Add 2 pixel image noise
        scene.addImagePointNoise(2.0);

        scene.addExtrinsicNoise(0.01);

        //        scene.extrinsics[0].constant = true;
    }

    Scene scene;

   private:
    OptimizationOptions opoptions;
};
#if 0
TEST(BundleAdjustment, Default)
{
    for (int i = 0; i < 10; ++i)
    {
        BundleAdjustmentTest test;
        BAOptions options;
        test.test(options);
    }
}

TEST(BundleAdjustment, PartialConstant)
{
    for (int i = 0; i < 10; ++i)
    {
        BundleAdjustmentTest test;
        test.scene.extrinsics[0].constant = true;
        BAOptions options;
        test.test(options);
    }

    for (int i = 0; i < 10; ++i)
    {
        BundleAdjustmentTest test;
        test.scene.extrinsics[0].constant = true;
        test.scene.extrinsics[1].constant = true;
        BAOptions options;
        test.test(options);
    }
}
#endif
TEST(BundleAdjustment, Huber)
{
    Random::setSeed(923652);
    for (int i = 0; i < 1; ++i)
    {
        BundleAdjustmentTest test;
        BAOptions options;
        options.huberMono   = 1;
        options.huberStereo = 0.1;
        test.test(options);
    }
}
