/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/framework/framework.h"
#include "saiga/core/image/image.h"
#include "saiga/core/math/Eigen_Compile_Checker.h"
#include "saiga/core/math/random.h"
#include "saiga/core/time/all.h"
#include "saiga/core/util/BinaryFile.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/features/Features.h"
#include "saiga/vision/reconstruction/EightPoint.h"
#include "saiga/vision/reconstruction/FivePoint.h"
#include "saiga/vision/reconstruction/TwoViewReconstruction.h"
#include "saiga/vision/scene/Scene.h"
#include "saiga/vision/scene/SynteticScene.h"
#include "saiga/vision/util/Random.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"

namespace Saiga
{
using FeatureDescriptor = DescriptorORB;



class FiveEightPointTest
{
   public:
    FiveEightPointTest()
    {
        //        Saiga::Random::setSeed(345345);
        scene = SynteticScene::CircleSphere(500, 2, 500, true);

        //        scene.images[0].se3 = SE3(Quat::Identity(), Vec3(0, 0, 0));
        //        scene.images[1].se3 = SE3(Quat::Identity(), Vec3(1, 0, 0));


        scene.applyErrorToImagePoints();

        //        std::cout << scene << std::endl;
        //        exit(0);

        K1 = scene.intrinsics[0];
        K2 = scene.intrinsics[0];

        // The transformation we want to reconstruct.
        T = scene.images[1].se3 * scene.images[0].se3.inverse();


        // Compute reference E and F
        reference_E = EssentialMatrix(SE3(), T);



        reference_T               = T;
        reference_T.translation() = reference_T.translation().normalized();
        reference_E               = EssentialMatrix(SE3(), reference_T);
        //        std::cout << "Ref T: " << T << std::endl;
        //        std::cout << skew(T.translation()) << std::endl;
        //        std::cout << T.rotationMatrix() << std::endl;
        //        std::cout << skew(T.translation()) * T.rotationMatrix() << std::endl;
        //        exit(0);
        reference_F = FundamentalMatrix(reference_E, K1, K2);

        N = scene.images[0].stereoPoints.size();
        SAIGA_ASSERT(scene.images[0].stereoPoints.size() == scene.images[1].stereoPoints.size());

        for (int i = 0; i < N; ++i)
        {
            auto ip1 = scene.images[0].stereoPoints[i];
            auto ip2 = scene.images[1].stereoPoints[i];

            points1.push_back(ip1.point);
            normalized_points1.push_back(K1.unproject2(ip1.point));

            points2.push_back(ip2.point);
            normalized_points2.push_back(K2.unproject2(ip2.point));

            SAIGA_ASSERT(ip1.wp != -1);
            SAIGA_ASSERT(ip1.wp == ip2.wp);
        }
    }

    SE3 T;
    SE3 reference_T;
    Mat3 reference_E, reference_F;

    IntrinsicsPinholed K1;
    IntrinsicsPinholed K2;

    int N;
    std::vector<Vec2> points1, points2;
    std::vector<Vec2> normalized_points1, normalized_points2;

    Scene scene;
};

TEST(EpipolarGeometry, EpipolarDistance)
{
    FiveEightPointTest test;
    for (int i = 0; i < test.N; ++i)
    {
        auto error_F = EpipolarDistanceSquared(test.points1[i], test.points2[i], test.reference_F);
        EXPECT_NEAR(sqrt(error_F), 0, 1e-10);

        auto error_E =
            EpipolarDistanceSquared(test.normalized_points1[i], test.normalized_points2[i], test.reference_E);
        EXPECT_NEAR(sqrt(error_E), 0, 1e-10);
    }
}

TEST(EpipolarGeometry, EightPointAlgorithm)
{
    for (int i = 0; i < 0; ++i)
    {
        FiveEightPointTest test;
        int offset = Random::uniformInt(0, test.N - 8);
        //        auto F     = FundamentalMatrixEightPoint(test.normalized_points1.data() + offset,
        //                                             test.normalized_points2.data() + offset);
        //        F          = test.K2.inverse().matrix().transpose() * F * test.K1.inverse().matrix();

        //        auto F = FundamentalMatrixEightPoint(test.points1.data() + offset, test.points2.data() + offset);
        auto F = FundamentalMatrixEightPointNormalized(test.points1.data() + offset, test.points2.data() + offset);
        F      = NormalizeEpipolarMatrix(F);

        auto E = EssentialMatrix(F, test.K1, test.K2);

        auto result =
            getValidTransformationFromE(E, test.normalized_points1.data(), test.normalized_points2.data(), test.N);

        auto T = result.first;



        Vec3 t     = T.translation().normalized();
        Vec3 ref_t = test.reference_T.translation().normalized();
        if (t(2) < 0) t *= -1;
        if (ref_t(2) < 0) ref_t *= -1;

        ExpectCloseRelative(t, ref_t, 1e-5, false);

        Vec4 q     = T.unit_quaternion().coeffs();
        Vec4 ref_q = test.reference_T.unit_quaternion().coeffs();

        if (q(3) < 0) q *= -1;
        if (ref_q(3) < 0) ref_q *= -1;

        ExpectCloseRelative(q, ref_q, 1e-5, false);


        {
            TwoViewReconstructionEightPoint tvr;
            RansacParameters params;
            params.maxIterations     = 500;
            double epipolarTheshold  = 1.5;
            params.residualThreshold = epipolarTheshold * epipolarTheshold;
            params.reserveN          = 2000;
            params.threads           = 8;
            tvr.init(params, test.scene.intrinsics[0]);

            tvr.compute(test.points1, test.points2, test.normalized_points1, test.normalized_points2);

            auto T = tvr.pose2();

            Vec3 t     = T.translation().normalized();
            Vec3 ref_t = test.reference_T.translation().normalized();
            if (t(2) < 0) t *= -1;
            if (ref_t(2) < 0) ref_t *= -1;

            ExpectCloseRelative(t, ref_t, 1e-5, false);

            Vec4 q     = T.unit_quaternion().coeffs();
            Vec4 ref_q = test.reference_T.unit_quaternion().coeffs();

            if (q(3) < 0) q *= -1;
            if (ref_q(3) < 0) ref_q *= -1;

            ExpectCloseRelative(q, ref_q, 1e-5, false);
        }
    }
}

TEST(EpipolarGeometry, FivePointAlgorithm)
{
    int failed = 0;
    for (int i = 0; i < 100; ++i)
    {
        FiveEightPointTest test;
        int offset = Random::uniformInt(0, test.N - 6);
        auto ps1   = test.normalized_points1.data() + offset;
        auto ps2   = test.normalized_points2.data() + offset;
        std::vector<Mat3> es;
        fivePointNister(ps1, ps2, es);

        Mat3 E;
        SE3 T;

        for (auto& E : es)
        {
            E = NormalizeEpipolarMatrix(E);
        }

        if (bestEUsing6Points(es, ps1, ps2, E, T))
        {
            Vec3 t     = T.translation().normalized();
            Vec3 ref_t = test.reference_T.translation().normalized();
            if (t(2) < 0) t *= -1;
            if (ref_t(2) < 0) ref_t *= -1;

            ExpectCloseRelative(t, ref_t, 1e-2, false);

            Vec4 q     = T.unit_quaternion().coeffs();
            Vec4 ref_q = test.reference_T.unit_quaternion().coeffs();

            if (q(3) < 0) q *= -1;
            if (ref_q(3) < 0) ref_q *= -1;

            ExpectCloseRelative(q, ref_q, 1e-2, false);
        }
        else
        {
            failed++;
        }
    }
    std::cout << "failed " << failed << std::endl;
}

TEST(EpipolarGeometry, Benchmark)
{
    int its = 50;
    {
        Mat3 sum;
        FiveEightPointTest test;
        auto stat = measureObject(its, [&]() {
            int offset = Random::uniformInt(0, test.N - 8);
            sum += FundamentalMatrixEightPoint(test.points1.data() + offset, test.points2.data() + offset);
        });
        std::cout << "FundamentalMatrixEightPoint: " << stat.median << " ms." << std::endl;
    }
    {
        Mat3 sum;
        FiveEightPointTest test;
        auto stat = measureObject(its, [&]() {
            int offset = Random::uniformInt(0, test.N - 8);
            sum += FundamentalMatrixEightPointNormalized(test.points1.data() + offset, test.points2.data() + offset);
        });
        std::cout << "FundamentalMatrixEightPointNormalized: " << stat.median << " ms." << std::endl;
    }
    {
        std::vector<Mat3> es;
        FiveEightPointTest test;
        auto stat = measureObject(its, [&]() {
            int offset = Random::uniformInt(0, test.N - 8);
            es.clear();
            fivePointNister(test.points1.data() + offset, test.points2.data() + offset, es);
        });
        std::cout << "fivePointNister: " << stat.median << " ms." << std::endl;
    }
}

}  // namespace Saiga
