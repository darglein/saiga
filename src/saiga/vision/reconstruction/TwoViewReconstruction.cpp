/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "TwoViewReconstruction.h"

namespace Saiga
{
TwoViewReconstruction::TwoViewReconstruction()
{
    IntrinsicsPinholed intr;
    scene.intrinsics.push_back(intr);
    scene.images.resize(2);

    scene.images[0].intr = 0;
    scene.images[1].intr = 0;

    scene.images[0].constant = true;

    int maxPoints = 2000;
    scene.worldPoints.reserve(maxPoints);
    scene.images[0].stereoPoints.reserve(maxPoints);
    scene.images[1].stereoPoints.reserve(maxPoints);

    op_options = defaultBAOptimizationOptions();
}

void TwoViewReconstruction::init(const RansacParameters& fivePointParams)
{
    fpr.init(fivePointParams);
    tmpArray.reserve(fivePointParams.reserveN);
}

void TwoViewReconstruction::compute(ArrayView<const Vec2> points1, ArrayView<const Vec2> points2)
{
    N = points1.size();
    inliers.clear();
    inliers.reserve(N);
    inlierMask.resize(N);

    pose1() = SE3();

    scene.worldPoints.clear();
    scene.images[0].stereoPoints.resize(N);
    scene.images[1].stereoPoints.resize(N);
    scene.worldPoints.resize(N);

    int threads = fpr.Params().threads;

#pragma omp parallel num_threads(threads)
    {
        int num_inliers;

        {
            num_inliers = fpr.solve(points1, points2, E, pose2(), inliers, inlierMask);
        }
#pragma omp single
        {
            inlierCount = num_inliers;
        }

#pragma omp for
        for (int i = 0; i < N; ++i)
        {
            auto&& wp  = scene.worldPoints[i];
            auto&& ip1 = scene.images[0].stereoPoints[i];
            auto&& ip2 = scene.images[1].stereoPoints[i];
            if (!inlierMask[i])
            {
                // outlier
                wp.valid = false;
                ip1.wp   = -1;
                ip2.wp   = -1;
                continue;
            }

            // inlier
            wp.p     = TriangulateHomogeneous<double, true>(pose1(), pose2(), points1[i], points2[i]);
            wp.valid = true;

            ip1.wp    = i;
            ip1.point = points1[i];

            ip2.wp    = i;
            ip2.point = points2[i];
        }
    }
    scene.fixWorldPointReferences();


    //    SAIGA_ASSERT(scene);
}

double TwoViewReconstruction::medianAngle()
{
    tmpArray.clear();
    auto c1 = pose1().inverse().translation();
    auto c2 = pose2().inverse().translation();

    for (auto& wp2 : scene.worldPoints)
    {
        if (!wp2.valid) continue;
        auto A = TriangulationAngle(c1, c2, wp2.p);
        tmpArray.push_back(A);
    }
    if (tmpArray.empty())
    {
        return 0;
    }
    std::sort(tmpArray.begin(), tmpArray.end());
    return tmpArray[tmpArray.size() / 2];
}

double TwoViewReconstruction::medianAngleByDepth()
{
    double medianDepthKF2 = getMedianDepth();
    auto c1               = pose1().inverse().translation();
    auto c2               = pose2().inverse().translation();
    double baseline       = (c1 - c2).norm();
    return atan2(baseline / 2.0, medianDepthKF2);
}

int TwoViewReconstruction::NumPointWithAngleAboveThreshold(double angle)
{
    int count = 0;
    auto c1   = pose1().inverse().translation();
    auto c2   = pose2().inverse().translation();

    for (auto& wp2 : scene.worldPoints)
    {
        if (!wp2.valid) continue;
        auto A = TriangulationAngle(c1, c2, wp2.p);
        if (A > angle)
        {
            count++;
        }
    }
    return count;
}

double TwoViewReconstruction::getMedianDepth()
{
    tmpArray.clear();
    for (auto& wp2 : scene.worldPoints)
    {
        if (!wp2.valid) continue;
        auto wp = wp2.p;
        tmpArray.push_back(wp.z());
    }
    if (tmpArray.empty())
    {
        return 0;
    }
    std::sort(tmpArray.begin(), tmpArray.end());
    return tmpArray[tmpArray.size() / 2];
}

int TwoViewReconstruction::optimize(int its, float thresholdChi1)
{
    scene.rel_pose_constraints.clear();
    if (rel_pose_weight_rotation > 0 || rel_pose_weight_translation > 0)
    {
        RelPoseConstraint rpc;
        rpc.img1               = 0;
        rpc.img2               = 1;
        rpc.rel_pose           = rel_pose_prediction;
        rpc.weight_rotation    = rel_pose_weight_rotation;
        rpc.weight_translation = rel_pose_weight_translation;
        scene.rel_pose_constraints.push_back(rpc);
    }


    ba_options.huberMono     = thresholdChi1;
    op_options.debugOutput   = false;
    op_options.minChi2Delta  = 1e-10;
    op_options.maxIterations = its;
    auto threshold2          = thresholdChi1 * thresholdChi1;

    ba.optimizationOptions = op_options;
    ba.baOptions           = ba_options;
    ba.create(scene);
    ba.initAndSolve();

    // recompute inliers
    inlierCount        = 0;
    int removed_points = 0;
    for (int i = 0; i < N; ++i)
    {
        if (!inlierMask[i]) continue;
        auto e1 = scene.residual2(scene.images[0], scene.images[0].stereoPoints[i]).squaredNorm();
        auto e2 = scene.residual2(scene.images[1], scene.images[1].stereoPoints[i]).squaredNorm();

        if (std::max(e1, e2) < threshold2)
        {
            inlierCount++;
        }
        else
        {
            inlierMask[i] = false;
            removed_points++;
        }
    }
    return inlierCount;
}

void TwoViewReconstruction::clear()
{
    inliers.clear();
    inlierMask.clear();
    tmpArray.clear();
    scene.worldPoints.clear();
    scene.images[0].stereoPoints.clear();
    scene.images[1].stereoPoints.clear();
    N           = 0;
    inlierCount = 0;
}

void TwoViewReconstruction::setMedianDepth(double d)
{
    auto md     = getMedianDepth();
    auto factor = d / md;
    scene.rescale(factor);
}

void TwoViewReconstructionEightPoint::init(const RansacParameters& ransac_params, IntrinsicsPinholed K)
{
    if (solve_normalized)
    {
        auto cpy = ransac_params;
        cpy.residualThreshold /= (K.fx * K.fx);
        epr.init(cpy);
    }
    else
    {
        epr.init(ransac_params);
    }
    tmpArray.reserve(ransac_params.reserveN);
    scene.intrinsics.front() = K;
}

void TwoViewReconstructionEightPoint::compute(ArrayView<const Vec2> points1, ArrayView<const Vec2> points2,
                                              ArrayView<const Vec2> normalized_points1,
                                              ArrayView<const Vec2> normalized_points2)
{
    N = points1.size();
    inliers.clear();
    inliers.reserve(N);
    inlierMask.resize(N);

    pose1() = SE3();

    scene.worldPoints.clear();
    scene.images[0].stereoPoints.resize(N);
    scene.images[1].stereoPoints.resize(N);
    scene.worldPoints.resize(N);

    IntrinsicsPinholed& K = scene.intrinsics.front();

    int threads = epr.Params().threads;

#pragma omp parallel num_threads(threads)
    {
        if (solve_normalized)
        {
            inlierCount = epr.solve(normalized_points1, normalized_points2, F, inliers, inlierMask);
        }
        else
        {
            inlierCount = epr.solve(points1, points2, F, inliers, inlierMask);
        }
    }

    {
        if (solve_normalized)
        {
            // Transform back to unnormalized space
            F = K.inverse().matrix().transpose() * F * K.inverse().matrix();
        }

        E                    = EssentialMatrix(F, K, K);
        auto [rel, relcount] = getValidTransformationFromE(E, normalized_points1.data(), normalized_points2.data(),
                                                           inlierMask, normalized_points1.size(), threads);

        pose1() = SE3();
        pose2() = rel;
    }

#pragma omp parallel num_threads(threads)
    {
#pragma omp for
        for (int i = 0; i < N; ++i)
        {
            auto&& wp  = scene.worldPoints[i];
            auto&& ip1 = scene.images[0].stereoPoints[i];
            auto&& ip2 = scene.images[1].stereoPoints[i];
            if (!inlierMask[i])
            {
                // outlier
                wp.valid = false;
                ip1.wp   = -1;
                ip2.wp   = -1;
                continue;
            }

            // inlier
            wp.p = TriangulateHomogeneous<double, true>(pose1(), pose2(), normalized_points1[i], normalized_points2[i]);
            wp.valid = true;

            ip1.wp    = i;
            ip1.point = points1[i];

            ip2.wp    = i;
            ip2.point = points2[i];
        }
    }
    scene.fixWorldPointReferences();
}

}  // namespace Saiga
