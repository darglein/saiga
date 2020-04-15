/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/time/all.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/ba/BAWrapper.h"
#include "saiga/vision/reconstruction/FivePoint.h"
#include "saiga/vision/scene/Scene.h"

namespace Saiga
{
/**
 * Complete two-view reconstruction based on the 5-point algorithm.
 *
 * Input:
 *      Set of feature matches (in normalized image space!!!)
 * Output:
 *      Relative Camera Transformation
 *      Set of geometric inliers
 *      3D world points of inliers
 * Optional Output (for further processing)
 *      Median triangulation angle
 */
class TwoViewReconstruction
{
   public:
    inline TwoViewReconstruction();
    // must be called once before running compute!
    void init(const RansacParameters& fivePointParams)
    {
        fpr.init(fivePointParams);
        tmpArray.reserve(fivePointParams.reserveN);
    }

    inline void compute(ArrayView<const Vec2> points1, ArrayView<const Vec2> points2, int threads);
    inline int NumPointWithAngleAboveThreshold(double angle);

    inline double medianAngle();
    inline double medianAngleByDepth();

    // scales the scene so that the median depth is d
    inline void setMedianDepth(double d);
    inline double getMedianDepth();

    // optimize with bundle adjustment
    inline int optimize(int its, float thresholdChi1);

    SE3& pose1() { return scene.extrinsics[0].se3; }
    SE3& pose2() { return scene.extrinsics[1].se3; }

    void clear()
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


    int N = 0;
    Mat3 E;
    std::vector<int> inliers;
    std::vector<char> inlierMask;
    int inlierCount = 0;
    //    AlignedVector<Vec3> worldPoints;
    Scene scene;

    std::vector<double> tmpArray;
    FivePointRansac fpr;
    Triangulation<double> triangulation;

    OptimizationOptions op_options;
    BAOptions ba_options;
    BAWrapper ba;
};


TwoViewReconstruction::TwoViewReconstruction()
{
    Intrinsics4 intr;
    scene.intrinsics.push_back(intr);
    scene.images.resize(2);

    scene.images[0].extr = 0;
    scene.images[0].intr = 0;
    scene.images[1].extr = 1;
    scene.images[1].intr = 0;

    scene.extrinsics.push_back(Extrinsics(SE3()));
    scene.extrinsics.push_back(Extrinsics(SE3()));

    scene.extrinsics[0].constant = true;

    int maxPoints = 2000;
    scene.worldPoints.reserve(maxPoints);
    scene.images[0].stereoPoints.reserve(maxPoints);
    scene.images[1].stereoPoints.reserve(maxPoints);

    op_options = defaultBAOptimizationOptions();
}



void TwoViewReconstruction::compute(ArrayView<const Vec2> points1, ArrayView<const Vec2> points2, int threads)
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


#pragma omp parallel num_threads(threads)
    {
        inlierCount = fpr.solve(points1, points2, E, pose2(), inliers, inlierMask);
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
            wp.p     = triangulation.triangulateHomogeneous(pose1(), pose2(), points1[i], points2[i]);
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
    ba_options.huberMono     = thresholdChi1;
    op_options.debugOutput   = false;
    op_options.minChi2Delta  = 1e-10;
    op_options.maxIterations = its;
    auto threshold2          = thresholdChi1 * thresholdChi1;

    ba.create(scene);
    ba.initAndSolve(op_options, ba_options);

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

void TwoViewReconstruction::setMedianDepth(double d)
{
    auto md     = getMedianDepth();
    auto factor = d / md;
    scene.rescale(factor);
}


}  // namespace Saiga
