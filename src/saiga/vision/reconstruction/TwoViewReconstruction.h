/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/vision/VisionTypes.h"
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
    inline void clear();
    inline void compute(Vec2* points1, Vec2* points2, int N);
    inline double medianAngle();

    // scales the scene so that the median depth is d
    inline void setMedianDepth(double d);
    inline double getMedianDepth();

    // build a saiga-scene.
    // mainly for debugging purposes
    inline Scene makeScene(Vec2* points1, Vec2* points2);

    Mat3 E;
    SE3 T;
    std::vector<int> inliers;
    int inlierCount;
    AlignedVector<Vec3> worldPoints;
};

void TwoViewReconstruction::clear()
{
    inliers.clear();
    worldPoints.clear();
}

void TwoViewReconstruction::compute(Vec2* points1, Vec2* points2, int N)
{
    clear();

    inlierCount = computeERansac(points1, points2, N, E, T, inliers);

    // triangulate points
    worldPoints.reserve(inlierCount);

    Triangulation<double> triangulation;
    for (auto i : inliers)
    {
        Vec3 p = triangulation.triangulateHomogeneous(SE3(), T, points1[i], points2[i]);
        worldPoints.push_back(p);
    }
}

double TwoViewReconstruction::medianAngle()
{
    std::vector<double> angles;

    auto c1 = Vec3(0, 0, 0);
    auto c2 = T.inverse().translation();

    for (auto& wp : worldPoints)
    {
        Vec3 v1 = (c1 - wp).normalized();
        Vec3 v2 = (c2 - wp).normalized();

        double cosA = v1.dot(v2);
        double A    = acos(cosA);
        angles.push_back(A);
    }
    std::sort(angles.begin(), angles.end());
    return angles[angles.size() / 2];
}

double TwoViewReconstruction::getMedianDepth()
{
    std::vector<double> depth;

    for (auto& wp : worldPoints)
    {
        depth.push_back(wp.z());
    }
    std::sort(depth.begin(), depth.end());
    return depth[depth.size() / 2];
}

void TwoViewReconstruction::setMedianDepth(double d)
{
    auto md     = getMedianDepth();
    auto factor = d / md;

    T.translation() = factor * T.translation();

    for (auto& wp : worldPoints)
    {
        wp *= factor;
    }
}

Scene TwoViewReconstruction::makeScene(Vec2* points1, Vec2* points2)
{
    Intrinsics4 intr;

    Scene scene;
    scene.intrinsics.push_back(intr);
    scene.images.resize(2);

    scene.images[0].extr = 0;
    scene.images[0].intr = 0;
    scene.images[1].extr = 1;
    scene.images[1].intr = 0;

    scene.extrinsics.push_back(Extrinsics(SE3()));
    scene.extrinsics.push_back(Extrinsics(T));

    for (int i = 0; i < inlierCount; ++i)
    {
        int idx = inliers[i];

        StereoImagePoint ip1;
        ip1.wp    = i;
        ip1.point = points1[idx];
        scene.images[0].stereoPoints.push_back(ip1);

        StereoImagePoint ip2;
        ip2.wp    = i;
        ip2.point = points2[idx];
        scene.images[1].stereoPoints.push_back(ip2);

        WorldPoint wp;
        wp.p = worldPoints[i];
        scene.worldPoints.push_back(wp);
    }
    scene.fixWorldPointReferences();
    return scene;
}


}  // namespace Saiga
