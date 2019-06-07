/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"

#include <vector>


namespace Saiga
{
/**
 * Computes the best rigid-transformation between two point clouds using the RANSAC algorithm.
 * The error is messuared by projection.
 */
class RegistrationProjectRANSAC
{
   public:
    // Must be set by the user
    int N;
    SE3 pose1, pose2;
    std::vector<Vec3> points1, points2;
    std::vector<Vec2> ips1, ips2;
    Intrinsics4 camera1, camera2;
    double threshold;


    // Compute T which maps from 1 to 2
    void solve() {}

    int numInliers(const SE3& T)
    {
        int count = 0;
        SE3 T12   = pose2 * T;
        SE3 T21   = pose1 * T.inverse();

        for (auto i : Range(0, N))
        {
            Vec2 point1inImage2 = camera2.project(T12 * points1[i]);
            Vec2 point2inImage1 = camera1.project(T21 * points2[i]);

            auto e1 = (point1inImage2 - ips2[i]).squaredNorm();
            auto e2 = (point2inImage1 - ips1[i]).squaredNorm();

            if (e1 < threshold && e2 < threshold)
            {
                count++;
            }
        }
        return count;
    }
};

}  // namespace Saiga
