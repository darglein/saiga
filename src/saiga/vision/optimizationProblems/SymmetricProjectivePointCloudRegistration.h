/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/vision/VisionTypes.h"

namespace Saiga
{
/**
 * Optimizes a transformation between two point clouds, by minimizing the reprojection error.
 *
 * Camera 1 (C1) observes point cloud 1 (P1).
 * Camera 2 (C2) observes point cloud 2 (P2).
 *
 * During loop closure, we found matches from C1 to P2 and matches from C2 to P1. We know there must be a transformation
 * (T) from C1 -> C2. Using this T we can project the points from P2 to C1 and vise versa:
 *
 * ip1 = project1(T * P2);
 * ip2 = project2(T.inverse() * P1);
 *
 * The residual is the difference from the projected point to the observation.
 * The template TransformationType can be either SE3 or Sim3.
 *
 * @brief The SymmetricProjectivePointCloudRegistration class
 */
template <typename TransformationType>
struct SymmetricProjectivePointCloudRegistration
{
    IntrinsicsPinholed K;
    TransformationType T;
    std::vector<Vec3> points1, points2;
    double chi2Threshold;

    struct Observation
    {
        Vec2 imagePoint;
        double weight = 1;
        int wp        = -1;
    };
    std::vector<Observation> obs1, obs2;


    Vec2 residual1(const Observation& obs)
    {
        if (obs.wp == -1) return {0, 0};
        auto wp   = points2[obs.wp];
        Vec2 proj = K.project(T.inverse() * wp);
        return obs.weight * (obs.imagePoint - proj);
    }

    Vec2 residual2(const Observation& obs)
    {
        if (obs.wp == -1) return {0, 0};
        auto wp   = points1[obs.wp];
        Vec2 proj = K.project(T * wp);
        return obs.weight * (obs.imagePoint - proj);
    }


    double chi2()
    {
        double res = 0;
        for (auto&& o : obs1)
        {
            res += residual1(o).squaredNorm();
        }
        for (auto&& o : obs2)
        {
            res += residual2(o).squaredNorm();
        }
        return res * 0.5;
    }

    int removeOutliers(double reprojectionThreshold)
    {
        int removedPoints = 0;
        for (auto&& o : obs1)
        {
            auto e = residual1(o).squaredNorm();
            if (e > reprojectionThreshold)
            {
                o.wp = -1;
                removedPoints++;
            }
        }
        for (auto&& o : obs2)
        {
            auto e = residual2(o).squaredNorm();
            if (e > reprojectionThreshold)
            {
                o.wp = -1;
                removedPoints++;
            }
        }
        return removedPoints;
    }

    // Add some noise to the point cloud an T
    void addNoise(float sigma)
    {
        for (auto& p : points1) p += Vec3::Random() * sigma;
        for (auto& p : points2) p += Vec3::Random() * sigma;
        T.translation() += Vec3::Random() * sigma;
    }

    void setRandom(int N)
    {
        for (int i = 0; i < N; ++i)
        {
            points1.push_back(Vec3::Random());
            points2.push_back(Vec3::Random());

            Observation o;
            o.wp         = i;
            o.imagePoint = Vec2::Random();
            obs1.push_back(o);
            obs2.push_back(o);
        }
    }
};



}  // namespace Saiga
