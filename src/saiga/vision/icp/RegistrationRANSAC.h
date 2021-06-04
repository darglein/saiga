/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/icp/ICPAlign.h"
#include "saiga/vision/util/Ransac.h"

#include <array>
#include <chrono>
#include <random>
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
    IntrinsicsPinholed camera1, camera2;
    double threshold;


    void clear()
    {
        points1.clear();
        points2.clear();
        ips1.clear();
        ips2.clear();
    }
    //
    /**
     * Compute T which maps from 1 to 2
     * The ransac error is computed by reprojection.
     *
     * Set computeScale=1 for monocular tracking to compensate scale drift
     *
     * Returns [Transformation,Scale,NumInliers]
     *
     * @brief solve
     * @param maxIterations
     * @param stopInliers Early termination when that amount of inliers are found
     */
    std::tuple<SE3, double, int> solve(int maxIterations, bool computeScale)
    {
        constexpr int sampleSize = 3;

        SAIGA_ASSERT(N > 0);
        std::uniform_int_distribution<unsigned int> dis(0, N - 1);


        std::array<Vec3, sampleSize> A;
        std::array<Vec3, sampleSize> B;

        SE3 bestT;
        int bestInliers  = 0;
        double bestScale = 1;

        for (int i = 0; i < maxIterations; ++i)
        {
            // Get 3 matches and store them in A,B
            for (auto j : Range(0, sampleSize))
            {
                auto idx = dis(gen);
                A[j]     = points1[idx];
                B[j]     = points2[idx];
            }

            // fit relative transformation with icp
            AlignedVector<ICP::Correspondence> corrs;
            for (int i = 0; i < (int)A.size(); ++i)
            {
                ICP::Correspondence c;
                c.srcPoint = A[i];
                c.refPoint = B[i];
                corrs.push_back(c);
            }


            double scale     = 1;
            double* scalePtr = computeScale ? &scale : nullptr;
            SE3 rel          = ICP::pointToPointDirect(corrs, scalePtr);

            int currentInliers = 0;

            if (scalePtr)
            {
                DSim3 T(rel, scale);
                // if we have that much scale drift something is broken an
                if (scale > 0.2 && scale < 5) currentInliers = numInliers(T);
            }
            else
            {
                currentInliers = numInliers(rel);
            }

            //            std::cout << "ransac test " << currentInliers << std::endl;
            if (currentInliers > bestInliers)
            {
                bestInliers = currentInliers;
                bestT       = rel;
                bestScale   = scale;
            }
        }
        return {bestT, bestScale, bestInliers};
    }

    template <typename Transformation>
    int numInliers(const Transformation& T)
    {
        int count          = 0;
        Transformation T12 = T;
        Transformation T21 = T.inverse();
        for (auto i : Range(0, N))
        {
            Vec3 point1inImage2 = camera2.project3(T12 * points1[i]);
            Vec3 point2inImage1 = camera1.project3(T21 * points2[i]);

            // projected point is behind one of the cameras
            if (point1inImage2(2) < 0 || point2inImage1(2) < 0) continue;

            // check reprojection error
            auto e1 = (point1inImage2.segment<2>(0) - ips2[i]).squaredNorm();
            auto e2 = (point2inImage1.segment<2>(0) - ips1[i]).squaredNorm();
            if (e1 < threshold && e2 < threshold)
            {
                count++;
            }
        }
        return count;
    }



   private:
    //    std::mt19937 gen = std::mt19937(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    std::mt19937 gen = std::mt19937(92730469346UL);
};

}  // namespace Saiga
