/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/icp/ICPAlign.h"

#include <chrono>
#include <random>
#include <vector>

//#define WORLD_SPACE_RANSAC

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
    /**
     * @brief solve
     * @param maxIterations
     * @param stopInliers Early termination when that amount of inliers are found
     */
    std::pair<SE3, int> solve(const SE3& guess, int maxIterations, int stopInliers)
    {
        //        std::cout << "Starting RANSAC... MaxIts=" << maxIterations << " Stopping at " << stopInliers << "
        //        inliers." << std::endl;
        SAIGA_ASSERT(N > 0);
        std::uniform_int_distribution<unsigned int> dis(0, N - 1);


        std::array<Vec3, 3> A;
        std::array<Vec3, 3> B;

        SE3 bestT;
        int bestInliers = 0;


        //        for (auto i : Range(0, maxIterations))
        for (int i = 0; i < maxIterations; ++i)
        {
            // Get 3 matches and store them in A,B
            for (auto j : Range(0, 3))
            {
                auto idx = dis(gen);
                A[j]     = points1[idx];
                B[j]     = points2[idx];
            }

            // fit trajectories with icp
            AlignedVector<ICP::Correspondence> corrs;
            for (int i = 0; i < (int)A.size(); ++i)
            {
                ICP::Correspondence c;
                c.srcPoint = A[i];
                c.refPoint = B[i];
                corrs.push_back(c);
            }
            SE3 rel = ICP::pointToPointDirect(corrs, guess, 4);

            int currentInliers = numInliers(rel);

            //            std::cout << "ransac test " << currentInliers << std::endl;
            if (currentInliers > bestInliers)
            {
                bestInliers = currentInliers;
                bestT       = rel;

                if (currentInliers > stopInliers)
                {
                    //                    break;
                }
            }
        }
        return {bestT, bestInliers};
    }

    int numInliers(const SE3& T)
    {
        int count = 0;
#ifdef WORLD_SPACE_RANSAC
        SE3 T12 = pose2 * T;
        SE3 T21 = pose1 * T.inverse();
#else
        SE3 T12 = T;
        SE3 T21 = T.inverse();
#endif



        for (auto i : Range(0, N))
        {
            Vec3 point1inImage2 = camera2.project3(T12 * points1[i]);
            Vec3 point2inImage1 = camera1.project3(T21 * points2[i]);

            // project behind one of the cameras
            if (point1inImage2(2) < 0 || point2inImage1(2) < 0) continue;

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
