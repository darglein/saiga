/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/HomogeneousLSE.h"
#include "saiga/vision/VisionTypes.h"

#include "Epipolar.h"

#include <random>

namespace Saiga
{
template <typename T>
class EightPoint
{
   public:
    using Vec9 = Eigen::Matrix<T, 9, 1>;
    using Vec4 = Eigen::Matrix<T, 4, 1>;
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using Vec2 = Eigen::Matrix<T, 2, 1>;
    using Mat3 = Eigen::Matrix<T, 3, 3>;



    template <typename _InputIterator>
    Mat3 computeF(_InputIterator points1, _InputIterator points2)
    {
        Eigen::Matrix<T, 8, 9, Eigen::RowMajor> A(8, 9);

        for (int i = 0; i < 8; ++i)
        {
            auto& p = *points1;
            auto& q = *points2;

            T px = p(0);
            T py = p(1);
            T qx = q(0);
            T qy = q(1);

            std::array<double, 9> ax = {px * qx, px * qy, px, py * qx, py * qy, py, qx, qy, 1};
            for (int j = 0; j < 9; ++j)
            {
                A(i, j) = ax[j];
            }

            ++points1;
            ++points2;
        }

        Vec9 f;
        solveHomogeneousJacobiSVD(A, f);

        Mat3 F;
        F << f(0), f(3), f(6), f(1), f(4), f(7), f(2), f(5), f(8);
        F = enforceRank2(F);
        F = F * (1.0 / F(2, 2));
        return F;
    }


    int computeFRansac(Vec2* points1, Vec2* points2, int N, Mat3& bestF, std::vector<int>& bestInlierMatches)
    {
        int maxIterations        = 1000;
        constexpr int sampleSize = 8;
        double epipolarTheshold  = 1;
        double thresholdSquared  = epipolarTheshold * epipolarTheshold;

        std::uniform_int_distribution<unsigned int> dis(0, N - 1);


        std::array<Vec2, sampleSize> A;
        std::array<Vec2, sampleSize> B;

        int bestInliers = 0;


        for (int i = 0; i < maxIterations; ++i)
        {
            for (auto j : Range(0, sampleSize))
            {
                auto idx = dis(gen);
                A[j]     = points1[idx];
                B[j]     = points2[idx];
            }

            Mat3 F = computeF(A.begin(), B.begin());

            std::vector<int> inlierMatches;
            int numInliers = 0;
            for (int i = 0; i < N; ++i)
            {
                auto dSquared = EpipolarDistanceSquared(points1[i], points2[i], F);

                if (dSquared < thresholdSquared)
                {
                    inlierMatches.push_back(i);
                    numInliers++;
                }
            }

            if (numInliers > bestInliers)
            {
                bestInliers       = numInliers;
                bestF             = F;
                bestInlierMatches = inlierMatches;
            }
        }


        std::cout << "EightPoint Ransac finished: " << bestInliers << " Inliers" << std::endl;


        return bestInliers;
    }

   private:
    std::mt19937 gen = std::mt19937(92730469346UL);
};



}  // namespace Saiga
