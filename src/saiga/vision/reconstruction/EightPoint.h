/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/math/HomogeneousLSE.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/util/Ransac.h"

#include "Epipolar.h"

#include <random>

namespace Saiga
{
// Computes the fundamental matrix F from 8 point correspondences.
SAIGA_VISION_API Mat3 FundamentalMatrixEightPoint(Vec2* points0, Vec2* points1);

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
        return FundamentalMatrixEightPoint(points1, points2);
    }


    int computeFRansac(Vec2* points1, Vec2* points2, int N, Mat3& bestF, std::vector<int>& bestInlierMatches)
    {
        int maxIterations        = 1000;
        constexpr int sampleSize = 8;
        double epipolarTheshold  = 1.5;
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



class SAIGA_VISION_API EightPointRansac : public RansacBase<EightPointRansac, Mat3, 8>
{
    using Model = Mat3;
    using Base  = RansacBase<EightPointRansac, Model, 8>;

   public:
    EightPointRansac() {}
    EightPointRansac(const RansacParameters& params) : Base(params) {}

    int solve(ArrayView<const Vec2> _points1, ArrayView<const Vec2> _points2, Mat3& bestF,
              std::vector<int>& bestInlierMatches, std::vector<char>& inlierMask);

    bool computeModel(const Subset& set, Model& model);

    double computeResidual(const Model& model, int i);

    ArrayView<const Vec2> points1;
    ArrayView<const Vec2> points2;
};



}  // namespace Saiga
