/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/util/Ransac.h"

#include <optional>

namespace Saiga
{
/**
 * Sources:
 * The code here is a mixed implementation between Opencv's and Colmaps's p3p implementation.
 * Overall it should be about twice as fast.
 *
 * https://github.com/opencv/opencv/blob/master/modules/calib3d/src/p3p.cpp
 * https://github.com/colmap/colmap/blob/dev/src/estimators/absolute_pose.cc
 *
 */
class SAIGA_VISION_API P3P
{
   public:
    /**
     * Solves the P3P Problem.
     *
     * Input:
     * worldPoints:             size 3 array with 3D points
     * normalizedImagePoints:   size 3 array with 2D points in normalized image space (without K)
     *
     * Output:
     * Up to 4 solutions.
     * Returns the number of solutions
     */
    static int solve(ArrayView<const Vec3> worldPoints, ArrayView<const Vec2> normalizedImagePoints,
                     std::array<SE3, 4>& results);


    /**
     * Filter out the correct solution by using a 4th point.
     * There might not be solution at all -> therefore optional
     */
    static std::optional<SE3> bestSolution(ArrayView<const SE3> candidates, const Vec3& fourthWorldPoint,
                                           const Vec2& fourthImagePoint);


    /**
     * Calls both functions above.
     * The input array size must be 4.
     *
     * This is the recommended function to call.
     */
    static std::optional<SE3> solve4(ArrayView<const Vec3> worldPoints, ArrayView<const Vec2> normalizedImagePoints);
};


class SAIGA_VISION_API P3PRansac : public RansacBase<P3PRansac, SE3, 4>
{
    using Model = SE3;
    using Base  = RansacBase<P3PRansac, SE3, 4>;

   public:
    P3PRansac() {}
    P3PRansac(const RansacParameters& params) : Base(params) {}



    int solve(ArrayView<const Vec3> _worldPoints, ArrayView<const Vec2> _normalizedImagePoints, SE3& bestT,
              std::vector<int>& bestInlierMatches, std::vector<char>& inlierMask)
    {
        worldPoints           = _worldPoints;
        normalizedImagePoints = _normalizedImagePoints;

        int idx;
        idx        = compute(_worldPoints.size());
        bestT      = models[idx];
        inlierMask = inliers[idx];

        bestInlierMatches.clear();
        bestInlierMatches.reserve(numInliers[idx]);
        for (int i = 0; i < N; ++i)
        {
            if (inliers[idx][i]) bestInlierMatches.push_back(i);
        }

        return numInliers[idx];
    }



    bool computeModel(const Subset& set, Model& model)
    {
        std::array<Vec3, 4> A;
        std::array<Vec2, 4> B;

        for (auto i : Range(0, (int)set.size()))
        {
            A[i] = worldPoints[set[i]];
            B[i] = normalizedImagePoints[set[i]];
        }

        auto res = P3P::solve4(A, B);


        if (res.has_value())
        {
            model = res.value();
            return true;
        }

        return false;
    }

    double computeResidual(const Model& model, int i)
    {
        Vec2 ip = (model * worldPoints[i]).hnormalized();
        return (ip - normalizedImagePoints[i]).squaredNorm();
    }

   private:
    ArrayView<const Vec3> worldPoints;
    ArrayView<const Vec2> normalizedImagePoints;
};


}  // namespace Saiga
