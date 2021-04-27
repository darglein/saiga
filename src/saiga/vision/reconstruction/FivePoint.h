/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/math/random.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/util/Ransac.h"

#include "Epipolar.h"
#include "unsupported/Eigen/Polynomials"

#include <random>

namespace Saiga
{
/*
 * This function is taken from opencv.
 * Source: https://github.com/opencv/opencv/blob/master/modules/calib3d/src/five-point.cpp
 *
 * This is a 5-point algorithm contributed to OpenCV by the author, Bo Li.
 * It implements the 5-point algorithm solver from Nister's paper:
 * Nister, An efficient solution to the five-point relative pose problem, PAMI, 2004.
 */
SAIGA_VISION_API void constructFivePointMatrix(double* e, double* A);

/**
 *  An Efficient Solution to the Five-Point Relative Pose Problem
 *  https://pdfs.semanticscholar.org/c288/7c83751d2c36c63139e68d46516ba3038909.pdf
 *
 * Computes the valid essential matrices from 5 point correspondences.
 * There are up to 10 solutions returned in es.
 *
 * This is basically a copy-paste from the opencv implementation, but ported to Eigen.
 *
 * The returned int is the number of solutions.
 */
SAIGA_VISION_API int fivePointNister(Vec2* points0, Vec2* points1, std::vector<Mat3>& es);

/**
 * Given the up to 10 essential matrices from the 5-point algorithm,
 * this function computes the best one including the relative transformation.
 *
 * If non of these are valid, false is returned.
 */
SAIGA_VISION_API bool bestEUsing6Points(const std::vector<Mat3>& es, const Vec2* points1, const Vec2* points2,
                                        Mat3& bestE, SE3& bestT);


SAIGA_VISION_API int computeERansac(ArrayView<const Vec2> points1, ArrayView<const Vec2> points2,
                                    const RansacParameters& params, Mat3& bestE, SE3& bestT,
                                    std::vector<int>& bestInlierMatches, std::vector<char>& inlierMask);



class SAIGA_VISION_API FivePointRansac : public RansacBase<FivePointRansac, std::pair<Mat3, SE3>, 6>
{
    using Model = std::pair<Mat3, SE3>;
    using Base  = RansacBase<FivePointRansac, Model, 6>;

   public:
    FivePointRansac() {}
    FivePointRansac(const RansacParameters& params) : Base(params) {}

    int solve(ArrayView<const Vec2> _points1, ArrayView<const Vec2> _points2, Mat3& bestE, SE3& bestT,
              std::vector<int>& bestInlierMatches, std::vector<char>& inlierMask);



    bool computeModel(const Subset& set, Model& model);

    double computeResidual(const Model& model, int i);

    ArrayView<const Vec2> points1;
    ArrayView<const Vec2> points2;
};



}  // namespace Saiga
