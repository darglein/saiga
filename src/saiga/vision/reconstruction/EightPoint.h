/**
 * Copyright (c) 2021 Darius Rückert
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
SAIGA_VISION_API Mat3 FundamentalMatrixEightPoint(const Vec2* points0, const Vec2* points1);

SAIGA_VISION_API Mat3 NormalizePoints(const Vec2* src_points, Vec2* dst_points, int N);
SAIGA_VISION_API Mat3 FundamentalMatrixEightPointNormalized(const Vec2* points0, const Vec2* points1);


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
