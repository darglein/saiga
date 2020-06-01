/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/time/all.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/ba/BAWrapper.h"
#include "saiga/vision/reconstruction/EightPoint.h"
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
class SAIGA_VISION_API TwoViewReconstruction
{
   public:
    TwoViewReconstruction();

    // must be called once before running compute!
    void init(const RansacParameters& fivePointParams);

    void compute(ArrayView<const Vec2> points1, ArrayView<const Vec2> points2, int threads);
    int NumPointWithAngleAboveThreshold(double angle);

    double medianAngle();
    double medianAngleByDepth();

    // scales the scene so that the median depth is d
    void setMedianDepth(double d);
    double getMedianDepth();

    // optimize with bundle adjustment
    int optimize(int its, float thresholdChi1);

    SE3& pose1() { return scene.images[0].se3; }
    SE3& pose2() { return scene.images[1].se3; }

    void clear();


    int N = 0;
    Mat3 E;
    std::vector<int> inliers;
    std::vector<char> inlierMask;
    int inlierCount = 0;

    Scene scene;

    std::vector<double> tmpArray;
    FivePointRansac fpr;
    //    Triangulation<double> triangulation;

    OptimizationOptions op_options;
    BAOptions ba_options;
    BAWrapper ba;
};


class SAIGA_VISION_API TwoViewReconstructionEightPoint : public TwoViewReconstruction
{
   public:
    // must be called once before running compute!
    void init(const RansacParameters& ransac_params, Intrinsics4 K);
    void compute(ArrayView<const Vec2> points1, ArrayView<const Vec2> points2, int threads);

   private:
    Mat3 F;
    EightPointRansac epr;
};


}  // namespace Saiga
