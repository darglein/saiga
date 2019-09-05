#pragma once

#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/util/Ransac.h"
// This code here is inspired (and partially copied) from Colmap.
// https://github.com/colmap/colmap
namespace Saiga
{
/**
 * Calculates a 3x3 homography matrix H so that
 * targetPoints[i] = H * sourcePoints[i]
 * This mapping is in 2d projective space -> H is up to a scale
 */
SAIGA_VISION_API Mat3 homography(ArrayView<const Vec2> points1, ArrayView<const Vec2> points2);


/**
 * The transformation error for a corresponding point pair.
 */
inline double homographyResidual(const Vec2& p1, const Vec2& p2, const Mat3& H)
{
    Vec3 p      = H * p1.homogeneous();
    double invz = 1.0 / p(2);
    Vec2 res(p2(0) - p(0) * invz, p2(1) - p(1) * invz);
    return res.squaredNorm();
}

#if 0
// solves H = aK * [R|t] for [R|t]
CameraExtrinsics getExtrinsicsFromHomography(const CameraIntrinsics& camera, const mat3d_t& H);

// solves H = a[R|t] for [R|t]
CameraExtrinsics getExtrinsicsFromHomography(const mat3d_t& H);

mat3d_t homographyRANSAC(const std::vector<vec2d_t>& sourcePoints, const std::vector<vec2d_t>& targetPoints,
                         std::vector<int>& outInliers, int numIterations = 1000, double inlierThreshold = 5,
                         int numSamples = 4);
#endif


class SAIGA_VISION_API HomographyRansac : public RansacBase<HomographyRansac, Mat3, 4>
{
    using Base  = RansacBase<HomographyRansac, Mat3, 4>;
    using Model = Mat3;

   public:
    HomographyRansac(const RansacParameters& params) : Base(params) {}

    int solve(ArrayView<const Vec2> _points1, ArrayView<const Vec2> _points2, Mat3& bestH)
    {
        points1 = _points1;
        points2 = _points2;

        int idx;

#pragma omp parallel num_threads(params.threads)
        {
            idx = compute(points1.size());
        }
        bestH = models[idx];
        return numInliers[idx];
    }



    bool computeModel(const Subset& set, Model& model)
    {
        std::array<Vec2, 4> p1;
        std::array<Vec2, 4> p2;
        for (auto i : Range(0, (int)set.size()))
        {
            p1[i] = points1[set[i]];
            p2[i] = points2[set[i]];
        }
        model = homography(p1, p2);
        return true;
    }

    double computeResidual(const Model& model, int i) { return homographyResidual(points1[i], points2[i], model); }

    ArrayView<const Vec2> points1;
    ArrayView<const Vec2> points2;
};



}  // namespace Saiga
