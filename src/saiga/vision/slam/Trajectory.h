/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/statistics.h"
#include "saiga/vision/VisionTypes.h"

/**
 * Compare the trajectory of two slam systems or to ground truth data.
 * All poses must be given in Camera->World transformations. (So invert, if the pose projects from world to camera)
 * The 'int' parameter in the pair is the frame id. This can be used for keyframe-based system
 * to get a more meaningful relative pose error, because the error is normalized to the number of frames
 * in-between.
 *
 *
 * The typical usage is:
 *
 * TrajectoryType data;
 * TrajectoryType groundTruth;
 *
 * for(...)
 *      // fill data and groundTruth
 *
 * align(data,groundTruth);
 * std::cout << "rpe: " << rpe(data,groundTruth) << std::endl;
 * std::cout << "ate: " << ate(data,groundTruth) << std::endl;
 *
 */
namespace Saiga
{
namespace Trajectory
{
struct Observation
{
    SE3 estimate;
    SE3 ground_truth;
};

struct SAIGA_VISION_API Scene
{
    AlignedVector<Observation> vertices;

    // right mult from estimate to ground_truth space
    SE3 extrinsics;

    SE3 transformation;
    double scale = 1;

    SE3 TransformVertex(const SE3& v) const
    {
        SE3 e = v;
        e.translation() *= scale;
        e = e * extrinsics;
        e = transformation * e;
        return e;
    }

    double chi2() const
    {
        double error = 0;
        for (auto& o : vertices)
        {
            SE3 e         = TransformVertex(o.estimate);
            Vec3 residual = e.translation() - o.ground_truth.translation();
            error += residual.squaredNorm();
        }
        return error;
    }

    double rmse() const { return sqrt(chi2() / vertices.size()); }



    // Optimization parameters
    bool optimize_scale      = true;
    bool optimize_extrinsics = false;

    void InitialAlignment();
#ifdef SAIGA_USE_CERES
    void OptimizeCeres();
#endif
};


SAIGA_VISION_API std::ostream& operator<<(std::ostream& strm, const Scene& scene);

using TrajectoryType = AlignedVector<std::pair<int, SE3>>;

/**
 * Align the trajectories by minimizing the squared error:
 * sum_i (A.position() - B.position)^2
 */
SAIGA_VISION_API extern std::pair<SE3, double> align(ArrayView<std::pair<int, SE3>> A, ArrayView<std::pair<int, SE3>> B,
                                                     bool computeScale);

/**
 * Root mean squared relative pose error (rpe).
 * sum_i (a[i].inverse()*a[i-1] - b[i].inverse()*b[i-1])^2
 */
SAIGA_VISION_API extern std::vector<double> rpe(ArrayView<const std::pair<int, SE3>> A,
                                                ArrayView<const std::pair<int, SE3>> B, int difference);
/**
 * Root mean squared absolute trajectory error (ate)
 * sum_i (a[i] - b[i])^2
 */
SAIGA_VISION_API extern std::vector<double> ate(ArrayView<const std::pair<int, SE3>> A,
                                                ArrayView<const std::pair<int, SE3>> B);

/**
 * RMS Absolute Rotational error in degrees (!!!)
 */
SAIGA_VISION_API extern std::vector<double> are(ArrayView<const std::pair<int, SE3>> A,
                                                ArrayView<const std::pair<int, SE3>> B);

}  // namespace Trajectory
}  // namespace Saiga
