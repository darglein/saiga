/**
 * Copyright (c) 2017 Darius Rückert
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
using TrajectoryType = AlignedVector<std::pair<int, SE3>>;

/**
 * Align the trajectories by minimizing the squared error:
 * sum_i (A.position() - B.position)^2
 */
SAIGA_VISION_API double align(TrajectoryType& A, TrajectoryType& B);

/**
 * Root mean squared relative pose error (rpe).
 * sum_i (a[i].inverse()*a[i-1] - b[i].inverse()*b[i-1])^2
 */
SAIGA_VISION_API std::vector<double> rpe(const TrajectoryType& A, const TrajectoryType& B);
/**
 * Root mean squared absolute trajectory error (ate)
 * sum_i (a[i] - b[i])^2
 */
SAIGA_VISION_API std::vector<double> ate(const TrajectoryType& A, const TrajectoryType& B);

}  // namespace Trajectory
}  // namespace Saiga
