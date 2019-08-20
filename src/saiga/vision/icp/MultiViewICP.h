/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/icp/ICPDepthMap.h"
#include "saiga/vision/util/Depthmap.h"

#include <vector>

namespace Saiga
{
namespace ICP
{
/**
 * Computes all pairwise point correspondences and solves the global system with all residuals.
 * Point-to-Plane metric is used.
 */
SAIGA_VISION_API void multiViewICP(const std::vector<Depthmap::DepthMap>& depthMaps, AlignedVector<SE3>& guesses,
                                   Intrinsics4 camera, int iterations,
                                   ProjectiveCorrespondencesParams params = ProjectiveCorrespondencesParams());


/**
 * Aligns all depthmaps relative to the first one.
 */
SAIGA_VISION_API void multiViewICPSimple(const std::vector<Depthmap::DepthMap>& depthMaps, AlignedVector<SE3>& guesses,
                                         Intrinsics4 camera, int iterations,
                                         ProjectiveCorrespondencesParams params = ProjectiveCorrespondencesParams());


}  // namespace ICP
}  // namespace Saiga
