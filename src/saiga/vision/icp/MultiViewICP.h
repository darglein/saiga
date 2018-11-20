/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <vector>
#include "saiga/vision/Depthmap.h"
#include "saiga/vision/icp/ICPDepthMap.h"

namespace Saiga
{
namespace ICP
{
/**
 * Computes all pairwise point correspondences and solves the global system with all residuals.
 * Point-to-Plane metric is used.
 */
SAIGA_GLOBAL void multiViewICP(const std::vector<Depthmap::DepthMap>& depthMaps, std::vector<SE3>& guesses,
                               Intrinsics4 camera, int iterations,
                               ProjectiveCorrespondencesParams params = ProjectiveCorrespondencesParams());


/**
 * Aligns all depthmaps relative to the first one.
 */
SAIGA_GLOBAL void multiViewICPSimple(const std::vector<Depthmap::DepthMap>& depthMaps, std::vector<SE3>& guesses,
                                     Intrinsics4 camera, int iterations,
                                     ProjectiveCorrespondencesParams params = ProjectiveCorrespondencesParams());


}  // namespace ICP
}  // namespace Saiga
