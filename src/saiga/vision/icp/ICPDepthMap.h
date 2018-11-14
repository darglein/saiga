/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <vector>

#include "saiga/vision/icp/ICPAlign.h"
#include "saiga/vision/Depthmap.h"


namespace Saiga {
namespace ICP {

/**
 * Search for point-to-point correspondences between two point clouds from depth images.
 */
SAIGA_GLOBAL std::vector<Correspondence> projectiveCorrespondences(
        Depthmap::DepthPointCloud refPc, Depthmap::DepthNormalMap refNormal,
        Depthmap::DepthPointCloud srcPc, Depthmap::DepthNormalMap srcNormal,
        SE3 T, Intrinsics4 camera,
        double distanceThres = 0.1, double cosNormalThres = 0.9,
        int searchRadius = 0, bool useInvDepthAsWeight = true,
        bool scaleDistanceThresByDepth = true
        );


/**
 * Aligns two depth images.
 * This function:
 *  - Computes the point clouds + normal maps
 *  - finds projective correspondences (function above) with default params
 *  - finds the rigid transformation between the point clouds with point-to-plane metric (see ICP align)
 */
SAIGA_GLOBAL SE3 alignDepthMaps(Depthmap::DepthMap referenceDepthMap, Depthmap::DepthMap sourceDepthMap, SE3 guess, Intrinsics4 camera, int iterations);

}
}
