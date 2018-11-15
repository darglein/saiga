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

struct SAIGA_GLOBAL DepthMapExtended
{
    Depthmap::DepthMap depth;
    Saiga::ArrayImage<Vec3> points;
    Saiga::ArrayImage<Vec3> normals;
    Intrinsics4 camera;
    SE3 pose; // W <- this

    DepthMapExtended(Depthmap::DepthMap depth, Intrinsics4 camera, SE3 pose);

};

/**
 * Search for point-to-point correspondences between two point clouds from depth images.
 */
SAIGA_GLOBAL std::vector<Correspondence> projectiveCorrespondences(
        const DepthMapExtended& ref, const DepthMapExtended& src,
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
SAIGA_GLOBAL SE3 alignDepthMaps(Depthmap::DepthMap referenceDepthMap, Depthmap::DepthMap sourceDepthMap, SE3 refPose, SE3 srcPose, Intrinsics4 camera, int iterations);

}
}
