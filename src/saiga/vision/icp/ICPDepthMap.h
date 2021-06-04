/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/util/Depthmap.h"
#include "saiga/vision/icp/ICPAlign.h"

#include <vector>


namespace Saiga
{
namespace ICP
{
struct SAIGA_VISION_API DepthMapExtended
{
    Depthmap::DepthMap depth;
    Saiga::ArrayImage<Vec3> points;
    Saiga::ArrayImage<Vec3> normals;
    IntrinsicsPinholed camera;
    SE3 pose;  // W <- this

    DepthMapExtended(const Depthmap::DepthMap& depth, const IntrinsicsPinholed& camera, const SE3& pose);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



struct ProjectiveCorrespondencesParams
{
    double distanceThres           = 0.1;
    double cosNormalThres          = 0.9;
    int searchRadius               = 0;
    bool useInvDepthAsWeight       = true;
    bool scaleDistanceThresByDepth = true;
    int stride                     = 1;
};

/**
 * Search for point-to-point correspondences between two point clouds from depth images.
 */
SAIGA_VISION_API AlignedVector<Correspondence> projectiveCorrespondences(const DepthMapExtended& ref,
                                                                     const DepthMapExtended& src,
                                                                     const ProjectiveCorrespondencesParams& params);


/**
 * Aligns two depth images.
 * This function:
 *  - Computes the point clouds + normal maps
 *  - finds projective correspondences (function above) with default params
 *  - finds the rigid transformation between the point clouds with point-to-plane metric (see ICP align)
 */
SAIGA_VISION_API SE3 alignDepthMaps(Depthmap::DepthMap referenceDepthMap, Depthmap::DepthMap sourceDepthMap,
                                const SE3& refPose, const SE3& srcPose, const IntrinsicsPinholed& camera, int iterations,
                                ProjectiveCorrespondencesParams params = ProjectiveCorrespondencesParams());

}  // namespace ICP
}  // namespace Saiga
