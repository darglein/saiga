/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/image.h"
#include "saiga/vision/VisionTypes.h"

namespace Saiga
{
namespace Depthmap
{
/**
 *
 * Floating point depth images.
 * The depth is the actual world space depth of the given point.
 * For Kinect-like depth sensors the depth is given in meters.
 *
 */
using DepthType       = float;
using DepthMap        = ImageView<DepthType>;
using DepthPointCloud = ImageView<Vec3>;
using DepthNormalMap  = ImageView<Vec3>;


/**
 * Projects the depth points to camera space.
 * For invalid depth value the point will be at infinity.
 */
SAIGA_VISION_API void toPointCloud(DepthMap dm, DepthPointCloud pc, const IntrinsicsPinholed& camera);


/**
 * Computes the normals in camera space for each point with
 * the cross product of central differences. If some points are invalid,
 * forward/backward differences are used.
 *
 * All invalid normals will be infinity.
 */
SAIGA_VISION_API void normalMap(DepthPointCloud pc, DepthNormalMap normals);


/**
 * Exports the point cloud in ASCII .ply format.
 * This can be viewed for example with meshlab.
 */
SAIGA_VISION_API void savePointCloudPLY(const std::string& file, DepthPointCloud pc, const Vec3& color = Vec3(1, 0, 0));


}  // namespace Depthmap
}  // namespace Saiga
