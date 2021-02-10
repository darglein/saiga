/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/geometry/all.h"
#include "saiga/core/image/all.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/VisionTypes.h"

#include "SparseTSDF.h"

#include <set>
namespace Saiga
{
SAIGA_VISION_API std::vector<vec3> MeshToPointCloud(const std::vector<Triangle>& triangles, int N);


SAIGA_VISION_API float Distance(const std::vector<Triangle>& triangles, const vec3& p);

// Convert a list of triangles to a block-sparse TSDF
// This method is slow, because it computes the point-surface distances analytically.
SAIGA_VISION_API std::shared_ptr<SparseTSDF> MeshToTSDF(const std::vector<Triangle>& triangles, float voxel_size,
                                                        int r);
}  // namespace Saiga
