/**
 * Copyright (c) 2021 Darius Rückert
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
struct SimpleVertex
{
    vec3 position;
    vec3 normal;

    // The triangle index and its barycentric coordinates
    int triangle_index;
    vec3 bary;
    float radius = 1;
};
using SimplePointCloud = std::vector<SimpleVertex>;

// The Sample Probability of the i-th triangle is: Area(triangle[i]) * weight[i]
SAIGA_VISION_API SimplePointCloud RandomSampleMesh(const std::vector<Triangle>& triangles,
                                                   const std::vector<float>& weights, int N);

// The search radius for each point is weights[i] * radius
SAIGA_VISION_API SimplePointCloud ReducePointsPoissonDisc(const SimplePointCloud& points, float radius);

// The Sample Probability of the i-th triangle is: Area(triangle[i]) * weight[i]
SAIGA_VISION_API SimplePointCloud MeshToPointCloudPoissonDisc2(const std::vector<Triangle>& triangles,
                                                               const std::vector<float>& weights, int max_samples,
                                                               float radius);


SAIGA_VISION_API float Distance(const std::vector<Triangle>& triangles, const vec3& p);

// Convert a list of triangles to a block-sparse TSDF
// This method is slow, because it computes the point-surface distances analytically.
SAIGA_VISION_API std::shared_ptr<SparseTSDF> MeshToTSDF(const std::vector<Triangle>& triangles, float voxel_size,
                                                        int r);
}  // namespace Saiga
