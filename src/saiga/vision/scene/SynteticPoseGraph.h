/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/scene/PoseGraph.h"

namespace Saiga
{
namespace SyntheticPoseGraph
{
SAIGA_VISION_API PoseGraph Linear(int num_vertices, int num_connections);
SAIGA_VISION_API PoseGraph Circle(double radius, int num_vertices, int num_connections);

SAIGA_VISION_API PoseGraph CircleWithDrift(double radius, int num_vertices, int num_connections, double sigma,
                                           double sigma_scale);
}  // namespace SyntheticPoseGraph
}  // namespace Saiga
