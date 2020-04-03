/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/scene/PoseGraph.h"

namespace Saiga
{
namespace SyntheticPoseGraph
{
SAIGA_VISION_API PoseGraph CreateLinearPoseGraph(int num_vertices, int num_connections);
}
}  // namespace Saiga
