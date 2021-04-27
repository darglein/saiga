/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/geometry/iRect.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/DataStructures/ArrayView.h"

#include "RectilinearCover.h"
#include "aabb.h"
#include "intersection.h"
#include "ray.h"
#include "triangle.h"

namespace Saiga
{
namespace RectangularDecomposition
{
SAIGA_CORE_API std::vector<ivec3> RemoveDuplicates(PointView points);



// The trivial decomposition converts each element into a 1x1 rectangle.
SAIGA_CORE_API RectangleList DecomposeTrivial(PointView points);

// Combines all neighboring elements in x-direction with the same (y,z) coordinate.
SAIGA_CORE_API RectangleList DecomposeRowMerge(PointView points);

// Combines all neighboring elements in x-direction with the same (y,z) coordinate.
SAIGA_CORE_API RectangleList DecomposeOctTree(PointView points, float merge_factor = 1.0, bool merge_layer = true);

}  // namespace RectangularDecomposition
}  // namespace Saiga
