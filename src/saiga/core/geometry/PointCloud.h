/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"

#include "vertex.h"

#include <vector>

namespace Saiga
{
template <typename VertexType>
class PointCloud
{
   public:
    std::vector<VertexType> points;
};



}  // namespace Saiga
