/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/math.h"
#include "saiga/geometry/vertex.h"

#include <vector>

namespace Saiga {


template<typename VertexType>
class PointCloud
{
public:
    std::vector<VertexType> points;
};




}
