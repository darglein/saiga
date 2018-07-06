/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/glm.h"

#include "saiga/geometry/vertex.h"
#include "saiga/geometry/aabb.h"
#include "saiga/geometry/triangle.h"

#include "saiga/util/assert.h"


namespace Saiga {


template<typename VertexType, typename IndexType = uint32_t>
class LineMesh
{
public:



public:
    std::vector<VertexType> vertices;
    std::vector<IndexType> indices;
};


}
