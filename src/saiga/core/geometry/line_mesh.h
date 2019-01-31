/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/core/util/math.h"

#include "aabb.h"
#include "triangle.h"
#include "vertex.h"


namespace Saiga
{
template <typename VertexType, typename IndexType = uint32_t>
class LineMesh
{
   public:
    std::vector<VertexType> toLineList();

    void fromLineList();
    int numLines() { return indices.size() / 2; }

   public:
    std::vector<VertexType> vertices;
    std::vector<IndexType> indices;
};


template <typename VertexType, typename IndexType>
std::vector<VertexType> LineMesh<VertexType, IndexType>::toLineList()
{
    SAIGA_ASSERT(indices.size() % 2 == 0);
    std::vector<VertexType> res(indices.size());
    for (unsigned int i = 0; i < indices.size(); i++)
    {
        res[i] = vertices[indices[i]];
    }
    return res;
}

template <typename VertexType, typename IndexType>
void LineMesh<VertexType, IndexType>::fromLineList()
{
    indices.resize(vertices.size());
    for (IndexType i = 0; i < indices.size(); ++i) indices[i] = i;
}


}  // namespace Saiga
