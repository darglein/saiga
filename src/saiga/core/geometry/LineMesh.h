/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"

#include "Frustum.h"
#include "Mesh.h"
#include "triangle.h"
#include "vertex.h"

namespace Saiga
{
template <typename _VertexType, typename _IndexType = uint32_t>
class LineMesh : public Mesh<_VertexType>
{
   public:
    using VertexType = _VertexType;
    using IndexType  = _IndexType;

    using Base = Mesh<VertexType>;
    using Base::aabb;
    using Base::addVertex;
    using Base::size;
    using Base::vertices;

    using Line = Vector<IndexType, 2>;

    std::vector<VertexType> toLineList();

    void fromLineList();
    int numLines() { return lines.size(); }

    std::vector<Line> lines;
};


template <typename VertexType, typename IndexType>
std::vector<VertexType> LineMesh<VertexType, IndexType>::toLineList()
{
    SAIGA_ASSERT(lines.size() == 0);
    std::vector<VertexType> res(lines.size() * 2);
    for (unsigned int i = 0; i < lines.size(); i++)
    {
        res[i * 2]     = vertices[lines[i](0)];
        res[i * 2 + 1] = vertices[lines[i](1)];
    }
    return res;
}

template <typename VertexType, typename IndexType>
void LineMesh<VertexType, IndexType>::fromLineList()
{
    lines.resize(vertices.size() / 2);
    for (size_t i = 0; i < lines.size(); ++i)
    {
        lines[i] = Line(i * 2, i * 2 + 1);
    }
}



}  // namespace Saiga
