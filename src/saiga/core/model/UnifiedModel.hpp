/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

//#include "UnifiedModel.h"

namespace Saiga
{

template <typename VertexType>
std::vector<VertexType> UnifiedMesh::VertexList() const
{
    throw std::runtime_error(
        "UnifiedModel::Mesh() not implemented for the specified vertex type. See saiga/core/model/UnifiedModel.h for "
        "more information.");
}

template <>
SAIGA_CORE_API std::vector<Vertex> UnifiedMesh::VertexList() const;

template <>
SAIGA_CORE_API std::vector<VertexC> UnifiedMesh::VertexList() const;

template <>
SAIGA_CORE_API std::vector<VertexNC> UnifiedMesh::VertexList() const;


template <>
SAIGA_CORE_API std::vector<VertexNT> UnifiedMesh::VertexList() const;


template <typename IndexType>
std::vector<Vector<IndexType, 3>> UnifiedMesh::TriangleIndexList() const
{
    std::vector<Vector<IndexType, 3>> result;
    result.reserve(triangles.size());
    for (auto& t : triangles)
    {
        result.emplace_back(t.cast<IndexType>());
    }
    return result;
}

template <typename IndexType>
std::vector<Vector<IndexType, 2>> UnifiedMesh::LineIndexList() const
{
    std::vector<Vector<IndexType, 2>> result;
    result.reserve(lines.size());
    for (auto& t : lines)
    {
        result.emplace_back(t.cast<IndexType>());
    }
    return result;
}

}  // namespace Saiga
