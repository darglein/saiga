/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "UnifiedModel.h"

namespace Saiga
{
template <typename VertexType>
std::vector<VertexType> UnifiedModel::VertexList() const
{
    throw std::runtime_error(
        "UnifiedModel::Mesh() not implemented for the specified vertex type. See saiga/core/model/UnifiedModel.h for "
        "more information.");
}

template <>
std::vector<Vertex> UnifiedModel::VertexList() const;

template <>
std::vector<VertexNC> UnifiedModel::VertexList() const;


template <>
std::vector<VertexNT> UnifiedModel::VertexList() const;


template <>
std::vector<VertexNTD> UnifiedModel::VertexList() const;


template <typename IndexType>
std::vector<Vector<IndexType, 3>> UnifiedModel::IndexList() const
{
    std::vector<Vector<IndexType, 3>> result;
    result.reserve(triangles.size());
    for (auto& t : triangles)
    {
        result.emplace_back(t.cast<IndexType>());
    }
    return result;
}

}  // namespace Saiga
