/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "UnifiedModel.h"

namespace Saiga
{
template <typename VertexType, typename IndexType>
TriangleMesh<VertexType, IndexType> UnifiedModel::Mesh() const
{
    throw std::runtime_error(
        "UnifiedModel::Mesh() not implemented for the specified vertex type. See saiga/core/model/UnifiedModel.h for "
        "more information.");
}

template <>
TriangleMesh<Vertex, uint32_t> UnifiedModel::Mesh() const;

template <>
TriangleMesh<VertexNC, uint32_t> UnifiedModel::Mesh() const;


template <>
TriangleMesh<VertexNT, uint32_t> UnifiedModel::Mesh() const;


template <>
TriangleMesh<VertexNTD, uint32_t> UnifiedModel::Mesh() const;



}  // namespace Saiga
