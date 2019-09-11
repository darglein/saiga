/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"

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

    std::vector<VertexType> toLineList();

    void fromLineList();
    int numLines() { return indices.size() / 2; }

    std::vector<IndexType> indices;


    // ========================== Create Functions ============================
    // Create simple line geometry for debugging and visualization.
    // Each function defines the vertex attributes which are required.
    // Most of them require position only.
    // You can set a custom color by calling setColor(...).


    /**
     * Simple debug grid placed into the x-z plane with y=0.
     *
     * dimension: number of lines in x and z direction.
     * spacing:   distance between lines
     */
    void createGrid(const ivec2& dimension, const vec2& spacing);


    /**
     * Debug camera frustum.
     * Created by backprojecting the "far-plane corners" of the unit cube.
     * p = inv(proj) * corner
     *
     *
     * @param farPlaneLimit Distance at which the far plane should be drawn. -1 uses the original far plane.
     */
    void createFrustum(const mat4& proj, float farPlaneDistance = -1, bool vulkanTransform = false);


    /**
     * Similar to above but uses a computer vision K matrix and the image dimensions.
     */
    void createFrustumCV(const mat3& K, float farPlaneDistance, int w, int h);

    /**
     * Initializes a K matrix with a 90 degree FOV and creates the frustum.
     */
    void createFrustumCV(float farPlaneLimit, int w, int h);
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

#include "LineMeshCreate.h"
