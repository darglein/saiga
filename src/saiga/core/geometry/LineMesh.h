/**
 * Copyright (c) 2017 Darius Rückert
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


    void createAABB(const AABB& box);


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

    /**
     * Initializes a K matrix with a 90 degree FOV and creates the frustum.
     */
    void createFrustum(const Frustum& frustum);
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
    for (IndexType i = 0; i < lines.size(); ++i)
    {
        lines[i] = ivec2(i * 2, i * 2 + 1);
    }
}



}  // namespace Saiga

#include "LineMeshCreate.h"
