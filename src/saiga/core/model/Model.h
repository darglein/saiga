/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/triangle_mesh.h"



namespace Saiga
{
template <typename vertex_t, typename index_t>
class SAIGA_TEMPLATE TriangleModel
{
   public:
    using VertexType = vertex_t;
    using IndexType  = index_t;

    AABB boundingBox;
    TriangleMesh<vertex_t, index_t> mesh;

    void normalizePosition();

    /**
     * Normalize the scale so that no vertex lies outside of the range [-1,-1,-1]-[1,1,1].
     * Remark: This normalizes the positon beforehand.
     */
    void normalizeScale();
    /**
     * Transforms the vertices and normals that the up axis is Y when before the up axis was Z.
     *
     * Many 3D CAD softwares (i.e. Blender) are using a right handed coordinate system with Z pointing upwards.
     * This frameworks uses a right haned system with Y pointing upwards.
     */


    void ZUPtoYUP();
};

template <typename vertex_t, typename index_t>
void TriangleModel<vertex_t, index_t>::normalizePosition()
{
    mat4 t = identityMat4();
    mesh.transform(t);
    boundingBox.setPosition(make_vec3(0));
}



template <typename vertex_t, typename index_t>
void TriangleModel<vertex_t, index_t>::ZUPtoYUP()
{
    const mat4 m = make_mat4(1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1);
    mesh.transform(m);
    mesh.transformNormal(m);
}


}  // namespace Saiga
