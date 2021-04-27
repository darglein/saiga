/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/aabb.h"
#include "saiga/core/math/math.h"



namespace Saiga
{
/**
 * The base class for all meshes.
 *
 * - TriangleMesh
 * - LineMesh
 * - PointMesh
 *
 * The only thing in common is the vector of vertices which you can find here.
 * The vertex type must have a vec4 position member.
 */

template <typename _VertexType>
class Mesh
{
   public:
    using VertexType = _VertexType;


    std::vector<VertexType> vertices;

    /*
     * Transforms mesh with given matrix.
     * v[i] = T * v[i]
     */
    void transform(const mat4& T);

    void clear() { vertices.clear(); }

    /*
     * Adds vertex to mesh
     * return: index of new vertex
     */
    int addVertex(const VertexType& v)
    {
        vertices.push_back(v);
        return vertices.size() - 1;
    }


    /**
     * Computes the size in bytes for this mesh.
     * Returns the actual RAM size therefore uses capacity.
     */
    size_t size() const { return sizeof(VertexType) * vertices.capacity(); }

    void freeMemory()
    {
        clear();
        vertices.shrink_to_fit();
    }

    AABB aabb() const;

    void normalizePosition()
    {
        auto box = aabb();
        mat4 t   = translate(-box.getPosition());
        transform(t);
    }


    void normalizeScale(float padding = 0)
    {
        auto box = aabb();

        float total_max = box.max.array().maxCoeff();
        float total_min = box.min.array().minCoeff();
        float size = total_max - total_min;
        const auto s = 1 / (size + 2 * padding * size);
        //        float s = 1.0 /
        mat4 t = scale(make_vec3(s));
        transform(t);
    }

    // Sets the .color attribute of all vertices to the given color
    template <typename ColorType>
    void setColor(const ColorType& color);
};


template <typename VertexType>
void Mesh<VertexType>::transform(const mat4& T)
{
    for (auto& v : vertices)
    {
        // Make sure it works for user defined w components
        vec4 p     = make_vec4(make_vec3(v.position), 1);
        p          = T * p;
        v.position = make_vec4(make_vec3(p), v.position[3]);
    }
}

template <typename VertexType>
AABB Mesh<VertexType>::aabb() const
{
    AABB box;
    box.makeNegative();

    for (auto&& v : vertices)
    {
        box.growBox(make_vec3(v.position));
    }
    return box;
}

template <typename VertexType>
template <typename ColorType>
void Mesh<VertexType>::setColor(const ColorType& color)
{
    for (auto&& v : vertices)
    {
        v.color = color;
    }
}


}  // namespace Saiga
