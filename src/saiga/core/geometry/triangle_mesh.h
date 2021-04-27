/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/aabb.h"
#include "saiga/core/geometry/triangle.h"
#include "saiga/core/geometry/vertex.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"

#include "Mesh.h"

#include <algorithm>
#include <cstring>
#include <numeric>

namespace Saiga
{
/*
 * Data structur for simple triangle meshes.
 * Can be turned into a IndexedVertexBuffer for drawing with OpenGL
 */

template <typename vertex_t, typename index_t>
class TriangleMesh : public Mesh<vertex_t>
{
   public:
    using VertexType = vertex_t;
    using IndexType  = index_t;

    using Base = Mesh<vertex_t>;
    using Base::aabb;
    using Base::addVertex;
    using Base::size;
    using Base::vertices;

    using Face = Vector<IndexType, 3>;

    void transformNormal(const mat4& trafo);

    /*
     * Deletes all vertices and faces.
     */

    void clear()
    {
        Base::clear();
        faces.resize(0);
    }



    /*
     * Adds face to mesh.
     * The indices of the face should match existing vertices
     * return: index of new face
     */
    int addFace(const Face& f)
    {
        faces.push_back(f);
        return faces.size() - 1;
    }

    int addFace(index_t f[3]) { return addFace(Face(f[0], f[1], f[2])); }
    int addFace(index_t v0, index_t v1, index_t v2) { return addFace(Face(v0, v1, v2)); }

    /*
     * Adds given vertices and the 2 corresponding triangles to mesh
     */

    void addQuad(vertex_t verts[4]);
    void addTriangle(vertex_t verts[3]);
    void addTriangle(const Triangle& t);

    /*
     * Adds 2 Triangles given by 4 vertices and form a quad.
     * The vertices should be orderd counter clockwise
     */

    void addQuad(index_t inds[4]);


    /*
     * Subdivides the triangle at index 'face' into 4 triangles.
     * The new triangles will be added to the mesh and the old will be overwritten
     */
    void subdivideFace(int face);

    /*
     * Inverts the triangle at index 'face'.
     * The order of the indices will be reversed.
     */
    void invertFace(int face);
    void invertMesh();

    /*
     * Converts the index face data structur to a simple triangle list.
     */

    std::vector<Triangle> toTriangleList() const;

    /*
     * Adds the complete mesh 'other' to the current mesh.
     */
    void addMesh(const TriangleMesh<vertex_t, index_t>& other);

    template <typename mesh_vertex_t, typename mesh_index_t>
    void addMesh(const TriangleMesh<mesh_vertex_t, mesh_index_t>& other);


    /**
     * Computes the per vertex normal by weighting each face normal by its surface area.
     */
    void computePerVertexNormal();

    /**
     * Removes all vertices that are not referenced by a triangle.
     * Computes the new vertex indices for each triangle.
     */
    void removeUnusedVertices();


    /**
     * Computes the size in bytes for this triangle mesh.
     */
    size_t size() const;
    void free();

    int numIndices() const { return faces.size() * 3; }



    bool isValid() const;

    bool empty() const { return faces.empty() || vertices.empty(); }

    /**
     * Sorts the vertices by (x,y,z) lexical.
     * The face indices are correct to match the new vertices.
     */
    void sortVerticesByPosition(double epsilon = 1e-5);


    /**
     * Removes subsequent vertices if they have identical position.
     * It make sense to call it after 'sortVerticesByPosition'.
     *
     * The face indices are updated accordingly.
     */
    void removeSubsequentDuplicates(double epsilon = 1e-5);

    /**
     * Removes all triangles, which reference a vertex twice
     */
    void removeDegenerateFaces();

    float distancePointMesh(const vec3& x);

    template <typename v, typename i>
    friend std::ostream& operator<<(std::ostream& os, const TriangleMesh<v, i>& dt);


    std::vector<index_t> getIndexList() const
    {
        std::vector<index_t> indices(numIndices());
        std::copy(&faces[0](0), &faces[0](0) + numIndices(), indices.data());
        return indices;
    }

    /**
     * Writes this mesh in OFF format to the given output stream.
     */
    //    void saveMeshOff(std::ostream& strm) const;
    //    void saveMeshOffColor(std::ostream& strm) const;

   public:
    //    std::vector<vertex_t> vertices;
    std::vector<Face> faces;
};



template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::transformNormal(const mat4& trafo)
{
    for (vertex_t& v : vertices)
    {
        vec4 p   = make_vec4(make_vec3(v.normal), 0);
        p        = trafo * p;
        v.normal = make_vec4(make_vec3(p), v.normal[3]);
    }
}



template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::addQuad(vertex_t verts[])
{
    int index = vertices.size();
    for (int i = 0; i < 4; i++)
    {
        addVertex(verts[i]);
    }

    faces.push_back(Face(index, index + 1, index + 2));
    faces.push_back(Face(index, index + 2, index + 3));
}



template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::addTriangle(vertex_t verts[])
{
    int index = vertices.size();
    for (int i = 0; i < 3; i++)
    {
        addVertex(verts[i]);
    }

    faces.push_back(Face(index, index + 1, index + 2));
}

template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::addTriangle(const Triangle& t)
{
    vertex_t ts[3];
    ts[0].position = make_vec4(t.a, 1);
    ts[1].position = make_vec4(t.b, 1);
    ts[2].position = make_vec4(t.c, 1);
    addTriangle(ts);
}


template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::addQuad(index_t inds[])
{
    faces.push_back(Face(inds[0], inds[1], inds[2]));
    faces.push_back(Face(inds[2], inds[3], inds[0]));
}



template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::subdivideFace(int f)
{
    Face face = faces[f];

    // create 3 new vertices in the middle of the edges

    auto p1 = vertices[face(0)].position;
    auto p2 = vertices[face(1)].position;
    auto p3 = vertices[face(2)].position;

    int v1 = addVertex(vertex_t(vec4((p1 + p2) / 2.0f)));
    int v2 = addVertex(vertex_t(vec4((p1 + p3) / 2.0f)));
    int v3 = addVertex(vertex_t(vec4((p2 + p3) / 2.0f)));


    faces.push_back(Face(face(1), v3, v1));

    faces.push_back(Face(face(2), v2, v3));

    faces.push_back(Face(v1, v3, v2));
    faces[f] = Face(face(0), v1, v2);
}



template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::invertFace(int f)
{
    Face& face = faces[f];
    Face face2;
    face2(0) = face(2);
    face2(1) = face(1);
    face2(2) = face(0);
    face     = face2;
}

template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::invertMesh()
{
    for (Face& face : faces)
    {
        Face face2;
        face2(0) = face(2);
        face2(1) = face(1);
        face2(2) = face(0);
        face     = face2;
    }

    for (vertex_t& v : vertices)
    {
        v.normal = -v.normal;
    }
}

template <typename vertex_t, typename index_t>
std::vector<Triangle> TriangleMesh<vertex_t, index_t>::toTriangleList() const
{
    std::vector<Triangle> output;
    Triangle t;
    for (const Face& f : faces)
    {
        t.a = make_vec3(vertices[f(0)].position);
        t.b = make_vec3(vertices[f(1)].position);
        t.c = make_vec3(vertices[f(2)].position);
        output.push_back(t);
    }
    return output;
}

template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::addMesh(const TriangleMesh<vertex_t, index_t>& other)
{
    int oldVertexCount = this->vertices.size();
    for (vertex_t v : other.vertices)
    {
        this->vertices.push_back(v);
    }

    for (Face f : other.faces)
    {
        f(0) += oldVertexCount;
        f(1) += oldVertexCount;
        f(2) += oldVertexCount;
        this->addFace(f);
    }
}

template <typename vertex_t, typename index_t>
template <typename mesh_vertex_t, typename mesh_index_t>
void TriangleMesh<vertex_t, index_t>::addMesh(const TriangleMesh<mesh_vertex_t, mesh_index_t>& other)
{
    int oldVertexCount = this->vertices.size();
    for (vertex_t v : other.vertices)
    {
        this->vertices.push_back(v);
    }

    for (auto f : other.faces)
    {
        f(0) += oldVertexCount;
        f(1) += oldVertexCount;
        f(2) += oldVertexCount;
        this->addFace(f);
    }
}

template <typename vertex_t, typename index_t>
size_t TriangleMesh<vertex_t, index_t>::size() const
{
    return faces.capacity() * sizeof(Face) + vertices.capacity() * sizeof(vertex_t) +
           sizeof(TriangleMesh<vertex_t, index_t>);
}

template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::free()
{
    faces.clear();
    faces.shrink_to_fit();
    vertices.clear();
    vertices.shrink_to_fit();
}

template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::computePerVertexNormal()
{
    //#pragma omp parallel for
    for (int i = 0; i < (int)vertices.size(); ++i)
    {
        // Note:
        // We keep the original w value intact, because it might be used
        // by the application.
        vec3 n             = make_vec3(0);
        vertices[i].normal = make_vec4(n, vertices[i].normal[3]);
    }

    //#pragma omp parallel for
    for (int i = 0; i < (int)faces.size(); ++i)
    {
        Face& f = faces[i];
        vec3 a  = make_vec3(vertices[f(0)].position);
        vec3 b  = make_vec3(vertices[f(1)].position);
        vec3 c  = make_vec3(vertices[f(2)].position);
        vec3 n  = cross(b - a, c - a);
        // Note: do not normalize here because the length is the surface area
        vertices[f(0)].normal += make_vec4(n, 0);
        vertices[f(1)].normal += make_vec4(n, 0);
        vertices[f(2)].normal += make_vec4(n, 0);
    }

    //#pragma omp parallel for
    for (int i = 0; i < (int)vertices.size(); ++i)
    {
        vec3 n             = normalize(make_vec3(vertices[i].normal));
        vertices[i].normal = make_vec4(n, vertices[i].normal[3]);
    }
}

template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::removeUnusedVertices()
{
    std::vector<int> vmap(vertices.size(), -1);
    auto vcopy = vertices;
    vertices.clear();
    for (int i = 0; i < (int)faces.size(); ++i)
    {
        auto& f = faces[i];

        for (int i = 0; i < 3; ++i)
        {
            auto& v = f[i];
            if (vmap[v] == -1)
            {
                int count = vertices.size();
                vmap[v]   = count;
                vertices.push_back(vcopy[v]);
            }
            v = vmap[v];
        }
    }
}



template <typename vertex_t, typename index_t>
bool TriangleMesh<vertex_t, index_t>::isValid() const
{
    // check if all referenced vertices exist
    for (Face f : faces)
    {
        if (f(0) < 0 || f(0) >= (int)vertices.size()) return false;
        if (f(1) < 0 || f(1) >= (int)vertices.size()) return false;
        if (f(2) < 0 || f(2) >= (int)vertices.size()) return false;
    }
    return true;
}

template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::sortVerticesByPosition(double epsilon)
{
    double eps_squared = epsilon * epsilon;

    std::vector<int> tmp_indices(vertices.size());
    std::vector<int> tmp_indices2(vertices.size());
    std::iota(tmp_indices.begin(), tmp_indices.end(), 0);

    std::sort(tmp_indices.begin(), tmp_indices.end(), [&](int a, int b) {
        vec4 p1 = vertices[a].position;
        vec4 p2 = vertices[b].position;

        vec4 p3 = (p1 - p2);
        p3      = p3.array() * p3.array();

        p1[0] = p3[0] < eps_squared ? p2[0] : p1[0];
        p1[1] = p3[1] < eps_squared ? p2[1] : p1[1];
        p1[2] = p3[2] < eps_squared ? p2[2] : p1[2];

        return std::tie(p1[0], p1[1], p1[2]) < std::tie(p2[0], p2[1], p2[2]);
    });

    std::vector<vertex_t> new_vertices(vertices.size());
    for (int i = 0; i < (int)new_vertices.size(); ++i)
    {
        new_vertices[i]              = vertices[tmp_indices[i]];
        tmp_indices2[tmp_indices[i]] = i;
    }

    for (auto& f : faces)
    {
        f(0) = tmp_indices2[f(0)];
        f(1) = tmp_indices2[f(1)];
        f(2) = tmp_indices2[f(2)];
    }
    vertices.swap(new_vertices);
}

template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::removeSubsequentDuplicates(double epsilon)
{
    if (vertices.size() <= 1) return;

    double eps_squared = epsilon * epsilon;

    std::vector<int> tmp_indices(vertices.size());
    std::vector<bool> valid(vertices.size(), false);
    std::vector<vertex_t> new_vertices;

    int currentIdx = -1;
    vec4 currentPos;

    for (int i = 0; i < (int)vertices.size(); ++i)
    {
        auto& p = vertices[i].position;
        if (i == 0 || (p - currentPos).squaredNorm() > eps_squared)
        {
            new_vertices.push_back(vertices[i]);
            currentIdx++;
            currentPos = p;
            valid[i]   = true;
        }
        tmp_indices[i] = currentIdx;
    }

    for (int i = 0; i < (int)vertices.size(); ++i)
    {
        if (valid[i]) new_vertices[tmp_indices[i]] = vertices[i];
    }

    for (auto& f : faces)
    {
        f(0) = tmp_indices[f(0)];
        f(1) = tmp_indices[f(1)];
        f(2) = tmp_indices[f(2)];
    }
    vertices.swap(new_vertices);
}

template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::removeDegenerateFaces()
{
    faces.erase(std::remove_if(faces.begin(), faces.end(),
                               [](const Face& f) { return f(0) == f(1) || f(0) == f(2) || f(1) == f(2); }),
                faces.end());
}

template <typename vertex_t, typename index_t>
float TriangleMesh<vertex_t, index_t>::distancePointMesh(const vec3& x)
{
    float dis = std::numeric_limits<float>::infinity();

    for (const Face& f : faces)
    {
        Triangle t;
        t.a = make_vec3(vertices[f(0)].position);
        t.b = make_vec3(vertices[f(1)].position);
        t.c = make_vec3(vertices[f(2)].position);
        dis = std::min(dis, t.Distance(x));
    }
    return dis;
}

template <typename vertex_t, typename index_t>
std::ostream& operator<<(std::ostream& os, const TriangleMesh<vertex_t, index_t>& dt)
{
    os << "TriangleMesh. V=" << dt.vertices.size() << " F=" << dt.faces.size();
    return os;
}


template <typename vertex_t, typename index_t>
void saveMeshOff(const TriangleMesh<vertex_t, index_t>& mesh, std::ostream& strm)
{
    strm << "OFF"
         << "\n";
    // first line: number of vertices, number of faces, number of edges (can be ignored)
    strm << mesh.vertices.size() << " " << mesh.faces.size() << " 0"
         << "\n";

    for (auto const& v : mesh.vertices)
    {
        strm << v.position[0] << " " << v.position[1] << " " << v.position[2] << "\n";
    }

    for (auto const& f : mesh.faces)
    {
        strm << "3"
             << " " << f[0] << " " << f[1] << " " << f[2] << "\n";
    }
}

template <typename vertex_t, typename index_t>
void saveMeshOffColor(const TriangleMesh<vertex_t, index_t>& mesh, std::ostream& strm)
{
    strm << "COFF"
         << "\n";
    // first line: number of vertices, number of faces, number of edges (can be ignored)
    strm << mesh.vertices.size() << " " << mesh.faces.size() << " 0"
         << "\n";

    for (auto const& v : mesh.vertices)
    {
        strm << v.position[0] << " " << v.position[1] << " " << v.position[2] << " ";
        strm << v.color[0] << " " << v.color[1] << " " << v.color[2] << " " << v.color[3] << "\n";
    }

    for (auto const& f : mesh.faces)
    {
        strm << "3"
             << " " << f[0] << " " << f[1] << " " << f[2] << "\n";
    }
}


}  // namespace Saiga
