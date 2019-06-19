/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/aabb.h"
#include "saiga/core/geometry/triangle.h"
#include "saiga/core/geometry/vertex.h"
#include "saiga/core/util/assert.h"
#include "saiga/core/math/math.h"

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
class TriangleMesh
{
   public:
    struct SAIGA_ALIGN(4) Face
    {
        index_t v1, v2, v3;
        Face() {}
        Face(const index_t& v1, const index_t& v2, const index_t& v3) : v1(v1), v2(v2), v3(v3) {}
        index_t& operator[](int idx)
        {
            // assume index_t alignment
            return *((&v1) + idx);
        }

        const index_t& operator[](int idx) const
        {
            // assume index_t alignment
            return *((&v1) + idx);
        }
    };

    //    typedef IndexedVertexBuffer<vertex_t,index_t> buffer_t;



    /*
     * Create empty triangle mesh
     */

    TriangleMesh(void);
    ~TriangleMesh(void) {}

    /*
     * Transforms mesh with given matrix.
     * All vertices are multiplied with 'trafo'
     */

    void transform(const mat4& trafo);
    void transformNormal(const mat4& trafo);

    /*
     * Deletes all vertices and faces.
     */

    void clear()
    {
        vertices.resize(0);
        faces.resize(0);
    }

    /*
     * Adds vertex to mesh and updates enclosing AABB.
     * return: index of new vertex
     */

    int addVertex(const vertex_t& v)
    {
        vertices.push_back(v);
        boundingBox.growBox(make_vec3(v.position));
        return vertices.size() - 1;
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

    void toTriangleList(std::vector<Triangle>& output);

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
    size_t size();
    void free();

    int numIndices() { return faces.size() * 3; }

    AABB calculateAabb();

    bool isValid();

    /**
     * Sorts the vertices by (x,y,z) lexical.
     * The face indices are correct to match the new vertices.
     */
    void sortVerticesByPosition();


    /**
     * Removes subsequent vertices if they have identical position.
     * It make sense to call it after 'sortVerticesByPosition'.
     *
     * The face indices are updated accordingly.
     */
    void removeSubsequentDuplicates();

    /**
     * Removes all triangles, which reference a vertex twice
     */
    void removeDegenerateFaces();

    template <typename v, typename i>
    friend std::ostream& operator<<(std::ostream& os, const TriangleMesh<v, i>& dt);

    AABB& getAabb() { return boundingBox; }

    std::vector<index_t> getIndexList()
    {
        std::vector<index_t> indices(numIndices());
        std::copy(&faces[0].v1, &faces[0].v1 + numIndices(), indices.data());
        return indices;
    }

    /**
     * Writes this mesh in OFF format to the given output stream.
     */
    void saveMeshOff(std::ostream& strm);

   public:
    std::vector<vertex_t> vertices;
    std::vector<Face> faces;
    AABB boundingBox;
};



template <typename vertex_t, typename index_t>
TriangleMesh<vertex_t, index_t>::TriangleMesh(void)
{
    boundingBox.makeNegative();
}

template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::transform(const mat4& trafo)
{
    for (vertex_t& v : vertices)
    {
        // Make sure it works for user defined w components
        vec4 p     = make_vec4(make_vec3(v.position), 1);
        p          = trafo * p;
        v.position = make_vec4(make_vec3(p), v.position[3]);
    }
    boundingBox.transform(trafo);
}

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
void TriangleMesh<vertex_t, index_t>::addQuad(index_t inds[])
{
    faces.push_back(Face(inds[0], inds[1], inds[2]));
    faces.push_back(Face(inds[2], inds[3], inds[0]));
}



template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::subdivideFace(int f)
{
    Face face = faces[f];

#define _TM_POS(xs) (vertices[face.xs].position)
    // create 3 new vertices in the middle of the edges

    int v1 = addVertex(vertex_t(vec4((_TM_POS(v1) + _TM_POS(v2)) / 2.0f)));
    int v2 = addVertex(vertex_t(vec4((_TM_POS(v1) + _TM_POS(v3)) / 2.0f)));
    int v3 = addVertex(vertex_t(vec4((_TM_POS(v2) + _TM_POS(v3)) / 2.0f)));


    faces.push_back(Face(face.v2, v3, v1));

    faces.push_back(Face(face.v3, v2, v3));

    faces.push_back(Face(v1, v3, v2));
    faces[f] = Face(face.v1, v1, v2);
}



template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::invertFace(int f)
{
    Face& face = faces[f];
    Face face2;
    face2.v1 = face.v3;
    face2.v2 = face.v2;
    face2.v3 = face.v1;
    face     = face2;
}

template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::invertMesh()
{
    for (Face& face : faces)
    {
        Face face2;
        face2.v1 = face.v3;
        face2.v2 = face.v2;
        face2.v3 = face.v1;
        face     = face2;
    }

    for (vertex_t& v : vertices)
    {
        v.normal = -v.normal;
    }
}

template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::toTriangleList(std::vector<Triangle>& output)
{
    Triangle t;
    for (Face& f : faces)
    {
        t.a = make_vec3(vertices[f.v1].position);
        t.b = make_vec3(vertices[f.v2].position);
        t.c = make_vec3(vertices[f.v3].position);
        output.push_back(t);
    }
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
        f.v1 += oldVertexCount;
        f.v2 += oldVertexCount;
        f.v3 += oldVertexCount;
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
        f.v1 += oldVertexCount;
        f.v2 += oldVertexCount;
        f.v3 += oldVertexCount;
        this->addFace(f.v1, f.v2, f.v3);
    }
}

template <typename vertex_t, typename index_t>
size_t TriangleMesh<vertex_t, index_t>::size()
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
        vec3 a  = make_vec3(vertices[f.v1].position);
        vec3 b  = make_vec3(vertices[f.v2].position);
        vec3 c  = make_vec3(vertices[f.v3].position);
        vec3 n  = cross(b - a, c - a);
        // Note: do not normalize here because the length is the surface area
        vertices[f.v1].normal += make_vec4(n, 0);
        vertices[f.v2].normal += make_vec4(n, 0);
        vertices[f.v3].normal += make_vec4(n, 0);
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
AABB TriangleMesh<vertex_t, index_t>::calculateAabb()
{
    boundingBox.makeNegative();

    for (vertex_t& v : vertices)
    {
        boundingBox.growBox(make_vec3(v.position));
    }
    return boundingBox;
}

template <typename vertex_t, typename index_t>
bool TriangleMesh<vertex_t, index_t>::isValid()
{
    // check if all referenced vertices exist
    for (Face f : faces)
    {
        if (f.v1 < 0 || f.v1 >= vertices.size()) return false;
        if (f.v2 < 0 || f.v2 >= vertices.size()) return false;
        if (f.v3 < 0 || f.v3 >= vertices.size()) return false;
    }
    return true;
}

template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::sortVerticesByPosition()
{
    std::vector<int> tmp_indices(vertices.size());
    std::vector<int> tmp_indices2(vertices.size());
    std::iota(tmp_indices.begin(), tmp_indices.end(), 0);

    std::sort(tmp_indices.begin(), tmp_indices.end(), [&](int a, int b) {
        auto p1 = vertices[a].position;
        auto p2 = vertices[b].position;

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
        f.v1 = tmp_indices2[f.v1];
        f.v2 = tmp_indices2[f.v2];
        f.v3 = tmp_indices2[f.v3];
    }
    vertices.swap(new_vertices);
}

template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::removeSubsequentDuplicates()
{
    if (vertices.size() <= 1) return;

    std::vector<int> tmp_indices(vertices.size());
    std::vector<bool> valid(vertices.size(), false);
    std::vector<vertex_t> new_vertices;

    int currentIdx = -1;
    vec4 currentPos;

    for (int i = 0; i < vertices.size(); ++i)
    {
        auto p = vertices[i].position;
        if (p != currentPos || i == 0)
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
        f.v1 = tmp_indices[f.v1];
        f.v2 = tmp_indices[f.v2];
        f.v3 = tmp_indices[f.v3];
    }
    vertices.swap(new_vertices);
}

template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::removeDegenerateFaces()
{
    faces.erase(std::remove_if(faces.begin(), faces.end(),
                               [](const Face& f) { return f.v1 == f.v2 || f.v1 == f.v3 || f.v2 == f.v3; }),
                faces.end());
}

template <typename vertex_t, typename index_t>
void TriangleMesh<vertex_t, index_t>::saveMeshOff(std::ostream& strm)
{
    strm << "OFF" << std::endl;
    // first line: number of vertices, number of faces, number of edges (can be ignored)
    strm << vertices.size() << " " << faces.size() << " 0" << std::endl;

    for (auto& v : vertices)
    {
        strm << v.position[0] << " " << v.position[1] << " " << v.position[2] << std::endl;
    }

    for (auto& f : faces)
    {
        strm << "3"
             << " " << f[0] << " " << f[1] << " " << f[2] << std::endl;
    }
}

template <typename vertex_t, typename index_t>
std::ostream& operator<<(std::ostream& os, const TriangleMesh<vertex_t, index_t>& dt)
{
    os << "TriangleMesh. V=" << dt.vertices.size() << " F=" << dt.faces.size();
    return os;
}

}  // namespace Saiga
