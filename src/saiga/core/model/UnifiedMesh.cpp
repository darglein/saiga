/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "UnifiedMesh.h"

#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/tostring.h"

#include "internal/noGraphicsAPI.h"

#include "model_loader_obj.h"
#include "model_loader_ply.h"

namespace Saiga
{

UnifiedMesh& UnifiedMesh::transform(const mat4& T)
{
    if (HasPosition())
    {
        for (auto& p : position)
        {
            p = (T * make_vec4(p, 1)).head<3>();
        }
    }
    if (HasNormal())
    {
        for (auto& n : normal)
        {
            n = (T * make_vec4(n, 0)).head<3>();
        }
    }
    return *this;
}

UnifiedMesh& UnifiedMesh::SetVertexColor(const vec4& c)
{
    color.resize(position.size());
    for (auto& co : color)
    {
        co = c;
    }
    return *this;
}

UnifiedMesh& UnifiedMesh::FlipNormals()
{
    for (auto& n : normal)
    {
        n = -n;
    }
    return *this;
}

UnifiedMesh& UnifiedMesh::FlatShading()
{
    auto flatten = [this](auto old) {
        decltype(old) flat;
        for (auto& tri : triangles)
        {
            flat.push_back(old[tri(0)]);
            flat.push_back(old[tri(1)]);
            flat.push_back(old[tri(2)]);
        }
        return flat;
    };

    if (!position.empty()) position = flatten(position);
    if (!normal.empty()) normal = flatten(normal);
    if (!color.empty()) color = flatten(color);
    if (!texture_coordinates.empty()) texture_coordinates = flatten(texture_coordinates);
    if (!data.empty()) data = flatten(data);
    if (!bone_info.empty()) bone_info = flatten(bone_info);

    std::vector<ivec3> flat_triangles;
    for (int i = 0; i < triangles.size(); ++i)
    {
        flat_triangles.push_back(ivec3(i * 3, i * 3 + 1, i * 3 + 2));
    }
    triangles = flat_triangles;

    CalculateVertexNormals();

    return *this;
}

UnifiedMesh& UnifiedMesh::EraseVertices(ArrayView<int> vertices)
{
    SAIGA_ASSERT(triangles.empty());
    SAIGA_ASSERT(lines.empty());

    std::vector<int> valid_vertex(NumVertices(), 1);
    for (auto v : vertices)
    {
        valid_vertex[v] = 0;
    }



    auto erase = [&](auto old) {
        decltype(old) flat;
        for (int i = 0; i < NumVertices(); ++i)
        {
            if (valid_vertex[i])
            {
                flat.push_back(old[i]);
            }
        }
        return flat;
    };

    if (!position.empty()) position = erase(position);
    if (!normal.empty()) normal = erase(normal);
    if (!color.empty()) color = erase(color);
    if (!texture_coordinates.empty()) texture_coordinates = erase(texture_coordinates);
    if (!data.empty()) data = erase(data);
    if (!bone_info.empty()) bone_info = erase(bone_info);
    return *this;
}

UnifiedMesh& UnifiedMesh::Normalize(float dimensions)
{
    auto box = BoundingBox();
    float s  = dimensions / box.maxSize();
    vec3 p   = box.getPosition();

    mat4 S = scale(vec3(s, s, s));
    mat4 T = translate(-p);



    return transform(S * T);
}

AABB UnifiedMesh::BoundingBox() const
{
    AABB box;
    box.makeNegative();
    for (auto& p : position)
    {
        box.growBox(p);
    }
    return box;
}


std::vector<Triangle> UnifiedMesh::TriangleSoup() const
{
    std::vector<Triangle> result;
    for (auto t : triangles)
    {
        Triangle tri;
        tri.a = position[t(0)];
        tri.b = position[t(1)];
        tri.c = position[t(2)];
        result.push_back(tri);
    }
    return result;
}


UnifiedMesh& UnifiedMesh::CalculateVertexNormals()
{
    normal.resize(position.size());
    std::fill(normal.begin(), normal.end(), vec3(0, 0, 0));

    for (auto& tri : triangles)
    {
        vec3 n = cross(position[tri(1)] - position[tri(0)], position[tri(2)] - position[tri(0)]);
        normal[tri(0)] += n;
        normal[tri(1)] += n;
        normal[tri(2)] += n;
    }

    for (auto& n : normal)
    {
        n.normalize();
    }

    return *this;
}


template <>
std::vector<Vertex> UnifiedMesh::VertexList() const
{
    SAIGA_ASSERT(HasPosition());



    std::vector<Vertex> mesh;

    mesh.resize(NumVertices());
    for (int i = 0; i < NumVertices(); ++i)
    {
        mesh[i].position = make_vec4(position[i], 1);
    }


    return mesh;
}


template <>
std::vector<VertexC> UnifiedMesh::VertexList() const
{
    SAIGA_ASSERT(HasPosition());


    std::vector<VertexC> mesh;


    mesh.resize(NumVertices());
    for (int i = 0; i < NumVertices(); ++i)
    {
        mesh[i].position = make_vec4(position[i], 1);
    }

    if (HasColor())
    {
        for (int i = 0; i < NumVertices(); ++i)
        {
            mesh[i].color = color[i];
        }
    }
    else
    {
        for (int i = 0; i < NumVertices(); ++i)
        {
            mesh[i].color = vec4(1, 1, 1, 1);
        }
    }
    return mesh;
}



template <>
std::vector<VertexNC> UnifiedMesh::VertexList() const
{
    SAIGA_ASSERT(HasPosition());


    std::vector<VertexNC> mesh;


    mesh.resize(NumVertices());
    for (int i = 0; i < NumVertices(); ++i)
    {
        mesh[i].position = make_vec4(position[i], 1);
    }

    if (HasColor())
    {
        for (int i = 0; i < NumVertices(); ++i)
        {
            mesh[i].color = color[i];
        }
    }
    else
    {
        for (int i = 0; i < NumVertices(); ++i)
        {
            mesh[i].color = vec4(1, 1, 1, 1);
        }
    }

    if (HasNormal())
    {
        for (int i = 0; i < NumVertices(); ++i)
        {
            mesh[i].normal = make_vec4(normal[i], 0);
        }
    }
    return mesh;
}


template <>
std::vector<VertexNT> UnifiedMesh::VertexList() const
{
    SAIGA_ASSERT(HasPosition());
    SAIGA_ASSERT(HasTC());

    std::vector<VertexNT> mesh;


    mesh.resize(NumVertices());
    for (int i = 0; i < NumVertices(); ++i)
    {
        mesh[i].position = make_vec4(position[i], 1);
        mesh[i].texture  = texture_coordinates[i];
    }

    if (HasNormal())
    {
        for (int i = 0; i < NumVertices(); ++i)
        {
            mesh[i].normal = make_vec4(normal[i], 0);
        }
    }

    return mesh;
}

template <>
std::vector<VertexNTD> UnifiedMesh::VertexList() const
{
    SAIGA_ASSERT(HasPosition());
    SAIGA_ASSERT(HasTC());


    std::vector<VertexNTD> mesh;

    mesh.resize(NumVertices());
    for (int i = 0; i < NumVertices(); ++i)
    {
        mesh[i].position = make_vec4(position[i], 1);
        mesh[i].texture  = texture_coordinates[i];
    }

    if (HasNormal())
    {
        for (int i = 0; i < NumVertices(); ++i)
        {
            mesh[i].normal = make_vec4(normal[i], 0);
        }
    }

    if (HasData())
    {
        for (int i = 0; i < NumVertices(); ++i)
        {
            mesh[i].data = data[i];
        }
    }

    return mesh;
}


template <>
std::vector<BoneVertexCD> UnifiedMesh::VertexList() const
{
    std::vector<BoneVertexCD> mesh;

    SAIGA_ASSERT(HasPosition());
    SAIGA_ASSERT(HasBones());


    mesh.resize(NumVertices());
    for (int i = 0; i < NumVertices(); ++i)
    {
        mesh[i].position = make_vec4(position[i], 1);
    }

    for (int i = 0; i < NumVertices(); ++i)
    {
        mesh[i].bone_info = bone_info[i];
        mesh[i].bone_info.normalizeWeights();
    }

    if (HasColor())
    {
        for (int i = 0; i < NumVertices(); ++i)
        {
            mesh[i].color = color[i];
        }
    }
    else
    {
        for (int i = 0; i < NumVertices(); ++i)
        {
            mesh[i].color = vec4(1, 1, 1, 1);
        }
    }

    if (HasNormal())
    {
        for (int i = 0; i < NumVertices(); ++i)
        {
            mesh[i].normal = make_vec4(normal[i], 0);
        }
    }


    return mesh;
}


}  // namespace Saiga
